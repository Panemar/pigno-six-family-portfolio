from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from portfolio_operators import GraphGalerkinOperator, HistoricalCapacityDataset  # noqa: E402

PIGNO=ROOT.parent; V4=PIGNO/"structure_preserving_pigno_v4"; D=V4/"s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA=D/"S8_CAPACITY_FULL_DT_DATASET.h5"; GRAPH=D/"S8_GRAPH_INPUTS.npz"
VAR=V4/"s8_physical32_variational_residual_preflight_V40_A_E6_C10_1T_v2"/"S8_PHYSICAL32_VARIATIONAL_PREFLIGHT.h5"
PROTOCOL=ROOT/"s6_capacity_common"/"SIX_ROUTE_CAPACITY_PROTOCOL.json"; WITNESS=ROOT/"s6_capacity_common"/"R3_ELEMENTWISE_VARIATIONAL_WITNESS.json"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def atomic(path,obj):
    p=Path(path);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(obj,indent=2),encoding="utf-8");os.replace(t,p)
def scl(x,axes):
    y=np.sqrt(np.mean(np.square(x),axis=axes));pos=y[y>0];floor=max(float(np.median(pos))*1e-3 if pos.size else 0,1e-10);return np.maximum(y,floor).astype(np.float32)
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).eps))
def gv(loss,pars):
    gs=torch.autograd.grad(loss,pars,retain_graph=True,allow_unused=True);return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(pars,gs)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--ablation",choices=("data_only","physics_informed"),required=True);ap.add_argument("--epochs",type=int,default=None);ap.add_argument("--optimization-repair",choices=("none","constant_after_warmup"),default="none");ap.add_argument("--run-revision",default="V1");args=ap.parse_args()
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8")); witness=json.loads(WITNESS.read_text(encoding="utf-8"))
    if not witness["decision"]["capacity_training_authorized"]:raise RuntimeError("R3 structural witness blocks capacity")
    epochs=150 if args.epochs is None else args.epochs
    suffix="" if epochs==150 else f"_SMOKE_E{epochs}";opt_suffix="" if args.optimization_repair=="none" else "_OPT_CONSTANT_LR";run=f"S6_R3_GRAPH_GALERKIN_CAPACITY_{args.ablation.upper()}_REP_PETROV_PHYSICAL32{opt_suffix}{suffix}_{args.run_revision}";out=ROOT/"s6_capacity_runs"/run
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    seed=int(protocol["common_budget"]["seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():raise RuntimeError("cuda required")
    dev=torch.device("cuda:0");data=HistoricalCapacityDataset(DATA,GRAPH);basis=data.observation_basis().astype(np.float32)
    qfield=np.einsum("ndr,tr->tnd",basis,data.q,optimize=True).astype(np.float32);vfield=np.einsum("ndr,tr->tnd",basis,data.qdot,optimize=True).astype(np.float32)
    qfield[:,:,:3]=data.translation;vfield[:,:,:3]=data.velocity; fixed=data.fixed_dof[data.observation_node]
    if max(np.max(np.abs(qfield[:,fixed])),np.max(np.abs(vfield[:,fixed])))>1e-10:raise RuntimeError("target BC")
    node=data.graph_node_features.astype(np.float32);edge=data.edge_attr.astype(np.float32);node=(node-node.mean(0))/np.maximum(node.std(0),1e-8);edge=(edge-edge.mean(0))/np.maximum(edge.std(0),1e-8)
    temporal=np.c_[data.global_series,data.reduced_force.astype(np.float32),data.time_s.astype(np.float32)/data.time_s[-1]];temporal=((temporal-temporal.mean(0))/np.maximum(temporal.std(0),np.maximum(np.max(np.abs(temporal),axis=0)*1e-6,1e-8))).astype(np.float32)
    load=data.load_node_force.astype(np.float32)/scl(data.load_node_force,(0,1));qs=scl(data.q,0);vs=scl(data.qdot,0)
    with h5py.File(VAR,"r") as h:
        pt=h["time_s"][:];M=h["operator/M"][:];C=h["operator/C"][:];K=h["operator/K"][:];force=h["force/prescribed"][:].T;acc=h["state/qddot_direct_FEM_COMSOL_panel"][:].T
    panel=np.array([np.argmin(np.abs(data.time_s-t)) for t in pt]);as_=scl(acc,0);fs=scl(force,0)
    T={"node":torch.tensor(node,device=dev),"edge":torch.tensor(edge,device=dev),"ei":torch.tensor(data.edge_index,device=dev,dtype=torch.long),"temporal":torch.tensor(temporal[None],device=dev),"load":torch.tensor(load[None],device=dev),"ln":torch.tensor(data.load_node,device=dev,dtype=torch.long),"basis":torch.tensor(basis,device=dev),"free":torch.tensor((~fixed)[None,None],device=dev,dtype=torch.float32),"qt":torch.tensor(data.q[None],device=dev,dtype=torch.float32),"vt":torch.tensor(data.qdot[None],device=dev,dtype=torch.float32),"qft":torch.tensor(qfield[None],device=dev),"vft":torch.tensor(vfield[None],device=dev),"qs":torch.tensor(qs,device=dev),"vs":torch.tensor(vs,device=dev),"as":torch.tensor(as_,device=dev),"panel":torch.tensor(panel,device=dev,dtype=torch.long),"M":torch.tensor(M,device=dev,dtype=torch.float32),"C":torch.tensor(C,device=dev,dtype=torch.float32),"K":torch.tensor(K,device=dev,dtype=torch.float32),"force":torch.tensor(force,device=dev,dtype=torch.float32),"fs":torch.tensor(fs,device=dev)}
    qfscale=torch.tensor(scl(qfield,(0,1)),device=dev);vfscale=torch.tensor(scl(vfield,(0,1)),device=dev)
    model=GraphGalerkinOperator(node.shape[1],edge.shape[1],temporal.shape[1]).to(dev);pars=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(pars,lr=8e-4,weight_decay=1e-5);physics=args.ablation=="physics_informed";pw=0.0
    cols=["epoch","elapsed_s","lr","loss","qfield_loss","vfield_loss","q32_loss","v32_loss","weak_loss","physics_weight","data_grad_norm","physics_grad_norm","gradient_cosine","gradient_norm","score","peak_vram_GiB","displacement_X_relative_l2","displacement_Y_relative_l2","displacement_Z_relative_l2","velocity_X_relative_l2","velocity_Y_relative_l2","velocity_Z_relative_l2","rotation_X_relative_l2","rotation_Y_relative_l2","rotation_Z_relative_l2","rotation_rate_X_relative_l2","rotation_rate_Y_relative_l2","rotation_rate_Z_relative_l2","physical_q_relative_l2","physical_qdot_relative_l2","weak_median","weak_p90","hard_BC_max_abs","finite"]
    with (out/"live_progress.csv").open("w",newline="",encoding="utf-8") as f:csv.DictWriter(f,fieldnames=cols).writeheader()
    def event(name,**kw):
        with (out/"RUN_LOG.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),"event":name,**kw})+"\n")
    event("run_started",run_id=run,parameters=sum(p.numel() for p in pars),script_sha256=sha(Path(__file__)),elementwise_witness_status=witness["status"])
    atomic(out/"status.json",{"status":"RUNNING","run_id":run,"epoch":0,"maximum_epochs":epochs,"HPO_authorized":False})
    best=float("inf");be=0;e0=None;start=time.perf_counter()
    def forward(temporal=T["temporal"],load=T["load"]):
        o=model(T["node"],T["ei"],T["edge"],temporal,load,T["ln"]);q=o["q_normalized"]*T["qs"];v=o["v_normalized"]*T["vs"];a=o["a_physical_normalized"]*T["as"];qf=torch.einsum("ndr,btr->btnd",T["basis"],q)*T["free"];vf=torch.einsum("ndr,btr->btnd",T["basis"],v)*T["free"];o.update(q=q,v=v,a=a,qfield=qf,vfield=vf);return o
    def weak(o):
        q=o["q"][0,T["panel"],:32];v=o["v"][0,T["panel"],:32];a=o["a"][0,T["panel"]];r=a@T["M"].T+v@T["C"].T+q@T["K"].T-T["force"];return torch.mean((r/T["fs"]).square()),r
    def measure(o):
        q=o["q"].detach().cpu().numpy()[0];v=o["v"].detach().cpu().numpy()[0];qf=o["qfield"].detach().cpu().numpy()[0];vf=o["vfield"].detach().cpu().numpy()[0];_,r=weak(o);r=r.detach().cpu().numpy();rat=np.linalg.norm(r,axis=1)/np.maximum(np.linalg.norm(force,axis=1),np.finfo(float).eps)
        m={"physical_q_relative_l2":rel(q[:,:32],data.q[:,:32]),"physical_qdot_relative_l2":rel(v[:,:32],data.qdot[:,:32])}
        for i,a in enumerate("XYZ"):m[f"displacement_{a}_relative_l2"]=rel(qf[:,:,i],qfield[:,:,i]);m[f"velocity_{a}_relative_l2"]=rel(vf[:,:,i],vfield[:,:,i]);m[f"rotation_{a}_relative_l2"]=rel(qf[:,:,i+3],qfield[:,:,i+3]);m[f"rotation_rate_{a}_relative_l2"]=rel(vf[:,:,i+3],vfield[:,:,i+3])
        m.update(weak_median=float(np.median(rat)),weak_p90=float(np.percentile(rat,90)),hard_BC_max_abs=float(max(np.max(np.abs(qf[:,fixed])),np.max(np.abs(vf[:,fixed])))));m["finite"]=all(np.isfinite(x) for x in m.values());return m,q,v,qf,vf,r
    for epoch in range(epochs+1):
        model.train();o=forward();lq=torch.mean(((o["qfield"]-T["qft"])/qfscale).square());lv=torch.mean(((o["vfield"]-T["vft"])/vfscale).square());sq=torch.mean(((o["q"][:,:,:32]-T["qt"][:,:,:32])/T["qs"][:32]).square());sv=torch.mean(((o["v"][:,:,:32]-T["vt"][:,:,:32])/T["vs"][:32]).square());dl=lq+lv+.25*(sq+sv);pl,_=weak(o);dg=gv(dl,pars);dn=float(torch.linalg.vector_norm(dg).cpu());pn=cos=float("nan")
        if physics:
            pg=gv(pl,pars);pn=float(torch.linalg.vector_norm(pg).cpu());cos=float((torch.dot(dg,pg)/(torch.linalg.vector_norm(dg)*torch.linalg.vector_norm(pg)).clamp_min(1e-20)).cpu());prop=max(1e-5,min(.05,.2*dn/max(pn,1e-20)));pw=prop if epoch==0 else .9*pw+.1*prop
        loss=dl+pw*pl;gn=0.0
        if epoch>0:
            opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(pars,1.0).cpu());warm=5;lr=8e-4*epoch/warm if epoch<=warm else (8e-4 if args.optimization_repair=="constant_after_warmup" else 8e-4*(.1+.9*.5*(1+math.cos(math.pi*(epoch-warm)/max(epochs-warm,1)))));[g.update(lr=lr) for g in opt.param_groups];opt.step()
        else:lr=0.0
        if epoch in {0,1,5,10,epochs} or epoch%5==0:
            model.eval();
            with torch.no_grad():ev=forward()
            m,*pred=measure(ev);score=sum(m[f"displacement_{a}_relative_l2"] for a in "XYZ")+.5*sum(m[f"velocity_{a}_relative_l2"] for a in "XYZ")+m["physical_q_relative_l2"]+.5*m["physical_qdot_relative_l2"]
            if e0 is None:e0=m
            if score<best:best=score;be=epoch;torch.save({"model_state":model.state_dict(),"epoch":epoch,"score":score,"metrics":m},out/"best_checkpoint.pt")
            row={"epoch":epoch,"elapsed_s":time.perf_counter()-start,"lr":lr,"loss":float(loss.detach().cpu()),"qfield_loss":float(lq.detach().cpu()),"vfield_loss":float(lv.detach().cpu()),"q32_loss":float(sq.detach().cpu()),"v32_loss":float(sv.detach().cpu()),"weak_loss":float(pl.detach().cpu()),"physics_weight":pw,"data_grad_norm":dn,"physics_grad_norm":pn,"gradient_cosine":cos,"gradient_norm":gn,"score":score,"peak_vram_GiB":torch.cuda.max_memory_allocated()/2**30,**m}
            with (out/"live_progress.csv").open("a",newline="",encoding="utf-8") as f:csv.DictWriter(f,fieldnames=cols).writerow(row)
            atomic(out/"status.json",{"status":"RUNNING","run_id":run,"epoch":epoch,"maximum_epochs":epochs,"best_epoch":be,"best_score":best,"current_metrics":m,"HPO_authorized":False});event("evaluation",epoch=epoch,score=score,metrics=m)
    ck=torch.load(out/"best_checkpoint.pt",map_location=dev,weights_only=True);model.load_state_dict(ck["model_state"]);model.eval()
    with torch.no_grad():final=forward()
    fm,q,v,qf,vf,res=measure(final);cut=600;tp=T["temporal"].clone();lp=T["load"].clone();tp[:,cut+1:]+=.1*torch.randn_like(tp[:,cut+1:]);lp[:,cut+1:]+=.1*torch.randn_like(lp[:,cut+1:]);
    with torch.no_grad():future=forward(tp,lp);zero_graph=forward(T["temporal"],torch.zeros_like(T["load"]))
    causal=float(torch.max(torch.abs(final["qfield"][:,:cut+1]-future["qfield"][:,:cut+1])).cpu());graph_sensitivity=float(torch.linalg.vector_norm(final["qfield"]-zero_graph["qfield"]).cpu()/torch.linalg.vector_norm(final["qfield"]).clamp_min(1e-20).cpu())
    lim=protocol["one_case_diagnostic_thresholds_not_final_utility"];vel=[fm[f"velocity_{a}_relative_l2"] for a in "XYZ"];gates={"finite":fm["finite"],"displacement_each_axis":max(fm[f"displacement_{a}_relative_l2"] for a in "XYZ")<=.05,"physical_q":fm["physical_q_relative_l2"]<=.05,"velocity_median":float(np.median(vel))<=.25,"velocity_worst":max(vel)<=.4,"physical_qdot":fm["physical_qdot_relative_l2"]<=.3,"hard_BC":fm["hard_BC_max_abs"]<=1e-12,"causal":causal<=1e-7,"graph_sensitivity":graph_sensitivity>1e-6}
    if physics:gates.update(weak_median=fm["weak_median"]<=.05,weak_p90=fm["weak_p90"]<=.10)
    passed=all(gates.values());status="PASS_S6_R3_ONE_CASE_CAPACITY" if passed else "REPAIR_REQUIRED_S6_R3_ONE_CASE_CAPACITY"
    with h5py.File(out/"best_prediction.h5","w") as h:h.attrs["run_id"]=run;h.create_dataset("time_s",data=data.time_s);h.create_dataset("prediction/q",data=q,compression="gzip");h.create_dataset("prediction/qdot",data=v,compression="gzip");h.create_dataset("prediction/sixdof",data=qf,compression="gzip");h.create_dataset("prediction/sixdof_rate",data=vf,compression="gzip");h.create_dataset("reference/sixdof",data=qfield,compression="gzip");h.create_dataset("reference/sixdof_rate",data=vfield,compression="gzip");h.create_dataset("diagnostic/weak_residual",data=res)
    repairs=["representation_Petrov_Galerkin"]+(["optimization_constant_learning_rate_after_warmup"] if args.optimization_repair!="none" else []);report={"status":status,"run_id":run,"route":"R3_GRAPH_NEURAL_GALERKIN","ablation":args.ablation,"representation_repair":"Petrov-Galerkin Physical32 FEM/COMSOL-compatible test space","optimization_repair":args.optimization_repair,"elementwise_graph_matrix_identity_claimed":False,"evidence_label":"historically exposed one-case capacity; not OOF, generalization or blind","best_epoch":int(ck["epoch"]),"epochs_executed":epochs,"parameter_count":sum(p.numel() for p in pars),"final_metrics":fm,"epoch0_metrics":e0,"diagnostic_gates":gates,"all_capacity_diagnostic_gates_pass":passed,"causality_future_perturbation_max_abs":causal,"graph_load_branch_sensitivity_relative_l2":graph_sensitivity,"final_physics_weight":pw,"HPO_authorized":False,"repairs_consumed":repairs,"source_hashes":{str(p):sha(p) for p in (DATA,GRAPH,VAR,PROTOCOL,WITNESS,Path(__file__))},"generated_utc":datetime.now(timezone.utc).isoformat()}
    atomic(out/"report.json",report);atomic(out/"status.json",{"status":status,"run_id":run,"best_epoch":int(ck["epoch"]),"final_metrics":fm,"HPO_authorized":False});event("run_finished",status=status,metrics=fm);print(json.dumps({"status":status,"run_id":run,"best_epoch":int(ck["epoch"]),"metrics":fm},indent=2))

if __name__=="__main__":main()
