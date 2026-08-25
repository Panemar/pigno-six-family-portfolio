#!/usr/bin/env python3
"""Generate gated paired OOF, scenario, temporal and field figures F19/F22/F24-F29."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from s12_evidence_context import load_aggregate, load_per_case, resolve


ROOT=Path(__file__).resolve().parents[1];S10=ROOT/"s10_nested_grouped_oof";S11=ROOT/"s11_five_seed_confirmation";S12=ROOT/"s12_final_diagnostics"
AUDIT=S11/"independent_oof_audit_v1";DECISION=S11/"S11_TO_S12_DECISION_V1.json";DATASET=S10/"S10_ORIGINAL_68CASE_DATASET.h5";B2=S10/"b2_common_split_target_clean_v1"/"S10_B2_COMMON_SPLIT_OOF.h5"
_spec=importlib.util.spec_from_file_location("fig_utils",ROOT/"scripts"/"65_generate_s12_core_oof_figures.py");_fig=importlib.util.module_from_spec(_spec);assert _spec.loader is not None;_spec.loader.exec_module(_fig)
AXES="XYZ"


def decode(values)->list[str]:return [value.decode() if isinstance(value,bytes) else str(value) for value in values]


def representative_seed_case(trial:str,aggregate:pd.DataFrame,per_case:pd.DataFrame,worst:bool=False)->tuple[int,str]:
    seed_score=aggregate[(aggregate.trial_id==trial)&(aggregate.variant=="physics")&(aggregate.quantity=="total_displacement")].groupby("seed").pooled_relative_l2.mean();seed=int((seed_score-seed_score.median()).abs().idxmin())
    case_score=per_case[(per_case.trial_id==trial)&(per_case.variant=="physics")&(per_case.seed==seed)&(per_case.quantity=="total_displacement")].groupby("case_id").relative_l2.mean();case=str(case_score.idxmax() if worst else (case_score-case_score.median()).abs().idxmin());return seed,case


def candidate_path(trial:str,variant:str,seed:int)->Path:return resolve(ROOT).field_path(trial,variant,seed)


def f19(per_case:pd.DataFrame,finalists:list[str])->None:
    rows=[];rng=np.random.default_rng(20260811)
    for trial in finalists:
        for axis in AXES:
            candidate=per_case[(per_case.trial_id==trial)&(per_case.variant=="physics")&(per_case.quantity=="total_displacement")&(per_case.axis==axis)].groupby("case_id").relative_l2.median().sort_index();baseline=per_case[(per_case.trial_id=="COMMON_B2")&(per_case.axis==axis)].set_index("case_id").relative_l2.sort_index();
            if not candidate.index.equals(baseline.index):raise RuntimeError("F19 paired-case identity drift")
            difference=candidate.to_numpy()-baseline.to_numpy();indices=rng.integers(0,len(difference),size=(10000,len(difference)));means=np.mean(difference[indices],axis=1);rows.append({"trial_id":trial,"axis":axis,"paired_mean_difference":float(np.mean(difference)),"ci95_low":float(np.percentile(means,2.5)),"ci95_high":float(np.percentile(means,97.5)),"probability_improvement":float(np.mean(means<0)),"case_count":len(difference)})
    frame=pd.DataFrame(rows);fig,panel=plt.subplots(figsize=(8.5,4.8));labels=[]
    for index,row in frame.iterrows():panel.errorbar(row.paired_mean_difference,index,xerr=[[row.paired_mean_difference-row.ci95_low],[row.ci95_high-row.paired_mean_difference]],fmt={"X":"o","Y":"s","Z":"D"}[row.axis],color="#2463A6",capsize=3);labels.append(f"{row.trial_id} — {row.axis}")
    panel.axvline(0,color="#333333",lw=.9);panel.set_yticks(range(len(frame)),labels);panel.set_xlabel("Paired mean relative-L2 difference: PIGNO − B2");panel.set_title("Trajectory-paired OOF improvement versus common B2");fig.tight_layout();_fig.save(fig,"F19","Trajectory-paired OOF improvement versus common B2","Mean paired trajectory error difference and 95% case bootstrap interval after taking the candidate median across five seeds. Negative values favor the physics-informed candidate; 68 trajectories, not snapshots, are resampled.",frame,{"units":"dimensionless relative-L2 difference","quantity":"total displacement","bootstrap_draws":10000})


def case_factor_table(fem:h5py.File)->pd.DataFrame:
    names=decode(fem["causal/static_feature_names"][:]);features=fem["causal/static_features"][:];frame=pd.DataFrame(features,columns=names);frame.insert(0,"case_id",decode(fem["case_id"][:]));return frame


def f22(per_case:pd.DataFrame,finalists:list[str],factors:pd.DataFrame)->None:
    selected=per_case[(per_case.trial_id.isin(finalists))&(per_case.variant=="physics")&(per_case.quantity=="total_displacement")].groupby(["trial_id","case_id","axis"],as_index=False).relative_l2.median();selected=selected.merge(factors,on="case_id",how="left");selected["axis_mean_error"]=selected.groupby(["trial_id","case_id"]).relative_l2.transform("mean");case=selected.drop_duplicates(["trial_id","case_id"]);rows=[]
    factor_columns=[("speed_kmh","Speed (km/h)"),("train_count","Train count"),("wind_mps","Wind (m/s)"),("seismic_scale_factor","Seismic scale")]
    fig,panels=plt.subplots(2,2,figsize=(10.8,7.2))
    for panel,(column,label) in zip(panels.ravel(),factor_columns):
        for trial,group in case.groupby("trial_id"):
            summary=group.groupby(column).axis_mean_error.agg(["median",lambda x:np.percentile(x,25),lambda x:np.percentile(x,75)]).reset_index();summary.columns=[column,"median","q25","q75"];panel.errorbar(summary[column],summary["median"],yerr=[summary["median"]-summary.q25,summary.q75-summary["median"]],marker="o",capsize=3,label=trial)
            rows.extend({"trial_id":trial,"factor":column,"level":float(row[column]),"median_relative_l2":float(row["median"]),"q25":float(row.q25),"q75":float(row.q75),"case_count":int((group[column]==row[column]).sum())} for _,row in summary.iterrows())
        panel.set_xlabel(label);panel.set_ylabel("Median case error across X/Y/Z");panel.set_ylim(bottom=0)
    panels[0,0].legend(fontsize=7);fig.suptitle("OOF error by frozen factorial scenario factors");fig.tight_layout();_fig.save(fig,"F22","OOF error by frozen factorial scenario factors","Median and interquartile trajectory error by speed, train count, wind and seismic scale. Candidate error is first median-aggregated across five seeds and then summarized across complete trajectories.",pd.DataFrame(rows),{"units":"dimensionless error; factor units on axes","quantity":"total displacement"})


def fixed_energy_node(target:np.ndarray,axis:int)->int:return int(np.argmax(np.mean(target[:,:,axis].astype(np.float64)**2,axis=0)))


def history_rows(trial:str,seed:int,case:str,fem:h5py.File,b2:h5py.File)->tuple[list[dict],dict]:
    cases=decode(fem["case_id"][:]);index=cases.index(case);time=fem["time_s"][:];target=fem["response/total_translation_m"][index]
    with h5py.File(candidate_path(trial,"physics",seed),"r") as physics,h5py.File(candidate_path(trial,"control",seed),"r") as control:
        fields={"FEM_COMSOL":target,"physics":physics["hybrid_total_displacement_m"][index][:],"control":control["hybrid_total_displacement_m"][index][:],"B2":b2["prediction_uvw_m"][index][:]}
    rows=[];nodes={}
    for axis,axis_name in enumerate(AXES):
        node=fixed_energy_node(target,axis);nodes[axis_name]=node
        for model,field in fields.items():rows.extend({"trial_id":trial,"seed":seed,"case_id":case,"axis":axis_name,"node_zero_based":node,"time_s":float(t),"model":model,"displacement_m":float(value)} for t,value in zip(time,field[:,node,axis]))
    return rows,nodes


def plot_histories(frame:pd.DataFrame,finalists:list[str],figure_id:str,title:str,caption:str)->None:
    fig,panels=plt.subplots(len(finalists),3,figsize=(12,3.1*len(finalists)),squeeze=False)
    colors={"FEM_COMSOL":"#252525","physics":"#2463A6","control":"#D18B20","B2":"#777777"};styles={"FEM_COMSOL":"-","physics":"-","control":"--","B2":":"}
    for row,trial in enumerate(finalists):
        for col,axis in enumerate(AXES):
            panel=panels[row,col];subset=frame[(frame.trial_id==trial)&(frame.axis==axis)]
            for model,group in subset.groupby("model"):panel.plot(group.time_s,1000*group.displacement_m,color=colors[model],linestyle=styles[model],label=model,lw=1)
            panel.set_title(f"{trial} — {axis}, node {int(subset.node_zero_based.iloc[0])}");panel.set_xlabel("Time (s)");panel.set_ylabel("Displacement (mm)")
    panels[0,-1].legend(fontsize=7);fig.suptitle(title);fig.tight_layout();_fig.save(fig,figure_id,title,caption,frame,{"units":"s and m in source; mm plotted","quantity":"total displacement"})


def f24_25(finalists:list[str],aggregate:pd.DataFrame,per_case:pd.DataFrame,fem:h5py.File,b2:h5py.File)->dict:
    selections={};representative_rows=[];worst_rows=[]
    for trial in finalists:
        rep_seed,rep_case=representative_seed_case(trial,aggregate,per_case,False);worst_seed,worst_case=representative_seed_case(trial,aggregate,per_case,True);rows,nodes=history_rows(trial,rep_seed,rep_case,fem,b2);representative_rows.extend(rows);rows,worst_nodes=history_rows(trial,worst_seed,worst_case,fem,b2);worst_rows.extend(rows);selections[trial]={"representative":{"seed":rep_seed,"case_id":rep_case,"nodes":nodes},"worst":{"seed":worst_seed,"case_id":worst_case,"nodes":worst_nodes}}
    plot_histories(pd.DataFrame(representative_rows),finalists,"F24","Representative OOF time histories","Median-seed, median-case histories at the fixed FEM/COMSOL maximum-energy observation node for each component. The same case, time, node and component are used for all models.")
    plot_histories(pd.DataFrame(worst_rows),finalists,"F25","Worst-case OOF time histories","Median-seed worst-case histories at fixed FEM/COMSOL maximum-energy nodes. Worst case is selected by mean X/Y/Z trajectory error, not visual appearance.")
    return selections


def f26(finalists:list[str],selections:dict,per_case:pd.DataFrame,fem:h5py.File,b2:h5py.File)->None:
    cases=decode(fem["case_id"][:]);time=fem["time_s"][:];rows=[];fig,panels=plt.subplots(len(finalists),1,figsize=(10,3.2*len(finalists)),squeeze=False)
    colors={"FEM_COMSOL":"#252525","physics":"#2463A6","control":"#D18B20","B2":"#777777"}
    for row,trial in enumerate(finalists):
        seed=selections[trial]["worst"]["seed"];case=selections[trial]["worst"]["case_id"];axis_errors=per_case[(per_case.trial_id==trial)&(per_case.variant=="physics")&(per_case.seed==seed)&(per_case.case_id==case)&(per_case.quantity=="total_displacement")].set_index("axis").relative_l2;axis_name=str(axis_errors.idxmax());axis=AXES.index(axis_name);index=cases.index(case);target=fem["response/total_translation_m"][index];node=int(np.argmax(np.max(np.abs(target[:,:,axis]),axis=0)));peak=int(np.argmax(np.abs(target[:,node,axis])));mask=np.abs(time-time[peak])<=1.0
        with h5py.File(candidate_path(trial,"physics",seed),"r") as physics,h5py.File(candidate_path(trial,"control",seed),"r") as control:fields={"FEM_COMSOL":target,"physics":physics["hybrid_total_displacement_m"][index][:],"control":control["hybrid_total_displacement_m"][index][:],"B2":b2["prediction_uvw_m"][index][:]}
        panel=panels[row,0]
        for model,field in fields.items():panel.plot(time[mask],1000*field[mask,node,axis],label=model,color=colors[model],linestyle="--" if model=="control" else ":" if model=="B2" else "-");rows.extend({"trial_id":trial,"seed":seed,"case_id":case,"axis":axis_name,"node_zero_based":node,"FEM_peak_time_s":float(time[peak]),"time_s":float(t),"model":model,"displacement_m":float(value)} for t,value in zip(time[mask],field[mask,node,axis]))
        panel.axvline(time[peak],color="#A34A3A",lw=.8);panel.set_title(f"{trial} — {case}, {axis_name}, node {node}");panel.set_xlabel("Time (s)");panel.set_ylabel("Displacement (mm)")
    panels[0,0].legend(fontsize=7);fig.suptitle("Peak-region temporal zoom at frozen worst cases");fig.tight_layout();_fig.save(fig,"F26","Peak-region temporal zoom at frozen worst cases","A ±1 s window around the FEM/COMSOL peak at the maximum-amplitude observation node of the worst component in each frozen worst case. No temporal alignment or phase shifting is applied.",pd.DataFrame(rows),{"units":"s and m in source; mm plotted","quantity":"total displacement"})


def f27(finalists:list[str],selections:dict,fem:h5py.File)->None:
    cases=decode(fem["case_id"][:]);time=fem["time_s"][:];coords=fem["observation/coords_m"][:];order=np.argsort(coords[:,2],kind="stable");rows=[];fig,panels=plt.subplots(len(finalists),3,figsize=(13,3.3*len(finalists)),squeeze=False)
    for row,trial in enumerate(finalists):
        seed=selections[trial]["representative"]["seed"];case=selections[trial]["representative"]["case_id"];index=cases.index(case);target=fem["response/total_translation_m"][index,:,:,1]
        with h5py.File(candidate_path(trial,"physics",seed),"r") as candidate:prediction=candidate["hybrid_total_displacement_m"][index,:, :,1][:]
        fields=[("FEM/COMSOL",target),("PIGNO",prediction),("Absolute error",np.abs(prediction-target))];limit=max(np.max(np.abs(target)),np.max(np.abs(prediction)))
        for col,(name,field) in enumerate(fields):
            panel=panels[row,col];data=1000*field[:,order].T;cmap="magma" if col==2 else "coolwarm";vmin=0 if col==2 else -1000*limit;vmax=None if col==2 else 1000*limit;image=panel.imshow(data,aspect="auto",origin="lower",extent=[time[0],time[-1],coords[order,2].min(),coords[order,2].max()],cmap=cmap,vmin=vmin,vmax=vmax);panel.set_title(f"{trial} — {name}");panel.set_xlabel("Time (s)");panel.set_ylabel("Ordered Z coordinate (m)");fig.colorbar(image,ax=panel,fraction=.025,pad=.02,label="Y displacement (mm)")
        for node_rank,node in enumerate(order):rows.extend({"trial_id":trial,"seed":seed,"case_id":case,"time_s":float(t),"node_zero_based":int(node),"Z_m":float(coords[node,2]),"FEM_Y_m":float(f),"PIGNO_Y_m":float(p),"absolute_error_m":float(abs(p-f))} for t,f,p in zip(time,target[:,node],prediction[:,node]))
    fig.suptitle("Representative vertical space–time fields");fig.tight_layout();_fig.save(fig,"F27","Representative vertical space–time fields","FEM/COMSOL, PIGNO and absolute-error vertical fields at the frozen representative seed/case. Nodes are stably ordered by physical Z coordinate; all 1201 saved times and 512 observations are retained.",pd.DataFrame(rows),{"units":"s, m in source; mm plotted","quantity":"Y vertical total displacement"})


def critical_context(trial:str,selections:dict,per_case:pd.DataFrame,fem:h5py.File)->dict:
    seed=selections[trial]["worst"]["seed"];case=selections[trial]["worst"]["case_id"];axis_errors=per_case[(per_case.trial_id==trial)&(per_case.variant=="physics")&(per_case.seed==seed)&(per_case.case_id==case)&(per_case.quantity=="total_displacement")].set_index("axis").relative_l2;axis_name=str(axis_errors.idxmax());axis=AXES.index(axis_name);cases=decode(fem["case_id"][:]);index=cases.index(case);target=fem["response/total_translation_m"][index];time=fem["time_s"][:];critical=int(np.argmax(np.max(np.abs(target[:,:,axis]),axis=1)));return {"trial":trial,"seed":seed,"case":case,"axis":axis,"axis_name":axis_name,"case_index":index,"time_index":critical,"time_s":float(time[critical])}


def f28_29(finalists:list[str],selections:dict,per_case:pd.DataFrame,fem:h5py.File)->None:
    coords=fem["observation/coords_m"][:];rows=[];shape_rows=[];fig28,axes28=plt.subplots(len(finalists),3,figsize=(13,3.4*len(finalists)),squeeze=False);fig29,axes29=plt.subplots(len(finalists),2,figsize=(12,3.8*len(finalists)),squeeze=False)
    for row,trial in enumerate(finalists):
        context=critical_context(trial,selections,per_case,fem);target=fem["response/total_translation_m"][context["case_index"],context["time_index"]]
        with h5py.File(candidate_path(trial,"physics",context["seed"]),"r") as candidate:prediction=candidate["hybrid_total_displacement_m"][context["case_index"],context["time_index"]][:]
        axis=context["axis"];limit=max(np.max(np.abs(target[:,axis])),np.max(np.abs(prediction[:,axis])));fields=[("FEM/COMSOL",target[:,axis]),("PIGNO",prediction[:,axis]),("Absolute error",np.abs(prediction[:,axis]-target[:,axis]))]
        for col,(name,values) in enumerate(fields):
            panel=axes28[row,col];image=panel.scatter(coords[:,2],coords[:,0],c=1000*values,cmap="magma" if col==2 else "coolwarm",vmin=0 if col==2 else -1000*limit,vmax=None if col==2 else 1000*limit,s=7);panel.set_aspect("equal",adjustable="box");panel.set_title(f"{trial} — {name}");panel.set_xlabel("Z longitudinal (m)");panel.set_ylabel("X transverse (m)");fig28.colorbar(image,ax=panel,fraction=.025,pad=.02,label=f"{context['axis_name']} displacement (mm)")
        span=max(np.ptp(coords[:,2]),np.ptp(coords[:,1]));max_disp=max(np.max(np.linalg.norm(target,axis=1)),np.max(np.linalg.norm(prediction,axis=1)),1e-12);scale=.05*span/max_disp
        order=np.argsort(coords[:,2],kind="stable")
        for col,(label,factor) in enumerate([("true displacement scale",1.0),(f"displacement ×{scale:.1f}",scale)]):
            panel=axes29[row,col];panel.scatter(coords[order,2]+factor*target[order,2],coords[order,1]+factor*target[order,1],s=7,color="#252525",label="FEM/COMSOL");panel.scatter(coords[order,2]+factor*prediction[order,2],coords[order,1]+factor*prediction[order,1],s=5,facecolors="none",edgecolors="#2463A6",label="PIGNO");panel.set_aspect("equal",adjustable="box");panel.set_title(f"{trial} — {label}");panel.set_xlabel("Z longitudinal (m)");panel.set_ylabel("Y vertical (m)")
        axes29[row,0].legend(fontsize=7)
        for node in range(512):
            rows.append({"trial_id":trial,**context,"node_zero_based":node,"X_m":coords[node,0],"Y_m":coords[node,1],"Z_m":coords[node,2],"FEM_displacement_m":target[node,axis],"PIGNO_displacement_m":prediction[node,axis],"absolute_error_m":abs(prediction[node,axis]-target[node,axis])})
            shape_rows.append({"trial_id":trial,**context,"node_zero_based":node,"X_m":coords[node,0],"Y_m":coords[node,1],"Z_m":coords[node,2],"FEM_uX_m":target[node,0],"FEM_uY_m":target[node,1],"FEM_uZ_m":target[node,2],"PIGNO_uX_m":prediction[node,0],"PIGNO_uY_m":prediction[node,1],"PIGNO_uZ_m":prediction[node,2],"readability_scale":scale})
    fig28.suptitle("Critical-instant spatial fields at frozen worst contexts");fig28.tight_layout();_fig.save(fig28,"F28","Critical-instant spatial fields at frozen worst contexts","True-aspect plan projections of FEM/COMSOL, PIGNO and absolute error at the FEM critical time of the worst component and frozen worst case.",pd.DataFrame(rows),{"units":"m in source; mm plotted","quantity":"worst-component total displacement"})
    fig29.suptitle("FEM/COMSOL and PIGNO deformed elevation shapes");fig29.tight_layout();_fig.save(fig29,"F29","FEM/COMSOL and PIGNO deformed elevation shapes","Left panels use the true displacement scale. Right panels magnify displacement only by the explicitly serialized factor so deformation shape is legible; undeformed geometry and coordinate axes are not silently compressed.",pd.DataFrame(shape_rows),{"units":"m","quantity":"three-component total displacement and deformed elevation","readability_scaling":"explicit per trial in source CSV"})


def main()->None:
    context=resolve(ROOT)
    ids=["F19","F22","F24","F25","F26","F27","F28","F29"]
    if any((_fig.FIGURES/f"{figure_id}.png").exists() for figure_id in ids):raise FileExistsError("One or more paired OOF field figures already exist")
    _fig.style();finalists=list(context.candidates);aggregate=load_aggregate(context);per_case=load_per_case(context)
    with h5py.File(DATASET,"r") as fem,h5py.File(B2,"r") as b2:
        f19(per_case,finalists);f22(per_case,finalists,case_factor_table(fem));selections=f24_25(finalists,aggregate,per_case,fem,b2);f26(finalists,selections,per_case,fem,b2);f27(finalists,selections,fem);f28_29(finalists,selections,per_case,fem)
    selection="median seed; median or worst complete trajectory" if context.five_seed_claim_allowed else "single admitted S10 diagnostic seed; median or worst complete trajectory"
    report={"status":"PASS_S12_PAIRED_OOF_FIELD_FIGURES","figure_ids":ids,"selection_rules":selection+"; fixed FEM energy/peak node; same time-node-component","evidence_mode":context.mode,"five_seed_claim_allowed":context.five_seed_claim_allowed,"training_or_tuning_performed":False,"final_decision_authorized":False};_fig.atomic_json(S12/"S12_PAIRED_OOF_FIELD_FIGURES_REPORT.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
