#!/usr/bin/env python3
"""Generate the first thesis-grade S12 OOF comparison figure set."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from s12_evidence_context import EvidenceContext, load_aggregate, load_per_case, resolve


ROOT=Path(__file__).resolve().parents[1]
S10=ROOT/"s10_nested_grouped_oof";S11=ROOT/"s11_five_seed_confirmation";S12=ROOT/"s12_final_diagnostics"
DECISION=S11/"S11_TO_S12_DECISION_V1.json";AUDIT=S11/"independent_oof_audit_v1"
FIGURES=S12/"figures";DATA=S12/"figure_data";CAPTIONS=S12/"captions";MANIFESTS=S12/"figure_manifests"
COLORS={"FEM":"#252525","physics":"#2463A6","control":"#D18B20","B2":"#777777","error":"#A34A3A"}
AXES="XYZ"


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def atomic_json(path:Path,payload:dict)->None:
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");os.replace(temporary,path)


def style()->None:
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,"axes.labelsize":9,"axes.edgecolor":"#555555","axes.linewidth":0.8,"axes.grid":True,"grid.color":"#E2E2E2","grid.linewidth":0.6,"grid.alpha":0.8,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","legend.frameon":False})


def save(fig:plt.Figure,figure_id:str,title:str,caption:str,frame:pd.DataFrame,metadata:dict)->None:
    try:
        evidence_context=resolve(ROOT)
    except (SystemExit,RuntimeError,FileNotFoundError):
        evidence_context=None
    if evidence_context is not None and not evidence_context.five_seed_claim_allowed:
        caption=caption.replace("across five seeds","for the single S10 diagnostic seed").replace("median across five seeds","single S10 diagnostic seed").replace("median-seed","single-seed").replace("median seed","single seed").replace("each finalist","the best-ranked non-promoted S10 route").replace("every finalist","the best-ranked non-promoted S10 route")
        metadata={**metadata,"evidence_mode":evidence_context.mode,"five_seed_claim_allowed":False,"diagnostic_only_non_promoted_route":True}
    for directory in (FIGURES,DATA,CAPTIONS,MANIFESTS):directory.mkdir(parents=True,exist_ok=True)
    png=FIGURES/f"{figure_id}.png";pdf=FIGURES/f"{figure_id}.pdf";csv=DATA/f"{figure_id}.csv";caption_path=CAPTIONS/f"{figure_id}.caption.json";manifest=MANIFESTS/f"{figure_id}.manifest.json"
    frame.to_csv(csv,index=False);fig.savefig(png,dpi=300,bbox_inches="tight");fig.savefig(pdf,bbox_inches="tight");plt.close(fig)
    atomic_json(caption_path,{"figure_id":figure_id,"title":title,"caption":caption,"units":metadata.get("units"),"evidence_label":"historically exposed grouped OOF; not blind or external"})
    caller=Path(inspect.stack()[1].filename).resolve()
    atomic_json(manifest,{"figure_id":figure_id,"title":title,"generated_utc":datetime.now(timezone.utc).isoformat(),"script":str(caller),"script_sha256":sha256(caller),"source_csv":str(csv),"source_csv_sha256":sha256(csv),"png":str(png),"png_sha256":sha256(png),"pdf":str(pdf),"pdf_sha256":sha256(pdf),**metadata})


def model_name(trial:str,variant:str)->str:
    if trial=="COMMON_B2":return "B2"
    return f"{trial}-{variant}"


def median_case_table(per_case:pd.DataFrame,finalists:list[str])->pd.DataFrame:
    selected=per_case[(per_case.quantity=="total_displacement") & (((per_case.trial_id.isin(finalists)) & (per_case.variant.isin(["physics","control"]))) | (per_case.trial_id=="COMMON_B2"))].copy()
    selected["model"]=selected.apply(lambda row:model_name(row.trial_id,row.variant),axis=1)
    candidate=selected[selected.trial_id!="COMMON_B2"].groupby(["trial_id","variant","model","case_id","axis"],as_index=False).relative_l2.median()
    b2=selected[selected.trial_id=="COMMON_B2"][["trial_id","variant","model","case_id","axis","relative_l2"]]
    return pd.concat([candidate,b2],ignore_index=True)


def f17_ecdf(table:pd.DataFrame)->None:
    fig,panels=plt.subplots(1,3,figsize=(11.2,3.4),sharey=True)
    for axis,panel in zip(AXES,panels):
        subset=table[table.axis==axis]
        for model,group in subset.groupby("model"):
            values=np.sort(group.relative_l2.to_numpy());prob=np.arange(1,len(values)+1)/len(values);color=COLORS["B2"] if model=="B2" else (COLORS["control"] if model.endswith("control") else COLORS["physics"]);line="--" if model.endswith("control") else "-";panel.step(values,prob,where="post",label=model,color=color,linestyle=line,alpha=.9)
        panel.set_title(f"{axis} — {'transverse' if axis=='X' else 'vertical' if axis=='Y' else 'longitudinal'}");panel.set_xlabel("Relative L2 error");panel.set_xlim(left=0)
    panels[0].set_ylabel("Empirical cumulative probability");panels[-1].legend(fontsize=7,loc="lower right");fig.suptitle("OOF case-error empirical distributions");fig.tight_layout()
    save(fig,"F17","OOF case-error empirical distributions","ECDF of trajectory-level total-displacement relative L2 after taking the median across five seeds for each finalist. B2 is the common-split deterministic comparator.",table,{"units":"dimensionless","quantity":"total displacement","axes":list(AXES)})


def f18_box(table:pd.DataFrame)->None:
    models=list(dict.fromkeys(table.model.tolist()));fig,panels=plt.subplots(1,3,figsize=(12,4),sharey=False)
    for axis,panel in zip(AXES,panels):
        groups=[table[(table.axis==axis)&(table.model==model)].relative_l2.to_numpy() for model in models];boxes=panel.boxplot(groups,tick_labels=models,patch_artist=True,showfliers=True,medianprops={"color":"#111111"});
        for box,model in zip(boxes["boxes"],models):box.set_facecolor(COLORS["B2"] if model=="B2" else COLORS["control"] if model.endswith("control") else COLORS["physics"]);box.set_alpha(.45)
        panel.set_title(axis);panel.set_ylabel("Relative L2 error");panel.tick_params(axis="x",rotation=35,labelsize=7)
    fig.suptitle("OOF error distributions by model and global component");fig.tight_layout()
    save(fig,"F18","OOF error distributions by model and global component","Trajectory-level total-displacement error distributions. Candidate values are seed medians; each box contains 68 paired trajectories, not snapshots.",table,{"units":"dimensionless","quantity":"total displacement"})


def f20_tails(table:pd.DataFrame)->None:
    rows=[]
    for (model,axis),group in table.groupby(["model","axis"]):
        values=group.relative_l2.to_numpy();rows.append({"model":model,"axis":axis,"P50":np.percentile(values,50),"P90":np.percentile(values,90),"P95":np.percentile(values,95),"worst":np.max(values)})
    frame=pd.DataFrame(rows);fig,panels=plt.subplots(1,3,figsize=(12,3.8),sharey=False)
    markers={"P50":"o","P90":"s","P95":"^","worst":"D"}
    for axis,panel in zip(AXES,panels):
        subset=frame[frame.axis==axis];x=np.arange(len(subset))
        for metric,marker in markers.items():panel.plot(x,subset[metric],marker=marker,label=metric,linewidth=1.2)
        panel.set_xticks(x,subset.model,rotation=35,ha="right",fontsize=7);panel.set_title(axis);panel.set_ylabel("Relative L2 error");panel.set_ylim(bottom=0)
    panels[-1].legend(fontsize=7);fig.suptitle("Median and tail OOF errors");fig.tight_layout()
    save(fig,"F20","Median and tail OOF errors","P50, P90, P95 and worst trajectory-level total-displacement relative L2. Tail metrics are noncompensatory in the final decision.",frame,{"units":"dimensionless","quantity":"total displacement"})


def f21_heatmap(table:pd.DataFrame)->None:
    pivot=table.pivot_table(index="case_id",columns=["model","axis"],values="relative_l2",aggfunc="first");order=pivot.max(axis=1).sort_values(ascending=False).index;pivot=pivot.loc[order]
    fig,ax=plt.subplots(figsize=(max(10,.32*pivot.shape[1]),11));image=ax.imshow(pivot.to_numpy(),aspect="auto",cmap="magma",interpolation="nearest");ax.set_xticks(np.arange(pivot.shape[1]),[f"{m}\n{a}" for m,a in pivot.columns],rotation=45,ha="right",fontsize=7);ax.set_yticks(np.arange(pivot.shape[0]),pivot.index,fontsize=5);ax.grid(False);ax.set_title("Trajectory × model × component OOF error");fig.colorbar(image,ax=ax,label="Relative L2 error",fraction=.02,pad=.02);fig.tight_layout()
    source=pivot.reset_index();source.columns=["case_id"]+[f"{m}__{a}" for m,a in pivot.columns]
    save(fig,"F21","Trajectory × model × component OOF error","Heatmap ordered by each trajectory's maximum model-component error. Candidate entries are medians across five seeds.",source,{"units":"dimensionless","quantity":"total displacement"})


def f23_seed(aggregate:pd.DataFrame,finalists:list[str],context:EvidenceContext)->None:
    frame=aggregate[(aggregate.trial_id.isin(finalists))&(aggregate.variant=="physics")&(aggregate.quantity=="total_displacement")].copy();frame["model"]=frame.trial_id
    fig,panels=plt.subplots(1,3,figsize=(10.5,3.5),sharey=False)
    for axis,panel in zip(AXES,panels):
        subset=frame[frame.axis==axis]
        for model,group in subset.groupby("model"):panel.plot(group.seed,group.pooled_relative_l2,marker="o",label=model)
        panel.set_title(axis);panel.set_xlabel("Seed");panel.set_ylabel("Pooled relative L2");panel.set_xticks([0,1,2,3,4]);panel.set_ylim(bottom=0)
    panels[-1].legend(fontsize=7);title="Five-seed OOF stability" if context.five_seed_claim_allowed else "S10 single-seed OOF diagnostic disclosure";fig.suptitle(title);fig.tight_layout()
    caption="Pooled total-displacement relative L2 for every finalist, global component and frozen seed. No seed is omitted or selected for favorability." if context.five_seed_claim_allowed else "Case-mean total-displacement relative L2 for the single S10 seed of the best-ranked route after no family qualified for S11. This is diagnostic negative-result evidence, not seed stability or finalist confirmation."
    save(fig,"F23",title,caption,frame,{"units":"dimensionless","quantity":"total displacement","evidence_mode":context.mode})


def residual_frame(finalists:list[str],context:EvidenceContext)->pd.DataFrame:
    rows=[]
    for trial in finalists:
        for seed in context.seeds:
            for fold in range(5):
                values={}
                for variant in ("physics","control"):
                    path=context.run_report(trial,fold,variant,seed);report=json.loads(path.read_text(encoding="utf-8"));values[variant]=float(report["validation_metrics"]["equilibrium_residual_median"])
                    rows.append({"trial_id":trial,"seed":seed,"outer_fold":fold,"variant":variant,"equilibrium_residual_median":values[variant]})
                rows.append({"trial_id":trial,"seed":seed,"outer_fold":fold,"variant":"paired_reduction","equilibrium_residual_median":1-values["physics"]/max(values["control"],1e-30)})
    return pd.DataFrame(rows)


def f37_residual(frame:pd.DataFrame)->None:
    reductions=frame[frame.variant=="paired_reduction"];fig,ax=plt.subplots(figsize=(8,4))
    for index,(trial,group) in enumerate(reductions.groupby("trial_id")):
        x=np.full(len(group),index)+np.linspace(-.12,.12,len(group));ax.scatter(x,100*group.equilibrium_residual_median,s=18,alpha=.65,label=trial);ax.plot([index-.22,index+.22],[100*group.equilibrium_residual_median.median()]*2,color="#111111",linewidth=2)
    ax.axhline(25,color=COLORS["B2"],linestyle="--",label="25% gate");ax.axhline(0,color="#333333",linewidth=.8);ax.set_xticks(range(reductions.trial_id.nunique()),list(reductions.trial_id.unique()));ax.set_ylabel("Paired residual reduction (%)");ax.set_title("Equilibrium-residual reduction by seed and outer fold");ax.legend(fontsize=7);fig.tight_layout()
    save(fig,"F37","Equilibrium-residual reduction by seed and outer fold","Each point is one paired physics/control seed-fold comparison. Horizontal black segments are medians; positive values favor the physics-informed variant.",frame,{"units":"percent for plotted paired reductions","quantity":"reduced equilibrium residual"})


def f38_bc(finalists:list[str],context:EvidenceContext)->None:
    rows=[]
    for trial in finalists:
        for seed in context.seeds:
            for fold in range(5):
                for variant in ("physics","control"):
                    report=json.loads(context.run_report(trial,fold,variant,seed).read_text(encoding="utf-8"));value=float(report["validation_metrics"]["hard_BC_max_abs"]);rows.append({"trial_id":trial,"seed":seed,"outer_fold":fold,"variant":variant,"hard_BC_max_abs_m":value,"plot_value_m":max(value,1e-16)})
    frame=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(8,4))
    labels=[];values=[]
    for (trial,variant),group in frame.groupby(["trial_id","variant"]):labels.append(f"{trial}\n{variant}");values.append(group.plot_value_m.to_numpy())
    ax.boxplot(values,tick_labels=labels);ax.axhline(1e-12,color=COLORS["error"],linestyle="--",label="1e-12 gate");ax.set_yscale("log");ax.set_ylabel("Max |BC translation| (m)");ax.set_title("Hard boundary-condition compliance");ax.tick_params(axis="x",rotation=25,labelsize=7);ax.legend();fig.tight_layout()
    save(fig,"F38","Hard boundary-condition compliance","Maximum support-translation violation for all seed-fold runs. Exact zeros are displayed at 1e-16 m solely for logarithmic visualization and remain zero in source data.",frame,{"units":"m","quantity":"hard BC violation"})


def f43_cost(aggregate:pd.DataFrame,finalists:list[str],context:EvidenceContext)->None:
    rows=[]
    for trial in finalists:
        for seed in context.seeds:
            metrics=aggregate[(aggregate.trial_id==trial)&(aggregate.variant=="physics")&(aggregate.seed==seed)&(aggregate.quantity=="total_displacement")];error=float(metrics.pooled_relative_l2.mean())
            reports=[]
            for fold in range(5):reports.append(json.loads(context.run_report(trial,fold,"physics",seed).read_text(encoding="utf-8")))
            rows.append({"trial_id":trial,"seed":seed,"mean_axis_pooled_relative_l2":error,"parameter_count":int(np.median([r["parameter_count"] for r in reports])),"peak_vram_GiB":float(np.max([r["peak_vram_GiB"] for r in reports]))})
    frame=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(7,4.5))
    for trial,group in frame.groupby("trial_id"):ax.scatter(group.parameter_count,group.mean_axis_pooled_relative_l2,s=35+25*group.peak_vram_GiB,label=trial,alpha=.75)
    ax.set_xlabel("Trainable parameters");ax.set_ylabel("Mean-axis pooled relative L2");ax.set_title("Predictive error versus model capacity and VRAM");ax.legend(fontsize=7);fig.tight_layout()
    save(fig,"F43","Predictive error versus model capacity and VRAM","Each point is one frozen seed. Marker area increases with peak VRAM. Training time, inference time and FEM speedup require separate measured timing and are not inferred here.",frame,{"units":"dimensionless error; parameter count; GiB VRAM","quantity":"computational comparison"})


def main()->None:
    context=resolve(ROOT)
    if FIGURES.exists() and any(FIGURES.glob("F17.*")):raise FileExistsError("S12 core OOF figures already exist")
    style();finalists=list(context.candidates);per_case=load_per_case(context);aggregate=load_aggregate(context);table=median_case_table(per_case,finalists)
    f17_ecdf(table);f18_box(table);f20_tails(table);f21_heatmap(table);f23_seed(aggregate,finalists,context);residual=residual_frame(finalists,context);f37_residual(residual);f38_bc(finalists,context);f43_cost(aggregate,finalists,context)
    report={"status":"PASS_S12_CORE_OOF_FIGURES","generated_utc":datetime.now(timezone.utc).isoformat(),"figure_ids":["F17","F18","F20","F21","F23","F37","F38","F43"],"figure_count":8,"training_or_tuning_performed":False,"S13_authorized":False}
    atomic_json(S12/"S12_CORE_OOF_FIGURES_REPORT.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
