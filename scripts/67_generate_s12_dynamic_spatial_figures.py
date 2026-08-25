#!/usr/bin/env python3
"""Generate S12 hotspot, spatial, spectral and kinematic figures F30-F36."""

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
from scipy.signal import coherence, csd, periodogram
from s12_evidence_context import EvidenceContext, load_aggregate, load_per_case, resolve


ROOT=Path(__file__).resolve().parents[1];S10=ROOT/"s10_nested_grouped_oof";S11=ROOT/"s11_five_seed_confirmation";S12=ROOT/"s12_final_diagnostics"
DYNAMIC=S12/"dynamic_spatial_multiseed_v1";DECISION=S11/"S11_TO_S12_DECISION_V1.json";AUDIT=S11/"independent_oof_audit_v1";DATASET=S10/"S10_ORIGINAL_68CASE_DATASET.h5";B2=S10/"b2_common_split_target_clean_v1"/"S10_B2_COMMON_SPLIT_OOF.h5"
_spec=importlib.util.spec_from_file_location("fig_utils",ROOT/"scripts"/"65_generate_s12_core_oof_figures.py");_fig=importlib.util.module_from_spec(_spec);assert _spec.loader is not None;_spec.loader.exec_module(_fig)
AXES="XYZ"


def representative_seed_and_case(trial:str,aggregate:pd.DataFrame,per_case:pd.DataFrame)->tuple[int,str]:
    seed_score=aggregate[(aggregate.trial_id==trial)&(aggregate.variant=="physics")&(aggregate.quantity=="total_displacement")].groupby("seed").pooled_relative_l2.mean();median=float(seed_score.median());seed=int((seed_score-median).abs().idxmin())
    cases=per_case[(per_case.trial_id==trial)&(per_case.variant=="physics")&(per_case.seed==seed)&(per_case.quantity=="total_displacement")].groupby("case_id").relative_l2.mean();case_median=float(cases.median());case=str((cases-case_median).abs().idxmin());return seed,case


def f30_hotspot(metrics:pd.DataFrame,finalists:list[str],context:EvidenceContext)->None:
    selected=metrics[((metrics.trial_id.isin(finalists))&(metrics.quantity=="total_displacement"))|((metrics.trial_id=="COMMON_B2")&(metrics.quantity=="total_displacement"))].copy();candidate=selected[selected.trial_id!="COMMON_B2"].groupby(["trial_id","variant","case_id","axis"],as_index=False).hotspot_distance_m.median();b2=selected[selected.trial_id=="COMMON_B2"][["trial_id","variant","case_id","axis","hotspot_distance_m"]];frame=pd.concat([candidate,b2],ignore_index=True);frame["model"]=frame.apply(lambda r:"B2" if r.trial_id=="COMMON_B2" else f"{r.trial_id}-{r.variant}",axis=1)
    models=list(dict.fromkeys(frame.model));fig,panels=plt.subplots(1,3,figsize=(12,3.8))
    for axis,panel in zip(AXES,panels):
        groups=[frame[(frame.axis==axis)&(frame.model==model)].hotspot_distance_m.to_numpy() for model in models];panel.boxplot(groups,tick_labels=models,showfliers=True);panel.set_title(axis);panel.set_ylabel("Hotspot distance (m)");panel.tick_params(axis="x",rotation=35,labelsize=7);panel.set_ylim(bottom=0)
    qualifier="trajectory medians across five seeds" if context.five_seed_claim_allowed else "single-seed S10 diagnostic values after no route qualified for S11"
    fig.suptitle("OOF hotspot localization error");fig.tight_layout();_fig.save(fig,"F30","OOF hotspot localization error",f"Euclidean distance between predicted and FEM/COMSOL maximum-response observation nodes; {qualifier}.",frame,{"units":"m","quantity":"total displacement hotspot","evidence_mode":context.mode})


def node_error_map(trial:str,seed:int,fem:h5py.File,context:EvidenceContext)->pd.DataFrame:
    path=context.field_path(trial,"physics",seed);coords=fem["observation/coords_m"][:];num=np.zeros((512,3));den=np.zeros((512,3))
    with h5py.File(path,"r") as candidate:
        for index in range(68):
            prediction=candidate["hybrid_total_displacement_m"][index].astype(np.float64);target=fem["response/total_translation_m"][index].astype(np.float64);num+=np.sum((prediction-target)**2,axis=0);den+=np.sum(target**2,axis=0)
    error=np.sqrt(num/np.maximum(den,1e-30));rows=[]
    for node in range(512):
        for axis,axis_name in enumerate(AXES):rows.append({"trial_id":trial,"seed":seed,"observation_node_zero_based":node,"X_m":coords[node,0],"Y_m":coords[node,1],"Z_m":coords[node,2],"axis":axis_name,"node_relative_l2":error[node,axis]})
    return pd.DataFrame(rows)


def f31_nodes(frame:pd.DataFrame,finalists:list[str],context:EvidenceContext)->None:
    fig=plt.figure(figsize=(12,3.8*len(finalists)))
    for row,trial in enumerate(finalists):
        for col,axis in enumerate(AXES):
            panel=fig.add_subplot(len(finalists),3,row*3+col+1,projection="3d");data=frame[(frame.trial_id==trial)&(frame.axis==axis)];points=panel.scatter(data.X_m,data.Z_m,data.Y_m,c=data.node_relative_l2,cmap="magma",s=7);span=np.ptp(data[["X_m","Z_m","Y_m"]].to_numpy(),axis=0);panel.set_box_aspect(np.maximum(span,1e-9));panel.set_proj_type("ortho");panel.view_init(elev=22,azim=-65,roll=0);panel.set_title(f"{trial} — {axis}");panel.set_xlabel("X transverse");panel.set_ylabel("Z longitudinal");panel.set_zlabel("Y vertical");panel.set_xticks([]);panel.set_yticks([]);panel.set_zticks([]);panel.grid(False);fig.colorbar(points,ax=panel,fraction=.025,pad=.02,label="Node relative L2")
    fig.suptitle("Nodewise OOF total-displacement error — true aspect");fig.tight_layout();_fig.save(fig,"F31","Nodewise OOF total-displacement error — true aspect","Nodewise pooled relative L2 over all 68 trajectories and 1201 saved times for the representative median seed of each finalist. Coordinates use X transverse, Z longitudinal and Y vertical without geometric compression.",frame,{"units":"dimensionless error; coordinates in m","quantity":"total displacement","geometry":"true aspect"})


def spectral_curves(target:np.ndarray,prediction:np.ndarray,dt:float)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    frequency,target_psd_nodes=periodogram(target,fs=1/dt,axis=0,detrend="constant",scaling="density");_,prediction_psd_nodes=periodogram(prediction,fs=1/dt,axis=0,detrend="constant",scaling="density");target_psd=np.mean(target_psd_nodes,axis=1);prediction_psd=np.mean(prediction_psd_nodes,axis=1)
    target_energy=np.sum(target.astype(np.float64)**2,axis=0);prediction_energy=np.sum(prediction.astype(np.float64)**2,axis=0);mask=(target_energy>max(float(np.max(target_energy))*1e-12,1e-30))&(prediction_energy>max(float(np.max(prediction_energy))*1e-12,1e-30))
    if not np.any(mask):return frequency,target_psd,prediction_psd,np.full_like(frequency,np.nan),np.full_like(frequency,np.nan)
    nperseg=min(256,target.shape[0]);coh_frequency,coh=coherence(target[:,mask],prediction[:,mask],fs=1/dt,nperseg=nperseg,axis=0);cross_frequency,cross=csd(target[:,mask],prediction[:,mask],fs=1/dt,nperseg=nperseg,axis=0);mean_coh=np.nanmean(coh,axis=1);mean_cross=np.mean(cross,axis=1);phase=np.angle(mean_cross);target_coarse=np.interp(coh_frequency,frequency,target_psd);phase[target_coarse<max(float(np.max(target_coarse))*1e-8,1e-30)]=np.nan;return frequency,target_psd,prediction_psd,np.interp(frequency,coh_frequency,mean_coh),np.interp(frequency,cross_frequency,phase,left=np.nan,right=np.nan)


def f32_34(finalists:list[str],representatives:dict[str,tuple[int,str]],fem:h5py.File,b2:h5py.File,context:EvidenceContext)->None:
    case_ids=[value.decode() if isinstance(value,bytes) else str(value) for value in fem["case_id"][:]];time=fem["time_s"][:];dt=float(np.median(np.diff(time)));rows=[]
    for trial in finalists:
        seed,case=representatives[trial];index=case_ids.index(case)
        with h5py.File(context.field_path(trial,"physics",seed),"r") as candidate:
            target=fem["response/total_translation_m"][index];prediction=candidate["hybrid_total_displacement_m"][index];baseline=b2["prediction_uvw_m"][index]
            for axis,axis_name in enumerate(AXES):
                for model,field in ((trial,prediction),("B2",baseline)):
                    frequency,target_psd,pred_psd,coh,phase=spectral_curves(target[:,:,axis],field[:,:,axis],dt)
                    rows.extend({"trial_context":trial,"seed":seed,"case_id":case,"axis":axis_name,"model":model,"frequency_hz":float(f),"target_psd":float(tp),"predicted_psd":float(pp),"coherence":float(c) if np.isfinite(c) else np.nan,"phase_rad":float(ph) if np.isfinite(ph) else np.nan} for f,tp,pp,c,ph in zip(frequency,target_psd,pred_psd,coh,phase))
    frame=pd.DataFrame(rows)
    for figure_id,metric,ylabel,title in (("F32","psd","PSD (m²/Hz)","PSD comparison at representative OOF trajectories"),("F33","coherence","Magnitude-squared coherence","Coherence at representative OOF trajectories"),("F34","phase_rad","Phase difference (rad)","Phase difference at representative OOF trajectories")):
        fig,panels=plt.subplots(len(finalists),3,figsize=(12,3.2*len(finalists)),squeeze=False)
        for row,trial in enumerate(finalists):
            for col,axis in enumerate(AXES):
                panel=panels[row,col];subset=frame[(frame.trial_context==trial)&(frame.axis==axis)]
                if figure_id=="F32":
                    first=subset[subset.model==trial];panel.semilogy(first.frequency_hz,np.maximum(first.target_psd,1e-30),color=_fig.COLORS["FEM"],label="FEM/COMSOL")
                    for model,group in subset.groupby("model"):panel.semilogy(group.frequency_hz,np.maximum(group.predicted_psd,1e-30),color=_fig.COLORS["B2"] if model=="B2" else _fig.COLORS["physics"],linestyle="--" if model=="B2" else "-",label=model)
                else:
                    for model,group in subset.groupby("model"):panel.plot(group.frequency_hz,group[metric],color=_fig.COLORS["B2"] if model=="B2" else _fig.COLORS["physics"],linestyle="--" if model=="B2" else "-",label=model)
                    if figure_id=="F33":panel.set_ylim(0,1.02)
                panel.set_xlim(0,20);panel.set_title(f"{trial} — {axis}");panel.set_xlabel("Frequency (Hz)");panel.set_ylabel(ylabel)
        panels[0,-1].legend(fontsize=7);fig.suptitle(title);fig.tight_layout();_fig.save(fig,figure_id,title,f"Full saved-grid {ylabel.lower()} for each finalist's median-seed, median-error representative trajectory. No low-pass filtering is applied; undefined low-energy phase values remain blank.",frame,{"units":ylabel,"quantity":"total displacement spectrum","frequency_range_hz":[0,20]})


def f35_bands(metrics:pd.DataFrame,finalists:list[str])->None:
    selected=metrics[((metrics.trial_id.isin(finalists))|(metrics.trial_id=="COMMON_B2"))&(metrics.quantity=="total_displacement")].copy();selected["model"]=selected.apply(lambda r:"B2" if r.trial_id=="COMMON_B2" else f"{r.trial_id}-{r.variant}",axis=1);frame=selected.groupby(["model","axis","band_low_hz","band_high_hz"],as_index=False).relative_energy_error.median();frame["band"]=frame.apply(lambda r:f"{r.band_low_hz:g}–{r.band_high_hz:g}",axis=1)
    fig,panels=plt.subplots(1,3,figsize=(13,4),sharey=False)
    for axis,panel in zip(AXES,panels):
        subset=frame[frame.axis==axis];pivot=subset.pivot(index="band",columns="model",values="relative_energy_error");pivot.plot.bar(ax=panel);panel.set_title(axis);panel.set_xlabel("Frequency band (Hz)");panel.set_ylabel("Median relative energy error");panel.tick_params(axis="x",rotation=35);panel.legend(fontsize=6)
    fig.suptitle("OOF spectral-energy error by frequency band");fig.tight_layout();_fig.save(fig,"F35","OOF spectral-energy error by frequency band","Median trajectory-level relative spectral-energy error. All saved-grid bands through Nyquist are retained; the 5 Hz boundary is reported, not used as a filter.",frame,{"units":"dimensionless","quantity":"total displacement spectral energy"})


def f36_kinematic(metrics:pd.DataFrame,finalists:list[str])->None:
    selected=metrics[(metrics.trial_id.isin(finalists))|(metrics.trial_id=="COMMON_FEM")].copy();candidate=selected[selected.trial_id!="COMMON_FEM"].groupby(["trial_id","variant","case_id","axis"],as_index=False).derivative_vs_velocity_relative_l2.median();floor=selected[selected.trial_id=="COMMON_FEM"][["trial_id","variant","case_id","axis","derivative_vs_velocity_relative_l2"]];frame=pd.concat([candidate,floor],ignore_index=True);frame["model"]=frame.apply(lambda r:"FEM/COMSOL saved-grid floor" if r.trial_id=="COMMON_FEM" else f"{r.trial_id}-{r.variant}",axis=1);models=list(dict.fromkeys(frame.model));fig,panels=plt.subplots(1,3,figsize=(12,4))
    for axis,panel in zip(AXES,panels):
        axis_rows=frame[frame.axis==axis];groups=[];labels=[]
        for model in models:
            values=axis_rows[axis_rows.model==model].derivative_vs_velocity_relative_l2.to_numpy(dtype=float);finite=values[np.isfinite(values)];groups.append(finite);labels.append(f"{model}\nn={len(finite)}/{len(values)}")
        panel.boxplot(groups,tick_labels=labels,showfliers=True);panel.set_title(axis);panel.set_ylabel("Relative L2: du/dt vs direct velocity");panel.tick_params(axis="x",rotation=30,labelsize=7);panel.set_ylim(bottom=0)
    fig.suptitle("Saved-grid kinematic consistency");fig.tight_layout();_fig.save(fig,"F36","Saved-grid kinematic consistency","Finite-difference displacement derivative versus direct velocity, assessed relative to the same calculation on FEM/COMSOL saved outputs. Boxplots use only mathematically defined relative errors and label valid/total counts; undefined zero-energy cases remain present in the source CSV. This is a consistency diagnostic, not the COMSOL integration algorithm.",frame,{"units":"dimensionless","quantity":"kinematic consistency","undefined_policy":"excluded from boxplot only; retained as null in source CSV with valid/total counts shown"})


def main()->None:
    report=DYNAMIC/"report.json"
    if not report.is_file():raise SystemExit("S12 dynamic/spatial audit is absent; figure generation made no changes")
    if json.loads(report.read_text(encoding="utf-8")).get("status")!="PASS_S12_DYNAMIC_SPATIAL_MULTISEED_AUDIT":raise RuntimeError("S12 dynamic/spatial audit did not pass")
    if (_fig.FIGURES/"F30.png").exists():raise FileExistsError("S12 dynamic figures already exist")
    _fig.style();context=resolve(ROOT);finalists=list(context.candidates);aggregate=load_aggregate(context);per_case=load_per_case(context);hotspot=pd.read_csv(DYNAMIC/"S12_SPATIAL_HOTSPOT_METRICS.csv");bands=pd.read_csv(DYNAMIC/"S12_BAND_ENERGY_METRICS.csv");kinematic=pd.read_csv(DYNAMIC/"S12_KINEMATIC_CONSISTENCY.csv");representatives={trial:representative_seed_and_case(trial,aggregate,per_case) for trial in finalists}
    f30_hotspot(hotspot,finalists,context)
    with h5py.File(DATASET,"r") as fem,h5py.File(B2,"r") as b2:
        node_frame=pd.concat([node_error_map(trial,representatives[trial][0],fem,context) for trial in finalists],ignore_index=True);f31_nodes(node_frame,finalists,context);f32_34(finalists,representatives,fem,b2,context)
    f35_bands(bands,finalists);f36_kinematic(kinematic,finalists)
    report_payload={"status":"PASS_S12_DYNAMIC_SPATIAL_FIGURES","figure_ids":["F30","F31","F32","F33","F34","F35","F36"],"representative_seed_case":{trial:{"seed":seed,"case_id":case} for trial,(seed,case) in representatives.items()},"evidence_mode":context.mode,"five_seed_claim_allowed":context.five_seed_claim_allowed,"training_or_tuning_performed":False}
    _fig.atomic_json(S12/"S12_DYNAMIC_SPATIAL_FIGURES_REPORT.json",report_payload);print(json.dumps(report_payload,indent=2))


if __name__=="__main__":main()
