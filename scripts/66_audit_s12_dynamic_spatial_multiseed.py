#!/usr/bin/env python3
"""Compute full saved-grid dynamic/spatial metrics for all S11 OOF seeds."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from s12_evidence_context import resolve


ROOT=Path(__file__).resolve().parents[1];S10=ROOT/"s10_nested_grouped_oof";S11=ROOT/"s11_five_seed_confirmation";S12=ROOT/"s12_final_diagnostics"
DECISION=S11/"S11_TO_S12_DECISION_V1.json";FIELDS=S11/"independent_oof_audit_v1";DATASET=S10/"S10_ORIGINAL_68CASE_DATASET.h5";B2=S10/"b2_common_split_target_clean_v1"/"S10_B2_COMMON_SPLIT_OOF.h5"
OUTPUT=S12/"dynamic_spatial_multiseed_v1";STAGING=S12/"dynamic_spatial_multiseed_v1.incomplete";AXES="XYZ"
_spec=importlib.util.spec_from_file_location("dynamic_metrics",ROOT/"scripts"/"57_audit_s10_oof_dynamic_spatial_metrics.py");_dynamic=importlib.util.module_from_spec(_spec);assert _spec.loader is not None;_spec.loader.exec_module(_dynamic)


def decode(value)->str:return value.decode("utf-8") if isinstance(value,bytes) else str(value)


def write_csv(path:Path,rows:list[dict])->None:
    if not rows:raise RuntimeError(f"Empty S12 metric table: {path}")
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def atomic_json(path:Path,payload:dict)->None:
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");os.replace(temporary,path)


def add_metrics(common:dict,prediction:np.ndarray,target:np.ndarray,coords:np.ndarray,dt:float,spectral_rows:list,band_rows:list,spatial_rows:list)->None:
    spectral,bands=_dynamic.spectral_metrics(prediction,target,dt);spatial=_dynamic.spatial_metrics(prediction,target,coords);spectral_rows.append({**common,**spectral});band_rows.extend({**common,**band} for band in bands);spatial_rows.append({**common,**spatial})


def main()->None:
    if OUTPUT.exists():raise FileExistsError("S12 dynamic/spatial evidence already exists")
    if STAGING.exists():
        staging_entries=list(STAGING.iterdir())
        if any(entry.name.lower()!="desktop.ini" or not entry.is_file() for entry in staging_entries):raise FileExistsError("Non-empty S12 dynamic/spatial staging exists")
        for entry in staging_entries:entry.unlink()
        STAGING.rmdir()
    context=resolve(ROOT);finalists=list(context.candidates);STAGING.mkdir(parents=True,exist_ok=False)
    spectral_rows=[];band_rows=[];spatial_rows=[];kinematic_rows=[]
    with h5py.File(DATASET,"r") as fem,h5py.File(B2,"r") as b2:
        case_ids=[decode(value) for value in fem["case_id"][:]];time_s=fem["time_s"][:].astype(np.float64);dt=float(np.median(np.diff(time_s)));coords=fem["observation/coords_m"][:].astype(np.float64)
        b2_ids=[decode(value) for value in b2["case_id"][:]]
        if len(case_ids)!=68 or len(b2_ids)!=68 or len(set(case_ids))!=68 or len(set(b2_ids))!=68 or set(b2_ids)!=set(case_ids) or not np.allclose(np.diff(time_s),dt,rtol=0,atol=1e-12):raise RuntimeError("S12 dynamic authority mismatch")
        b2_index_by_case={case_id:index for index,case_id in enumerate(b2_ids)}
        # B2 total displacement is common to all finalists and seeds.
        for case_index,case in enumerate(case_ids):
            target=fem["response/total_translation_m"][case_index];prediction=b2["prediction_uvw_m"][b2_index_by_case[case]]
            for axis,axis_name in enumerate(AXES):add_metrics({"trial_id":"COMMON_B2","variant":"common","seed":-1,"case_id":case,"quantity":"total_displacement","axis":axis_name},prediction[:,:,axis],target[:,:,axis],coords,dt,spectral_rows,band_rows,spatial_rows)
        for trial in finalists:
            for variant in ("physics","control"):
                for seed in context.seeds:
                    path=context.field_path(trial,variant,seed)
                    with h5py.File(path,"r") as candidate:
                        if [decode(value) for value in candidate["case_id"][:]]!=case_ids:raise RuntimeError(f"S12 candidate identity mismatch: {path}")
                        for case_index,case in enumerate(case_ids):
                            target_total=fem["response/total_translation_m"][case_index];target_delta=fem["response/delta_translation_m"][case_index];target_velocity=fem["response/delta_velocity_mps"][case_index]
                            total=candidate["hybrid_total_displacement_m"][case_index];delta=candidate["delta_displacement_m"][case_index];velocity=candidate["delta_velocity_mps"][case_index]
                            for axis,axis_name in enumerate(AXES):
                                base={"trial_id":trial,"variant":variant,"seed":seed,"case_id":case,"axis":axis_name}
                                add_metrics({**base,"quantity":"total_displacement"},total[:,:,axis],target_total[:,:,axis],coords,dt,spectral_rows,band_rows,spatial_rows)
                                add_metrics({**base,"quantity":"incremental_velocity"},velocity[:,:,axis],target_velocity[:,:,axis],coords,dt,spectral_rows,band_rows,spatial_rows)
                                candidate_kinematic=_dynamic.kinematic_metrics(delta[:,:,axis],velocity[:,:,axis],time_s)
                                kinematic_rows.append({**base,"model":"candidate","quantity":"incremental_displacement_to_velocity",**candidate_kinematic})
                                if trial==finalists[0] and variant=="physics" and seed==context.seeds[0]:
                                    fem_kinematic=_dynamic.kinematic_metrics(target_delta[:,:,axis],target_velocity[:,:,axis],time_s)
                                    kinematic_rows.append({"trial_id":"COMMON_FEM","variant":"common","seed":-1,"case_id":case,"axis":axis_name,"model":"FEM_COMSOL_SAVED_GRID_FLOOR","quantity":"incremental_displacement_to_velocity",**fem_kinematic})
    for rows in (spectral_rows,band_rows,spatial_rows,kinematic_rows):
        for row in rows:
            for value in row.values():
                if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)):raise RuntimeError("Non-finite S12 dynamic/spatial metric")
    write_csv(STAGING/"S12_SPECTRAL_METRICS.csv",spectral_rows);write_csv(STAGING/"S12_BAND_ENERGY_METRICS.csv",band_rows);write_csv(STAGING/"S12_SPATIAL_HOTSPOT_METRICS.csv",spatial_rows);write_csv(STAGING/"S12_KINEMATIC_CONSISTENCY.csv",kinematic_rows)
    os.replace(STAGING,OUTPUT)
    report={"status":"PASS_S12_DYNAMIC_SPATIAL_MULTISEED_AUDIT","generated_utc":datetime.now(timezone.utc).isoformat(),"finalists":finalists,"seeds":list(context.seeds),"evidence_mode":context.mode,"five_seed_claim_allowed":context.five_seed_claim_allowed,"case_count":68,"same_case_time_node_global_axis":True,"saved_dt_s":dt,"nyquist_hz":.5/dt,"spectral_filtering":"none","bands_hz":_dynamic.BANDS_HZ,"velocity_source":"direct PIGNO/control output and direct FEM/COMSOL extraction","acceleration_computed":False,"B2_velocity_fabricated":False,"kinematic_scope":"saved-grid consistency relative to FEM/COMSOL saved-grid floor","undefined_relative_metric_policy":"null with explicit applicability flag when reference energy is at or below 1e-30; absolute metrics remain reported","training_or_tuning_performed":False}
    atomic_json(OUTPUT/"report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
