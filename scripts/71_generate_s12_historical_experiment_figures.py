#!/usr/bin/env python3
"""Generate source-backed historical campaign figures F07-F16 without fitting operators."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
S9 = ROOT / "s9_multifidelity_hpo"
_spec = importlib.util.spec_from_file_location("fig_utils", ROOT / "scripts" / "65_generate_s12_core_oof_figures.py")
_fig = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fig)
ROUTE_COLOR = "#2463A6"
ROUTE_STYLES = {"R1":"-", "R2":"--", "R3":"-.", "R4":":", "R5":(0,(5,2,1,2)), "R6":(0,(3,1,1,1))}
ROUTE_MARKERS = {"R1":"o", "R2":"s", "R3":"^", "R4":"D", "R5":"P", "R6":"X"}


def reports(pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted((S9 / "runs").glob(pattern)):
        report_path = path / "report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # The original R4 physics reports used the generic Physical32/Newmark
        # implementation and are preserved only as negative history.  Forward
        # portfolio figures must use the effective pH-OpInf repair.
        if (
            report.get("route") == "R4"
            and report.get("variant") == "physics"
            and "REPAIRED_EFFECTIVE_PH_OPINF" not in str(report.get("run_id", ""))
        ):
            continue
        metrics = report.get("validation_metrics", {})
        config = report.get("configuration", {})
        rows.append({
            "trial_id": report.get("trial_id"), "route": report.get("route"), "variant": report.get("variant"),
            "fidelity": report.get("fidelity"), "fold": report.get("fold"), "run_id": report.get("run_id"),
            "best_epoch": report.get("best_epoch"), "parameters": report.get("parameter_count"),
            "peak_vram_GiB": report.get("peak_vram_GiB"),
            "disp_X": metrics.get("displacement_X_pooled_l2"), "disp_Y": metrics.get("displacement_Y_pooled_l2"),
            "disp_Z": metrics.get("displacement_Z_pooled_l2"), "residual": metrics.get("equilibrium_residual_median"),
            **{f"hp_{key}": value for key, value in config.items() if key not in {"trial_id", "route", "variant"}},
        })
    return pd.DataFrame(rows)


def f07() -> None:
    raw = pd.read_csv(ROOT / "s5_oracle_floors" / "FOLD_CLEAN_PRIMARY_ORACLE_FLOORS.csv")
    metrics = ["pooled_relative_l2_floor", "p90_case_relative_l2_floor", "worst_case_relative_l2_floor"]
    frame = raw.groupby(["component", "rank"], as_index=False)[metrics].median()
    fig, panels = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
    for (component, group), panel in zip(frame.groupby("component"), panels):
        for metric, marker in zip(metrics, ["o", "s", "D"]):
            panel.plot(group["rank"], group[metric], marker=marker, label=metric.replace("_relative_l2_floor", ""))
        panel.set_title(component); panel.set_xlabel("Representation rank"); panel.set_yscale("log"); panel.set_xticks(sorted(group["rank"].unique()))
    panels[0].set_ylabel("Median fold-clean oracle relative L2 floor"); panels[-1].legend(fontsize=7); fig.suptitle("Representation oracle floors before learning"); fig.tight_layout()
    _fig.save(fig, "F07", "Representation oracle floors before learning", "Median fold-clean pooled, P90 and worst-case displacement oracle floors across all nested inner partitions. These are representation limits, not learned-model errors.", frame, {"units":"dimensionless", "quantity":"displacement representation floor", "fold_aggregation":"median"})


def f08() -> None:
    frame = pd.read_csv(ROOT / "s6_capacity_decisions" / "S6_PORTFOLIO_CAPACITY_SUMMARY.csv")
    frame["evidence_status"] = np.where(
        frame.route.str.startswith("R4_"),
        "historical_pre_effective_ph_opinf_repair_not_forward_valid",
        "historical_capacity_admitted",
    )
    axes = ["X", "Y", "Z"]
    fig, panel = plt.subplots(figsize=(9.2, 4.8))
    for index, row in frame.iterrows():
        values = np.array([row[f"displacement_{axis}_relative_l2"] for axis in axes], dtype=float)
        panel.plot([index, index], [values.min(), values.max()], color="#777777", lw=1.2)
        for offset, (axis, value) in zip([-.12, 0, .12], zip(axes, values)):
            panel.scatter(index + offset, value, label=axis if index == 0 else None, s=34, marker={"X":"o","Y":"s","Z":"D"}[axis], color=ROUTE_COLOR)
    panel.set_xticks(range(len(frame)), [route.split("_")[0] for route in frame.route]); panel.set_xlabel("Frozen route; full family names in source table"); panel.set_ylabel("One-case capacity displacement relative L2"); panel.set_ylim(bottom=0); panel.legend(title="Global component"); panel.set_title("Six-family one-case capacity comparison"); fig.tight_layout()
    _fig.save(fig, "F08", "Six-family one-case capacity comparison", "Historical one-case capacity outcomes for all six routes. The R4 point is retained and explicitly marked in source data as pre-effective-pH-OpInf evidence that is not valid for forward R4 decisions; its repaired route re-enters on the common micropanel and factorial panel. The vertical segment spans X/Y/Z. This is representability history, not generalization.", frame, {"units":"dimensionless", "quantity":"capacity displacement error", "case_count":1, "R4_claim_boundary":"historical pre-repair only"})


def f09() -> None:
    summary = pd.read_csv(ROOT / "s6_capacity_decisions" / "S6_PORTFOLIO_CAPACITY_SUMMARY.csv")
    rows = []
    for route_number in range(1, 7):
        registry = pd.read_csv(ROOT / "s6_capacity_decisions" / f"R{route_number}_CAPACITY_RUN_REGISTRY.csv")
        final_id = summary.loc[summary.route.str.startswith(f"R{route_number}_"), "final_capacity_run_id"].iloc[0]
        initial = registry.iloc[0]
        final = registry.loc[registry.run_id == final_id].iloc[0]
        for stage, record in [("initial", initial), ("after directed S6 repairs", final)]:
            rows.append({"route":f"R{route_number}", "campaign":"S6 capacity", "stage":stage, "run_id":record.run_id, "displacement_axis_sum":sum(float(record[f"displacement_{axis}_relative_l2"]) for axis in "XYZ"), "evidence_status":"historical_pre_effective_ph_opinf_repair_not_forward_valid" if route_number == 4 else "historical_admitted"})
    micro = pd.read_csv(ROOT / "s7_directed_repairs" / "S6_S7_MICROPANEL_RUN_REGISTRY.csv")
    for route, group in micro.groupby("route"):
        physics = group[group.role == "physics"].iloc[0]; optimization = group[group.role == "optimization"].iloc[0]
        for stage, record in [("physics", physics), ("cosine optimization repair", optimization)]:
            rows.append({"route":route, "campaign":"S7 micropanel", "stage":stage, "run_id":record.run_id, "displacement_axis_sum":sum(float(record[f"disp_{axis}_l2"]) for axis in "XYZ"), "evidence_status":"historical_pre_effective_ph_opinf_repair_not_forward_valid" if route == "R4" else "historical_admitted"})
    frame = pd.DataFrame(rows); fig, panels = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for panel, campaign in zip(panels, ["S6 capacity", "S7 micropanel"]):
        subset = frame[frame.campaign == campaign]
        stages = list(dict.fromkeys(subset.stage.tolist()))
        for route, group in subset.groupby("route"):
            group = group.set_index("stage").loc[stages].reset_index(); panel.plot(range(len(stages)), group.displacement_axis_sum, marker=ROUTE_MARKERS[route], linestyle=ROUTE_STYLES[route], label=route, color=ROUTE_COLOR)
        panel.set_xticks(range(len(stages)), stages, rotation=20, ha="right"); panel.set_ylabel("Sum of X/Y/Z displacement relative L2"); panel.set_title(campaign); panel.set_ylim(bottom=0)
    panels[-1].legend(ncol=2, fontsize=7); fig.suptitle("Directed repair effects under frozen comparisons"); fig.tight_layout()
    _fig.save(fig, "F09", "Directed repair effects under frozen comparisons", "Historical initial-to-final S6 capacity repairs and S7 physics-to-cosine optimization intervention. R4 rows are retained only as pre-effective-pH-OpInf negative history and are excluded from forward R4 decisions. Lower is better; a slope alone does not override the noncompensatory gates.", frame, {"units":"dimensionless", "quantity":"sum of displacement component errors", "R4_claim_boundary":"historical pre-repair only"})


def f10() -> None:
    raw = pd.read_csv(ROOT / "s8_factorial_panel" / "S8_RUN_REGISTRY_V3_REPAIRED_R4.csv")
    frame = raw.groupby(["route", "variant"], as_index=False)[["disp_X_l2", "disp_Y_l2", "disp_Z_l2"]].median()
    frame["model"] = frame.route + "-" + frame.variant
    matrix = frame.set_index("model")[["disp_X_l2", "disp_Y_l2", "disp_Z_l2"]]
    fig, panel = plt.subplots(figsize=(7.2, 7.2)); image = panel.imshow(matrix.to_numpy(), aspect="auto", cmap="magma_r")
    panel.set_xticks(range(3), ["X transverse", "Y vertical", "Z longitudinal"]); panel.set_yticks(range(len(matrix)), matrix.index, fontsize=7); panel.grid(False); panel.set_title("Factorial-panel median error across two seeds")
    threshold=.5*(float(matrix.to_numpy().min())+float(matrix.to_numpy().max()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): panel.text(j, i, f"{matrix.iloc[i,j]:.3f}", ha="center", va="center", fontsize=7, color="white" if matrix.iloc[i,j]>threshold else "#111111")
    fig.colorbar(image, ax=panel, label="Pooled displacement relative L2"); fig.tight_layout()
    _fig.save(fig, "F10", "Factorial-panel median error across two seeds", "Median pooled displacement errors across the two frozen S8 seeds for every route and matched variant. R6 modal is retained as its additional rank-matched comparator.", frame, {"units":"dimensionless", "quantity":"factorial-panel displacement error", "seed_aggregation":"median of two"})


def f11(protocol: dict) -> None:
    low = set(json.loads((S9 / "LOW_FIDELITY_PROMOTION.json").read_text(encoding="utf-8"))["promoted_trial_ids"])
    medium = set(json.loads((S9 / "MEDIUM_FIDELITY_PROMOTION.json").read_text(encoding="utf-8"))["promoted_trial_ids"])
    high = set(json.loads((S9 / "S9_MULTIFIDELITY_FINAL_AUDIT_V2_REPAIRED_R4.json").read_text(encoding="utf-8"))["promoted_trial_ids"])
    rows=[]
    for trial in protocol["trials"]:
        trial_id=trial["trial_id"]
        for stage, admitted in [("low", True), ("medium", trial_id in low), ("high", trial_id in medium), ("S10", trial_id in high)]: rows.append({"trial_id":trial_id,"route":trial["route"],"stage":stage,"present":admitted})
    frame=pd.DataFrame(rows);fig,panel=plt.subplots(figsize=(10,6));stage_index={name:i for i,name in enumerate(["low","medium","high","S10"])}
    route_base={route:i for i,route in enumerate(["R1","R2","R4","R6"])}
    for route, group in frame.groupby("route"):
        trial_ids=sorted(group.trial_id.unique())
        for offset,trial_id in enumerate(trial_ids):
            subset=group[(group.trial_id==trial_id)&group.present];x=[stage_index[s] for s in subset.stage];y=np.full(len(x),route_base[route]+(offset-3.5)*.035);panel.plot(x,y,marker=ROUTE_MARKERS[route],ms=3,lw=.7,linestyle=ROUTE_STYLES[route],color=ROUTE_COLOR,alpha=.72)
    panel.set_xticks(range(4),["32 configs\nlow","12 configs\nmedium","4 configs\nhigh","3 promoted\nS10"]);panel.set_yticks(list(route_base.values()),list(route_base));panel.set_xlim(-.15,3.15);panel.set_title("Successive-halving promotion history");panel.set_ylabel("Frozen family");fig.tight_layout()
    _fig.save(fig,"F11","Successive-halving promotion history","All 32 deterministic Latin-hypercube configurations and their exact promotion path through low, medium and high fidelity. Absence at a later stage means pruning under the frozen lexicographic rule.",frame,{"units":"categorical","quantity":"multifidelity promotion"})


def low_trial_scores(protocol: dict) -> pd.DataFrame:
    low=reports("S9_LOW_*_PHYSICS_*")
    low["score"]=low[["disp_X","disp_Y","disp_Z"]].max(axis=1)
    score=low.groupby(["trial_id","route"],as_index=False).score.mean()
    config=pd.DataFrame(protocol["trials"])
    return config.merge(score,on=["trial_id","route"],how="inner")


def f12(frame: pd.DataFrame) -> None:
    columns=["width","graph_depth","temporal_modes","temporal_kernel","temporal_blocks","head_hidden","learning_rate","weight_decay","velocity_data_weight","state_loss_weight","equilibrium_loss_weight","gradient_clip","score"]
    normalized=frame[["trial_id","route","scheduler",*columns]].copy()
    for column in columns:
        values=normalized[column].astype(float);normalized[column]=(values-values.min())/max(values.max()-values.min(),1e-30)
    promoted=set(json.loads((S9/"MEDIUM_FIDELITY_PROMOTION.json").read_text(encoding="utf-8"))["promoted_trial_ids"])
    fig,panel=plt.subplots(figsize=(13,5));x=np.arange(len(columns))
    for _,row in normalized.iterrows(): panel.plot(x,row[columns].to_numpy(float),color=ROUTE_COLOR,linestyle=ROUTE_STYLES[row.route],alpha=.30,lw=2 if row.trial_id in promoted else .7)
    panel.set_xticks(x,columns,rotation=35,ha="right",fontsize=7);panel.set_ylabel("Min-max coordinate within 32-trial design");panel.set_title("S9 hyperparameter parallel coordinates; thick lines reached high fidelity");panel.set_ylim(-.03,1.03);fig.tight_layout()
    _fig.save(fig,"F12","S9 hyperparameter parallel coordinates","Normalized coordinates for the 32 deterministic Latin-hypercube trials. Line color denotes route and thick lines denote the four configurations promoted to high fidelity. Score is mean low-fidelity worst-axis pooled displacement L2.",frame,{"units":"mixed; raw values in source CSV","quantity":"S9 hyperparameter design"})


def f13(frame: pd.DataFrame) -> None:
    features=["width","graph_depth","temporal_modes","temporal_kernel","temporal_blocks","head_hidden","learning_rate","weight_decay","velocity_data_weight","state_loss_weight","equilibrium_loss_weight","gradient_clip","scheduler"]
    X=frame[features].copy();X["scheduler"]=(X.scheduler=="cosine").astype(int);y=frame.score.to_numpy(float);rows=[]
    for fold,(train,test) in enumerate(KFold(n_splits=4,shuffle=True,random_state=20260812).split(X)):
        model=RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=20260812+fold,n_jobs=1).fit(X.iloc[train],y[train]);base=np.mean(np.abs(model.predict(X.iloc[test])-y[test]));rng=np.random.default_rng(20260812+fold)
        for feature in features:
            for repeat in range(30):
                permuted=X.iloc[test].copy();permuted[feature]=rng.permutation(permuted[feature].to_numpy());mae=np.mean(np.abs(model.predict(permuted)-y[test]));rows.append({"feature":feature,"fold":fold,"repeat":repeat,"delta_MAE":mae-base,"baseline_MAE":base})
    importance=pd.DataFrame(rows);summary=importance.groupby("feature",as_index=False).delta_MAE.agg(["mean","std"]).reset_index().sort_values("mean")
    fig,panel=plt.subplots(figsize=(8.5,5));panel.barh(summary.feature,summary["mean"],xerr=summary["std"],color="#2463A6",alpha=.75);panel.axvline(0,color="#333333",lw=.8);panel.set_xlabel("Cross-validated permutation increase in MAE");panel.set_title("Exploratory S9 hyperparameter importance (n=32)");fig.tight_layout()
    _fig.save(fig,"F13","Exploratory S9 hyperparameter importance","Four-fold cross-validated random-forest permutation importance with 30 test-fold permutations. The 32-trial sample is small and interactions are confounded; rankings are diagnostic associations, not causal effects.",importance,{"units":"absolute increase in low-fidelity score MAE","quantity":"permutation importance","trial_count":len(frame)})


def f14() -> None:
    frame=reports("S9_HIGH_*");frame=frame.dropna(subset=["disp_X","disp_Y","disp_Z","residual","parameters"]);frame["predictive_error"]=frame[["disp_X","disp_Y","disp_Z"]].max(axis=1)
    frame["nondominated"]=True
    values=frame[["predictive_error","residual","parameters"]].to_numpy(float)
    for i in range(len(frame)):
        frame.iloc[i,frame.columns.get_loc("nondominated")]=not any(np.all(values[j]<=values[i]) and np.any(values[j]<values[i]) for j in range(len(frame)) if j!=i)
    fig,panel=plt.subplots(figsize=(8.5,5.2))
    for (route,variant),group in frame.groupby(["route","variant"]):
        color="#2463A6" if variant=="physics" else "#D18B20";panel.scatter(group.predictive_error,group.residual,s=20+80*(group.parameters/group.parameters.max()),facecolors=color if variant=="physics" else "none",edgecolors=color,marker=ROUTE_MARKERS[route],alpha=.75,label=f"{route}-{variant}")
    front=frame[frame.nondominated];panel.scatter(front.predictive_error,front.residual,s=150,facecolors="none",edgecolors="#111111",lw=1.3,label="nondominated");panel.set_yscale("log");panel.set_xlabel("Worst-axis pooled displacement relative L2");panel.set_ylabel("Equilibrium residual median");panel.set_title("S9 high-fidelity error–physics–cost Pareto view");panel.legend(fontsize=6,ncol=2);fig.tight_layout()
    _fig.save(fig,"F14","S9 high-fidelity error–physics–cost Pareto view","Each point is one high-fidelity fold run. Marker area scales with parameter count; open squares are controls and filled circles physics variants. Black rings mark nondominated points across predictive error, residual and parameter count.",frame,{"units":"dimensionless errors; parameter count","quantity":"high-fidelity Pareto diagnostics"})


def selected_progress() -> pd.DataFrame:
    frames=[]
    for trial_id in ["R1_LHS_07","R2_LHS_02","R4_LHS_03","R6_LHS_04"]:
        route=trial_id.split("_")[0];repair="_REPAIRED_EFFECTIVE_PH_OPINF" if route=="R4" else "";path=S9/"runs"/f"S9_HIGH_{trial_id}_FOLD_0_PHYSICS{repair}_SEED_20260812"/"live_progress.csv";data=pd.read_csv(path);data["trial_id"]=trial_id;data["route"]=route;data["fold"]=0;frames.append(data)
    return pd.concat(frames,ignore_index=True)


def f15(frame: pd.DataFrame) -> None:
    fig,panels=plt.subplots(1,3,figsize=(12,3.8),sharey=False)
    for axis,panel in zip("XYZ",panels):
        for trial,group in frame.groupby("trial_id"): panel.plot(group.epoch,group[f"val_{axis}_l2"],label=trial,color=ROUTE_COLOR,linestyle=ROUTE_STYLES[group.route.iloc[0]],marker=ROUTE_MARKERS[group.route.iloc[0]],markevery=6,ms=3)
        panel.set_title(axis);panel.set_xlabel("Epoch");panel.set_ylabel("Validation pooled relative L2");panel.set_ylim(bottom=0)
    panels[-1].legend(fontsize=7);fig.suptitle("S9 high-fidelity fold-0 validation convergence");fig.tight_layout()
    _fig.save(fig,"F15","S9 high-fidelity fold-0 validation convergence","Validation trajectories for the four high-fidelity configurations on the same fold. This convergence view is historical internal-selection evidence and is not OOF or a comparison to the external S10 folds.",frame,{"units":"dimensionless","quantity":"validation displacement convergence","fold":0})


def f16(frame: pd.DataFrame) -> None:
    source=frame[["trial_id","route","fold","epoch","train_data_loss","train_physics_loss"]].copy();source["data_gradient_norm_available"]=False;source["physics_gradient_norm_available"]=False;source["gradient_cosine_available"]=False
    fig,panels=plt.subplots(1,2,figsize=(10.8,4))
    for trial,group in source.groupby("trial_id"):
        valid=group[group.epoch>0];style=ROUTE_STYLES[group.route.iloc[0]];marker=ROUTE_MARKERS[group.route.iloc[0]];panels[0].plot(valid.epoch,valid.train_data_loss,label=trial,color=ROUTE_COLOR,linestyle=style,marker=marker,markevery=6,ms=3);panels[1].plot(valid.epoch,valid.train_physics_loss,label=trial,color=ROUTE_COLOR,linestyle=style,marker=marker,markevery=6,ms=3)
    panels[0].set_title("Recorded data term");panels[1].set_title("Recorded physics term");
    for panel in panels: panel.set_xlabel("Epoch");panel.set_ylabel("Training loss");panel.set_yscale("log")
    panels[1].legend(fontsize=7);fig.suptitle("Recorded S9 loss terms; gradient diagnostics unavailable");fig.tight_layout()
    _fig.save(fig,"F16","Recorded S9 loss terms; gradient diagnostics unavailable","Only data and physics loss terms were serialized per epoch for these runs. Per-term gradient norms and gradient cosines were not recorded and are explicitly unavailable; no values are reconstructed or imputed.",source,{"units":"training objective units","quantity":"recorded loss terms","gradient_diagnostics":"unavailable in source logs"})


def main() -> None:
    if any((_fig.FIGURES/f"F{index:02d}.png").exists() for index in range(7,17)):
        raise FileExistsError("One or more F07-F16 outputs already exist")
    _fig.style();protocol=json.loads((S9/"S9_MULTIFIDELITY_HPO_PROTOCOL.json").read_text(encoding="utf-8"));f07();f08();f09();f10();f11(protocol);trial_scores=low_trial_scores(protocol);f12(trial_scores);f13(trial_scores);f14();progress=selected_progress();f15(progress);f16(progress)
    report={"status":"PASS_S12_HISTORICAL_EXPERIMENT_FIGURES","figure_ids":[f"F{index:02d}" for index in range(7,17)],"training_or_tuning_performed":False,"OOF_or_final_decision_authorized":False}
    _fig.atomic_json(ROOT/"s12_final_diagnostics"/"S12_HISTORICAL_EXPERIMENT_FIGURES_REPORT.json",report);print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
