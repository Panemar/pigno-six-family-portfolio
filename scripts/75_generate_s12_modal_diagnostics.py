#!/usr/bin/env python3
"""Generate gated structural-reference and OOF projected-response modal diagnostics F39-F41."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from s12_evidence_context import resolve

ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
S12 = ROOT / "s12_final_diagnostics"
AUDIT = S11 / "independent_oof_audit_v1"
DECISION = S11 / "S11_TO_S12_DECISION_V1.json"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
MODAL_ROOT = ROOT.parent.parent / "Full Data Extracción" / "Original_extractions_20260801" / "modal_original_v1"
FEM_MODAL = MODAL_ROOT / "comsol_modal_original.h5"
GRAPH_MODAL = MODAL_ROOT / "low_mode_audit" / "modal_solution.npz"
LOW_REPORT = MODAL_ROOT / "low_mode_audit" / "report.json"

_spec = importlib.util.spec_from_file_location("fig_utils", ROOT / "scripts" / "65_generate_s12_core_oof_figures.py")
_fig = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fig)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def unit_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= 0):
        raise RuntimeError("Zero modal column encountered")
    return matrix / norms


def structural_evidence() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    report = json.loads(LOW_REPORT.read_text(encoding="utf-8"))
    matches = report["first_12_mode_matches"]
    with h5py.File(FEM_MODAL, "r") as modal:
        frequencies = modal["frequencies_real_hz"][:12].astype(np.float64)
        fem = modal["mode_shapes_real"][:12].astype(np.float64).reshape(12, -1).T
        coords = modal["coords_m"][:].astype(np.float64)
    with np.load(GRAPH_MODAL, allow_pickle=False) as graph:
        graph_modes = graph["modes_response_translation"][:12].astype(np.float64).reshape(12, -1).T
    fem = unit_columns(fem)
    graph_modes = unit_columns(graph_modes)
    aligned = graph_modes.copy()
    # Isolated modes receive sign alignment. The near-degenerate 6-9 cluster is
    # aligned as a subspace by orthogonal Procrustes, never forced one-to-one.
    cluster = np.arange(5, 9)
    isolated = [index for index in range(12) if index not in cluster]
    for index in isolated:
        if float(fem[:, index] @ aligned[:, index]) < 0:
            aligned[:, index] *= -1
    u, _, vt = np.linalg.svd(aligned[:, cluster].T @ fem[:, cluster], full_matrices=False)
    aligned[:, cluster] = aligned[:, cluster] @ (u @ vt)
    aligned = unit_columns(aligned)
    mac = np.square(np.abs(fem.T @ graph_modes))
    frequency_rows = pd.DataFrame(matches)
    frequency_rows["evidence_type"] = "structural_graph_auditor_vs_FEM_COMSOL"
    frequency_rows["cluster_policy"] = ["subspace_6_9" if 6 <= int(mode) <= 9 else "isolated_MAC" for mode in frequency_rows.comsol_mode]
    products = np.sum(np.square(fem), axis=1) * np.sum(np.square(aligned), axis=1)
    numerators = np.square(np.sum(fem * aligned, axis=1))
    floor = float(np.max(products) * 1e-12)
    dof_comac = np.divide(numerators, products, out=np.full_like(numerators, np.nan), where=products > floor).reshape(512, 3)
    node_comac = np.full(512, np.nan, dtype=np.float64)
    for node in range(512):
        finite = dof_comac[node, np.isfinite(dof_comac[node])]
        if finite.size:
            node_comac[node] = float(np.min(finite))
    comac_rows = []
    for node in range(512):
        comac_rows.append({"node_zero_based": node, "X_m": coords[node, 0], "Y_m": coords[node, 1], "Z_m": coords[node, 2], "COMAC_X": dof_comac[node, 0], "COMAC_Y": dof_comac[node, 1], "COMAC_Z": dof_comac[node, 2], "node_min_COMAC": node_comac[node], "energy_floor": floor})
    return frequencies, fem, mac, frequency_rows, pd.DataFrame(comac_rows)


def candidate_path(trial: str, seed: int) -> Path:
    return resolve(ROOT).field_path(trial, "physics", seed)


def dominant_in_band(signal: np.ndarray, dt: float, center: float, half_width: float = 0.25) -> tuple[float, float]:
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size))) ** 2
    frequency = np.fft.rfftfreq(centered.size, dt)
    mask = (frequency >= center - half_width) & (frequency <= center + half_width)
    if not np.any(mask):
        return np.nan, 0.0
    local = np.flatnonzero(mask)
    index = int(local[np.argmax(spectrum[mask])])
    return float(frequency[index]), float(np.sum(spectrum[mask]))


def temporal_mac(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    candidate = candidate - np.mean(candidate)
    denominator = float(reference @ reference) * float(candidate @ candidate)
    return float((reference @ candidate) ** 2 / denominator) if denominator > 1e-30 else np.nan


def projected_response_evidence(finalists: list[str], frequencies: np.ndarray, fem_modes: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    projector = np.linalg.pinv(fem_modes)
    rows = []
    subspace_rows = []
    with h5py.File(DATASET, "r") as dataset:
        cases = decode(dataset["case_id"][:])
        time = dataset["time_s"][:].astype(np.float64)
        dt = float(np.median(np.diff(time)))
        for trial in finalists:
            for seed in resolve(ROOT).seeds:
                path = candidate_path(trial, seed)
                with h5py.File(path, "r") as candidate:
                    if decode(candidate["case_id"][:]) != cases:
                        raise RuntimeError(f"Modal candidate identity mismatch: {path}")
                    for case_index, case_id in enumerate(cases):
                        target = dataset["response/total_translation_m"][case_index].astype(np.float64).reshape(time.size, -1)
                        prediction = candidate["hybrid_total_displacement_m"][case_index].astype(np.float64).reshape(time.size, -1)
                        target -= np.mean(target, axis=0, keepdims=True)
                        prediction -= np.mean(prediction, axis=0, keepdims=True)
                        q_target = target @ projector.T
                        q_prediction = prediction @ projector.T
                        band = [dominant_in_band(q_target[:, mode], dt, frequencies[mode]) for mode in range(12)]
                        energies = np.asarray([item[1] for item in band])
                        energy_floor = float(np.max(energies) * 1e-6) if np.max(energies) > 0 else np.inf
                        admitted = energies >= energy_floor
                        for mode in range(12):
                            target_peak, target_energy = band[mode]
                            prediction_peak, _ = dominant_in_band(q_prediction[:, mode], dt, frequencies[mode])
                            frequency_error = abs(prediction_peak - target_peak) / max(abs(target_peak), 1e-12) if admitted[mode] else np.nan
                            rows.append({"trial_id": trial, "seed": seed, "case_id": case_id, "mode": mode + 1, "reference_structural_frequency_hz": frequencies[mode], "target_response_peak_hz": target_peak, "prediction_response_peak_hz": prediction_peak, "response_peak_relative_error": frequency_error, "temporal_modal_coordinate_MAC": temporal_mac(q_target[:, mode], q_prediction[:, mode]) if admitted[mode] else np.nan, "target_band_energy": target_energy, "case_relative_energy_floor": energy_floor, "energetically_admitted": bool(admitted[mode]), "interpretation": "forced-response coordinate projected on fixed FEM/COMSOL structural modes; not a learned eigenmode"})
                        active = np.flatnonzero(admitted)
                        if active.size:
                            qt, _ = np.linalg.qr(q_target[:, active], mode="reduced")
                            qp, _ = np.linalg.qr(q_prediction[:, active], mode="reduced")
                            singular = np.linalg.svd(qt.T @ qp, compute_uv=False)
                            subspace_rows.append({"trial_id": trial, "seed": seed, "case_id": case_id, "admitted_modes": int(active.size), "response_subspace_MAC_mean": float(np.mean(np.square(singular))), "response_subspace_MAC_min": float(np.min(np.square(singular)))})
    return pd.DataFrame(rows), pd.DataFrame(subspace_rows)


def f39(structural: pd.DataFrame, response: pd.DataFrame, finalists: list[str]) -> None:
    summary = response[response.energetically_admitted].groupby(["trial_id", "mode"]).response_peak_relative_error.agg(P50="median", P90=lambda x: np.nanpercentile(x, 90), count="count").reset_index()
    rows = structural[["comsol_mode", "comsol_frequency_hz", "graph_frequency_hz", "frequency_error_percent", "cluster_policy"]].rename(columns={"comsol_mode": "mode"}).copy()
    rows["series"] = "structural_graph_auditor"
    response_rows = summary.copy();response_rows["series"] = "OOF_projected_response";rows = pd.concat([rows, response_rows], ignore_index=True, sort=False)
    fig, panels = plt.subplots(1, 2, figsize=(12, 4.7))
    panels[0].axhline(0, color="#777777", lw=.8);panels[0].plot(structural.comsol_mode, structural.frequency_error_percent, color="#2463A6", marker="o");panels[0].set_xlabel("FEM/COMSOL mode index");panels[0].set_ylabel("Graph auditor frequency error (%)");panels[0].set_title("Common structural reference")
    for trial in finalists:
        subset = summary[summary.trial_id == trial];panels[1].plot(subset["mode"], 100 * subset.P50, marker="o", label=f"{trial} P50");panels[1].plot(subset["mode"], 100 * subset.P90, marker="x", linestyle="--", label=f"{trial} P90")
    panels[1].set_xlabel("Fixed FEM/COMSOL mode index");panels[1].set_ylabel("Near-modal response peak error (%)");panels[1].set_title("OOF forced-response projections");panels[1].legend(fontsize=7)
    fig.suptitle("Structural frequency and projected-response frequency diagnostics");fig.tight_layout();_fig.save(fig, "F39", "Structural frequency and projected-response frequency diagnostics", "Left: independent Timoshenko graph versus FEM/COMSOL eigenfrequencies, with modes 6–9 treated as a cluster. Right: PIGNO OOF response projected onto the fixed FEM/COMSOL modes; peak-frequency errors use only case/mode bands above a documented relative energy floor. The right panel is not a PIGNO eigenfrequency estimate.", rows, {"units": "Hz and percent", "structural_modes": 12, "response_band_half_width_hz": .25, "response_energy_floor": "1e-6 of maximum modal-band energy within each case"})


def f40(mac: np.ndarray, response: pd.DataFrame, subspace: pd.DataFrame, finalists: list[str]) -> None:
    response_summary = response[response.energetically_admitted].groupby(["trial_id", "mode"]).temporal_modal_coordinate_MAC.median().unstack(0).reindex(index=range(1, 13), columns=finalists)
    rows = [{"evidence": "structural_MAC", "FEM_mode": i + 1, "graph_mode": j + 1, "value": mac[i, j]} for i in range(12) for j in range(12)]
    rows += [{"evidence": "OOF_temporal_modal_coordinate_MAC", "trial_id": trial, "mode": int(mode), "value": float(value)} for mode, values in response_summary.iterrows() for trial, value in values.items()]
    rows += [{"evidence": "OOF_response_subspace_MAC", **record} for record in subspace.to_dict("records")]
    fig, panels = plt.subplots(1, 2, figsize=(12, 5.1));image = panels[0].imshow(mac, vmin=0, vmax=1, cmap="Blues", origin="lower");panels[0].set_xticks(range(12), range(1, 13));panels[0].set_yticks(range(12), range(1, 13));panels[0].set_xlabel("Independent graph mode");panels[0].set_ylabel("FEM/COMSOL mode");panels[0].set_title("Structural translation MAC");fig.colorbar(image, ax=panels[0], fraction=.046, pad=.04)
    image = panels[1].imshow(response_summary.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto", origin="lower");panels[1].set_xticks(range(len(finalists)), finalists, rotation=20, ha="right");panels[1].set_yticks(range(12), range(1, 13));panels[1].set_xlabel("Physics-informed finalist");panels[1].set_ylabel("Fixed FEM/COMSOL mode");panels[1].set_title("Median OOF temporal-coordinate MAC");fig.colorbar(image, ax=panels[1], fraction=.046, pad=.04)
    fig.suptitle("Structural MAC and OOF projected-response consistency");fig.tight_layout();_fig.save(fig, "F40", "Structural MAC and OOF projected-response consistency", "The structural MAC matrix compares FEM/COMSOL mode shapes with the independent Timoshenko graph auditor; the close 6–9 cluster is interpreted by subspace metrics, not forced diagonal identity. The OOF matrix measures temporal MAC after projecting forced responses onto fixed FEM/COMSOL modes and does not claim that PIGNO outputs eigenvectors.", pd.DataFrame(rows), {"units": "dimensionless", "structural_modes": 12, "cluster_modes": "6-9", "response_subspace_rows_in_source": True})


def f41(comac: pd.DataFrame) -> None:
    fig, panels = plt.subplots(1, 2, figsize=(12, 4.7));image = panels[0].scatter(comac.Z_m, comac.X_m, c=comac.node_min_COMAC, cmap="Blues", vmin=0, vmax=1, s=8);panels[0].set_aspect("equal", adjustable="box");panels[0].set_xlabel("Z longitudinal (m)");panels[0].set_ylabel("X transverse (m)");panels[0].set_title("Conservative node COMAC");fig.colorbar(image, ax=panels[0], fraction=.035, pad=.03, label="minimum X/Y/Z COMAC")
    panels[1].hist(comac.node_min_COMAC.dropna(), bins=np.linspace(0, 1, 31), color="#2463A6", edgecolor="white");panels[1].set_xlabel("Minimum translational COMAC at node");panels[1].set_ylabel("Observation-node count");panels[1].set_title("Spatial distribution")
    fig.suptitle("Structural COMAC over the 512 bridge observations");fig.tight_layout();_fig.save(fig, "F41", "Structural COMAC over the 512 bridge observations", "Coordinate modal assurance between the first 12 FEM/COMSOL modes and the independent Timoshenko graph modes. Isolated modes are sign-aligned; the near-degenerate 6–9 cluster is orthogonally aligned as a subspace before COMAC. Node color is the conservative minimum over X, Y and Z above the serialized energy floor. This audits the shared structural representation, not learned PIGNO eigenmodes.", comac, {"units": "m coordinates; dimensionless COMAC", "structural_modes": 12, "cluster_alignment": "orthogonal Procrustes for modes 6-9"})


def main() -> None:
    context = resolve(ROOT)
    ids = ["F39", "F40", "F41"]
    if any((_fig.FIGURES / f"{figure_id}.png").exists() for figure_id in ids):
        raise FileExistsError("One or more modal diagnostic figures already exist")
    finalists = list(context.candidates)
    _fig.style()
    frequencies, fem_modes, mac, structural, comac = structural_evidence()
    response, subspace = projected_response_evidence(finalists, frequencies, fem_modes)
    f39(structural, response, finalists);f40(mac, response, subspace, finalists);f41(comac)
    report = {"status": "PASS_S12_MODAL_DIAGNOSTIC_FIGURES", "figure_ids": ids, "finalists": finalists, "evidence_mode": context.mode, "seeds": list(context.seeds), "five_seed_claim_allowed": context.five_seed_claim_allowed, "structural_reference": "FEM/COMSOL eigenmodes versus independent Timoshenko graph auditor", "candidate_scope": "OOF forced-response projections on fixed FEM/COMSOL modes; not candidate eigenmodes", "response_energy_floor": "1e-6 of maximum modal-band energy within each case", "cluster_policy": "modes 6-9 treated by subspace alignment", "training_or_tuning_performed": False, "final_decision_authorized": False}
    _fig.atomic_json(S12 / "S12_MODAL_DIAGNOSTIC_FIGURES_REPORT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
