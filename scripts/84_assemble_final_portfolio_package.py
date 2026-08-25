#!/usr/bin/env python3
"""Assemble the thesis package from admitted portfolio evidence only."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from visual_qa_validation import validate_manual_visual_qa

ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
S12 = ROOT / "s12_final_diagnostics"
S14 = ROOT / "s14_final_decision"
FINAL = ROOT / "thesis_physics_informed_operator_portfolio_final"
STAGING = ROOT / "thesis_physics_informed_operator_portfolio_final.incomplete"
DECISION = S14 / "S14_FINAL_SCIENTIFIC_DECISION.json"
FIGURE_REPORT = S14 / "S14_FINAL_DECISION_FIGURES_REPORT.json"

REPORT_NAMES = [
    "FINAL_PORTFOLIO_REPORT.md", "PHYSICS_INFORMED_FAMILY_COMPARISON.md", "LEGACY_RESULTS_REPORT.md",
    "DATA_AND_GRAPH_AUTHORITY_REPORT.md", "LOAD_CONTRACT_REPORT.md", "MODAL_VERIFICATION_REPORT.md",
    "MULTIOPERATOR_REPORT.md", "GALERKIN_VARIATIONAL_REPORT.md", "PORT_HAMILTONIAN_ENERGY_REPORT.md",
    "ROTATION_MULTISCALE_GRAPH_REPORT.md", "LOAD_DEPENDENT_ROM_REPORT.md", "BRIDGE_PINO_REPLICATION_REPORT.md",
    "HYPERPARAMETER_CALIBRATION_REPORT.md", "FULL_OOF_METRICS_REPORT.md", "RESULTS_INTERPRETATION.md",
    "THESIS_EVIDENCE_MAP.md", "CLAIMS_AND_LIMITATIONS.md", "NEGATIVE_RESULTS.md", "FINAL_DECISION.md",
]
ROUTE_REPORTS = {
    "R1": "BRIDGE_PINO_REPLICATION_REPORT.md", "R2": "MULTIOPERATOR_REPORT.md",
    "R3": "GALERKIN_VARIATIONAL_REPORT.md", "R4": "PORT_HAMILTONIAN_ENERGY_REPORT.md",
    "R5": "ROTATION_MULTISCALE_GRAPH_REPORT.md", "R6": "LOAD_DEPENDENT_ROM_REPORT.md",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No hay filas admitidas."
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame.astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows)


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)


def link_or_pointer(source: Path, destination: Path, registry: list[dict]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source); mode = "hardlink"
    try:
        os.link(source, destination)
    except OSError:
        mode = "external_pointer"
        pointer = destination.with_suffix(destination.suffix + ".pointer.json")
        write(pointer, json.dumps({"source": str(source), "sha256": digest, "size_bytes": source.stat().st_size, "reason": "hardlink unavailable; source remains immutable in project root"}, indent=2, ensure_ascii=False))
    registry.append({"category": destination.parent.name, "package_target": str(destination.relative_to(STAGING)), "source": str(source), "sha256": digest, "size_bytes": source.stat().st_size, "storage_mode": mode})


def common_header(decision: dict, title: str) -> str:
    return f"# {title}\n\n- Estado científico: `{decision['final_state']}`.\n- Referencia: modelo FEM implementado y resuelto en COMSOL.\n- Evidencia: {decision['evidence_label']}.\n- Rev7/Rev8, sensores y un panel FEM nuevo no forman parte de esta evidencia.\n"


def main() -> None:
    if FINAL.exists() or STAGING.exists():
        raise FileExistsError("Final package or incomplete staging package already exists")
    if not DECISION.is_file() or not FIGURE_REPORT.is_file():
        raise SystemExit("S14 decision or F44-F45 report is absent; packaging made no changes")
    if read(DECISION).get("status") != "PASS_S14_FINAL_SCIENTIFIC_DECISION" or read(FIGURE_REPORT).get("status") != "PASS_S14_FINAL_DECISION_FIGURES":
        raise RuntimeError("S14 decision and F44-F45 must be admitted before packaging")
    final_visual_qa = S14 / "S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1.json"
    validate_manual_visual_qa(final_visual_qa,S14/"S14_FINAL_DECISION_VISUAL_QA_READINESS_V1.json","PASS_S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1",["F44","F45"],S12/"figures")
    missing_figures = [number for number in range(1, 46) if not (S12 / "figures" / f"F{number:02d}.png").is_file()]
    if missing_figures:
        raise RuntimeError(f"Required F01-F45 figures are incomplete: {missing_figures}")

    decision = read(DECISION)
    gates = pd.read_csv(S14 / "S14_FAMILY_GATE_MATRIX.csv")
    claims = pd.read_csv(S14 / "S14_CLAIM_EVIDENCE_LIMITATION.csv")
    portfolio = read(ROOT / "PORTFOLIO_DEFINITION.json")
    promotion = read(S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json")
    family_matrix = pd.read_csv(ROOT / "PORTFOLIO_FAMILY_MATRIX.csv")
    s8_payload = read(ROOT / "s8_factorial_panel" / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION_V3_REPAIRED_R4.json")
    s8_table = pd.DataFrame(s8_payload["families"])
    s9_payload = read(ROOT / "audits" / "S9_PORTFOLIO_REPAIRED_R4_INDEPENDENT_AUDIT_V1.json")
    s9_table = pd.DataFrame([{key: row.get(key) for key in ("trial_id", "route", "noninferior_folds", "physical_ratio_worst")} for row in s9_payload["ranking"]])
    s10_table = pd.DataFrame([{key: row.get(key) for key in ("trial_id", "route", "eligible_for_S11", "noninferior_to_B2_all_axes", "noninferior_to_capacity_matched_control_all_axes", "predictive_material_gain", "physical_material_gain", "bootstrap_positive_axes", "median_equilibrium_residual_reduction")} for row in promotion["decisions"]])
    if decision.get("diagnostic_evidence_mode") == "S11 five-seed finalists":
        oof_source = pd.read_csv(S11 / "independent_oof_audit_v1" / "S11_OOF_AGGREGATE_BY_SEED.csv")
        oof_selected = oof_source[(oof_source.variant == "physics") & (oof_source.quantity == "total_displacement")]
        candidate_oof = oof_selected.groupby(["trial_id", "axis"], as_index=False).agg(seed_median_pooled_relative_l2=("pooled_relative_l2", "median"), seed_P90_pooled_relative_l2=("pooled_relative_l2", lambda values: values.quantile(0.9)), seeds=("seed", "nunique"))
        b2_oof = oof_source[(oof_source.trial_id == "COMMON_B2") & (oof_source.quantity == "total_displacement")][["trial_id", "axis", "pooled_relative_l2"]].copy(); b2_oof["seed_median_pooled_relative_l2"] = b2_oof.pop("pooled_relative_l2"); b2_oof["seed_P90_pooled_relative_l2"] = b2_oof["seed_median_pooled_relative_l2"]; b2_oof["seeds"] = 1
        oof_table = pd.concat([candidate_oof, b2_oof], ignore_index=True)
    else:
        oof_source = pd.read_csv(S10 / "independent_oof_audit_v1" / "S10_OOF_AGGREGATE_METRICS.csv")
        oof_table = oof_source[(oof_source.variant == "physics") & (oof_source.quantity == "displacement") & (oof_source.view == "total") & (oof_source.model.isin(["S10_HYBRID", "B2"]))][["trial_id", "model", "axis", "pooled_relative_l2", "mean_relative_l2", "p90_relative_l2", "worst_relative_l2"]].copy()
    STAGING.mkdir(parents=True)
    for directory in ("families", "hybrids", "configs", "src", "scripts", "tests", "checkpoints", "predictions_oof", "metrics", "tables", "figures", "logs", "reports", "manifests", "legacy_links"):
        (STAGING / directory).mkdir()

    for name in ("CURRENT_STATE_PORTFOLIO.md", "LEGACY_EXPERIMENT_LEDGER.csv", "DATA_AUTHORITY_AUDIT.md", "ACTIVE_BEAM_GRAPH_AUDIT.md", "LOAD_AND_BASE_STATE_AUDIT.md", "MODAL_REFERENCE_AUDIT.md", "SOURCE_AUDIT_PLAN.md", "PORTFOLIO_FAMILY_MATRIX.csv", "FAMILY_NONREDUNDANCY_REPORT.md", "METHOD_ADOPTION_PROTOCOL.md", "REPAIR_BUDGET.json", "PORTFOLIO_EXECUTION_DAG.json", "SPLIT_AND_OOF_PROTOCOL.json", "COMPUTE_BUDGET.json", "S0_PORTFOLIO_DECISION.json", "PORTFOLIO_DEFINITION.json", "EXPERIMENT_CONTRACT.json", "ACCEPTANCE_GATES.json", "NUMERICAL_AUTHORITY_DECISION.json", "SOURCE_TRANSFER_MATRIX.csv", "ACTIVE_BEAM_GRAPH.json", "LOAD_AND_BASE_STATE_CONTRACT.json", "MODAL_REFERENCE_CONTRACT.h5", "SPLIT_MANIFEST.json", "DATA_ACCESS_REGISTRY.csv", "DECISION_LOG.md", "FAILURE_LOG.md"):
        source = ROOT / name; copy_if_exists(source, STAGING / name)
    copy_if_exists(ROOT / "source_audit" / "SOURCE_TRANSFER_MATRIX.csv", STAGING / "SOURCE_TRANSFER_MATRIX.csv")
    copy_if_exists(ROOT / "source_audit" / "SOURCE_DECISION_REPORT.md", STAGING / "reports" / "SOURCE_DECISION_REPORT.md")
    for source in (ROOT / "scripts").glob("*"):
        if source.is_file() and source.suffix.lower() in {".py", ".ps1", ".md"}: shutil.copy2(source, STAGING / "scripts" / source.name)
    if (ROOT / "src").exists():
        for source in (ROOT / "src").rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc": copy_if_exists(source, STAGING / "src" / source.relative_to(ROOT / "src"))
    family_sources = {
        "R1_bridge_pino.py": "bridge_pino.py", "R2_mo_pigno.py": "mo_pigno.py",
        "R3_graph_galerkin.py": "graph_galerkin.py", "R4_port_hamiltonian.py": "port_hamiltonian.py",
        "R5_rotation_multiscale.py": "rotation_multiscale.py", "R6_ritz_krylov.py": "ritz_krylov.py",
    }
    for destination_name, source_name in family_sources.items():
        copy_if_exists(ROOT / "src" / "portfolio_operators" / source_name, STAGING / "families" / destination_name)
    for source_name in ("common.py", "capacity_data.py", "micropanel_heads.py"):
        copy_if_exists(ROOT / "src" / "portfolio_operators" / source_name, STAGING / "hybrids" / source_name)
    for source in (ROOT / "configs").glob("**/*") if (ROOT / "configs").exists() else []:
        if source.is_file():copy_if_exists(source, STAGING / "configs" / source.relative_to(ROOT / "configs"))
    config_candidates = set(ROOT.glob("*.json"))
    config_candidates.update(ROOT.rglob("*PROTOCOL*.json")); config_candidates.update(ROOT.rglob("*CONTRACT*.json"))
    for source in sorted(config_candidates):
        if source.is_file() and STAGING not in source.parents and source.stat().st_size < 16 * 1024 * 1024:
            copy_if_exists(source, STAGING / "configs" / source.relative_to(ROOT))
    for source in (ROOT / "tests").glob("**/*") if (ROOT / "tests").exists() else []:
        if source.is_file():copy_if_exists(source, STAGING / "tests" / source.relative_to(ROOT / "tests"))
    for source in (ROOT / "audits").glob("*.junit.xml"):
        copy_if_exists(source, STAGING / "tests" / "reports" / source.name)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": dict(sorted((distribution.metadata.get("Name", "unknown"), distribution.version) for distribution in importlib.metadata.distributions())),
        "note": "Captured from the interpreter that assembled the admitted final package.",
    }
    try:
        import torch
        environment["torch_cuda"] = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda, "device_count": torch.cuda.device_count(), "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}
    except Exception as error:
        environment["torch_cuda"] = {"inspection_error": type(error).__name__}
    write(STAGING / "configs" / "ENVIRONMENT_LOCK.json", json.dumps(environment, indent=2, ensure_ascii=False))
    shutil.copytree(S12 / "figures", STAGING / "figures", dirs_exist_ok=True)
    for source_dir, destination in ((S12 / "figure_data", STAGING / "tables" / "figure_data"), (S12 / "captions", STAGING / "figures" / "captions"), (S12 / "figure_manifests", STAGING / "manifests" / "figures")):
        if source_dir.exists():shutil.copytree(source_dir, destination, dirs_exist_ok=True)

    for source in ROOT.rglob("*.csv"):
        if STAGING in source.parents or source.stat().st_size > 256 * 1024 * 1024: continue
        relative = source.relative_to(ROOT); destination = STAGING / "tables" / relative
        copy_if_exists(source, destination)
    for source in ROOT.rglob("*.json"):
        if STAGING in source.parents or source.name.endswith("status.json") or source.stat().st_size > 64 * 1024 * 1024: continue
        copy_if_exists(source, STAGING / "metrics" / source.relative_to(ROOT))
    for source in list(ROOT.rglob("*.jsonl")) + list(ROOT.rglob("*.log")):
        if STAGING in source.parents: continue
        copy_if_exists(source, STAGING / "logs" / source.relative_to(ROOT))

    binary_registry: list[dict] = []
    trials = [row["trial_id"] for row in decision.get("ordered_candidates", [])]
    evidence_root = S11 if decision.get("diagnostic_evidence_mode") == "S11 five-seed finalists" else S10
    for source in (evidence_root / "runs").glob("S10_OUTER_*"):
        if trials and not any(trial in source.name for trial in trials): continue
        for filename, category in (("best_checkpoint.pt", "checkpoints"), ("final_checkpoint.pt", "checkpoints"), ("predictions.h5", "predictions_oof")):
            candidate = source / filename
            if candidate.is_file():link_or_pointer(candidate, STAGING / category / source.name / candidate.name, binary_registry)
    audit_dir = S11 / "independent_oof_audit_v1" if evidence_root == S11 else S10 / "independent_oof_audit_v1"
    for source in audit_dir.glob("*OOF_FIELDS.h5"):
        if trials and not any(trial in source.name for trial in trials): continue
        link_or_pointer(source, STAGING / "predictions_oof" / source.name, binary_registry)
    with (STAGING / "manifests" / "BINARY_ARTIFACT_REGISTRY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(binary_registry[0]) if binary_registry else ["category", "package_target", "source", "sha256", "size_bytes", "storage_mode"]); writer.writeheader(); writer.writerows(binary_registry)

    gate_table = md_table(gates)
    claim_table = md_table(claims)
    family_table_md = md_table(family_matrix[["route_id", "canonical_name", "spatial_representation", "temporal_representation", "physics_mechanism", "primary_state", "mandatory_data_only_control"]])
    s8_table_md = md_table(s8_table[["route", "family", "primary_seed_count", "worst_pooled_over_seeds", "worst_P90_over_seeds", "worst_case_over_seeds", "worst_physical_residual_ratio_over_seeds", "parameters", "promotion"]])
    s9_table_md = md_table(s9_table)
    s10_table_md = md_table(s10_table)
    oof_table_md = md_table(oof_table)
    reports = STAGING / "reports"
    write(reports / "FINAL_PORTFOLIO_REPORT.md", common_header(decision, "Informe final del portafolio PIGNO") + f"\n## Dictamen\n\n`{decision['final_state']}`\n\n## Definición de las seis familias\n\n{family_table_md}\n\n## Panel factorial S8\n\n{s8_table_md}\n\n## Selección S9\n\n{s9_table_md}\n\n## Decisión S10→S11\n\n{s10_table_md}\n\n## Evidencia OOF admitida\n\n{oof_table_md}\n\n## Comparación no compensatoria final\n\n{gate_table}\n")
    write(reports / "PHYSICS_INFORMED_FAMILY_COMPARISON.md", common_header(decision, "Comparación de familias physics-informed") + "\nLas seis familias congeladas se comparan bajo la misma autoridad, split, presupuesto y puertas. 'Not reached' no se reetiqueta como fallo experimental y ninguna mejora física compensa una violación predictiva.\n\n## Formulaciones\n\n" + family_table_md + "\n\n## Evidencia S8\n\n" + s8_table_md + "\n\n## Evidencia S9\n\n" + s9_table_md + "\n\n## Gates finales\n\n" + gate_table)
    write(reports / "LEGACY_RESULTS_REPORT.md", common_header(decision, "Resultados históricos preservados") + "\nEl ledger histórico se conserva sin reetiquetar evidencia previa como datos nuevos. Rev7 y Rev8 quedan excluidos como autoridad científica. Véase `LEGACY_EXPERIMENT_LEDGER.csv`.\n")
    write(reports / "DATA_AND_GRAPH_AUTHORITY_REPORT.md", common_header(decision, "Autoridad de datos y grafo") + "\nLa autoridad única es el FEM/COMSOL original y el grafo Beam activo auditado. Las comparaciones mantienen caso, tiempo, nodo, componente, unidades y ejes X transversal, Y vertical y Z longitudinal.\n")
    write(reports / "LOAD_CONTRACT_REPORT.md", common_header(decision, "Contrato de cargas") + "\nLas entradas conservan causalidad, riel activo, configuración factorial, velocidad y cargas del caso. El FEM/COMSOL es referencia numérica; no se compara COMSOL contra FEM como si fueran fuentes distintas.\n")
    write(reports / "MODAL_VERIFICATION_REPORT.md", common_header(decision, "Verificación modal") + "\nSe distinguen los modos estructurales FEM/COMSOL, su auditor Timoshenko independiente y las coordenadas de respuesta forzada proyectadas. La POD de respuesta no se declara modo estructural ni se atribuyen eigenpares a la PIGNO.\n")
    route_descriptions = {row["id"].split("_")[0]: row for row in portfolio["routes"]}
    for route, report_name in ROUTE_REPORTS.items():
        row = gates[gates.route == route].iloc[0].to_dict(); description = route_descriptions.get(route, {}); formulation = family_matrix[family_matrix.route_id.str.startswith(route + "_")]; s8_route = s8_table[s8_table.route == route]; s9_route = s9_table[s9_table.route == route]; s10_route = s10_table[s10_table.route == route]
        write(reports / report_name, common_header(decision, f"Ruta {route}") + f"\n## Definición congelada\n\n```json\n{json.dumps(description, indent=2, ensure_ascii=False)}\n```\n\n## Representación, física y control\n\n{md_table(formulation)}\n\n## Panel factorial S8\n\n{md_table(s8_route)}\n\n## Búsqueda multifidelidad S9\n\n{md_table(s9_route)}\n\n## OOF S10\n\n{md_table(s10_route)}\n\n## Puertas alcanzadas\n\n{md_table(pd.DataFrame([row]))}\n\nNo se infiere éxito de etapas no alcanzadas. Las reparaciones dirigidas conservan la identidad de la familia y su control data-only.\n")
    write(reports / "HYPERPARAMETER_CALIBRATION_REPORT.md", common_header(decision, "Calibración hiperparamétrica") + "\nLa criba y promoción multifidelidad conservan todos los ensayos, configuraciones, seeds, folds, stop reasons y fallos. S10/S11 no reutilizan objetivos externos para retunar el espacio de búsqueda.\n\n## Ranking auditado S9\n\n" + s9_table_md + "\n\n## Promoción OOF\n\n" + s10_table_md)
    write(reports / "FULL_OOF_METRICS_REPORT.md", common_header(decision, "Métricas OOF completas") + f"\nModo de confirmación: `{decision['diagnostic_evidence_mode']}`. Las métricas agregadas no sustituyen P90, peor caso, PSD, coherencia, fase, error modal, hotspot, BC y residuo físico.\n\n## Resumen numérico\n\n{oof_table_md}\n\n## Decisión de promoción\n\n{s10_table_md}\n")
    write(reports / "RESULTS_INTERPRETATION.md", common_header(decision, "Interpretación de resultados") + f"\nLa selección es no compensatoria: una mejora física no compensa degradación predictiva fuera de tolerancia, y una media favorable no compensa colas, semillas inestables o pérdida de utilidad del grafo. F01–F45 constituyen el atlas reproducible.\n\n## Evidencia emparejada\n\n{oof_table_md}\n\n## Claims admitidos\n\n{claim_table}\n")
    write(reports / "THESIS_EVIDENCE_MAP.md", common_header(decision, "Mapa de evidencia para tesis") + "\n" + claim_table)
    write(reports / "CLAIMS_AND_LIMITATIONS.md", common_header(decision, "Claims y limitaciones") + "\n" + claim_table + "\n\nNo están validados: estado completo de seis GDL, aceleraciones como salida final, sensores, daño, ruido, velocidad continua ni excitaciones fuera del dominio observado.\n")
    negative = gates[gates.final_selected != "PASS"]
    write(reports / "NEGATIVE_RESULTS.md", common_header(decision, "Resultados negativos") + "\nLos fallos y rutas no promovidas se conservan como evidencia. No se permite que una reparación cambie de familia ni que un caso aislado cierre una ruta antes de su presupuesto.\n\n" + md_table(negative))
    write(reports / "FINAL_DECISION.md", common_header(decision, "Decisión científica final") + f"\n## Estado\n\n`{decision['final_state']}`\n\n## Ganador\n\n{json.dumps(decision.get('winner'), indent=2, ensure_ascii=False)}\n\nUna confirmación externa genuina requiere un panel FEM nuevo diseñado después de congelar el modelo y ejecutado solo con autorización expresa.\n")

    write(STAGING / "README.md", common_header(decision, "Paquete final del portafolio PIGNO") + "\nContiene código, contratos, tablas, F01–F45, informes y manifiestos. Los binarios usan hardlinks cuando el sistema de archivos lo permite; cualquier puntero externo se identifica explícitamente en `manifests/BINARY_ARTIFACT_REGISTRY.csv`.\n")
    write(STAGING / "METHODOLOGY.md", common_header(decision, "Metodología reproducible") + "\nSecuencia: autoridad y ledger → fuentes → grafo/cargas/modos → capacidad → reparaciones dirigidas → panel factorial → búsqueda multifidelidad → nested grouped OOF → cinco semillas solo para finalistas → diagnósticos → decisión.\n")
    numbered_scripts = sorted(path.name for path in (ROOT / "scripts").glob("*.py") if path.name.split("_",1)[0].isdigit())
    def scripts_between(start: int, stop: int) -> list[str]:
        return [name for name in numbered_scripts if start <= int(name.split("_",1)[0]) <= stop]
    stage_commands = {
        "S0_S5_authority_design": ["01_audit_s1_s2_authority_graph_modal.py", "02_s5_fold_clean_primary_oracle.py", "03_build_68case_causal_input_vds.py", "04_close_incremental_base_mapping.py"],
        "S6_capacity_micropanel": scripts_between(5, 32),
        "S8_factorial": scripts_between(33, 36),
        "S9_multifidelity": scripts_between(37, 42),
        "R4_effective_pH_OpInf_repair_history": scripts_between(88, 95),
        "S10_nested_OOF": scripts_between(43, 58) + scripts_between(96, 97) + scripts_between(102, 102),
        "S11_five_seed": scripts_between(59, 64),
        "S12_diagnostics": scripts_between(65, 80) + scripts_between(83, 83) + scripts_between(98, 100),
        "S13_external_FEM_panel": "DESIGN_ONLY_REQUIRES_EXPLICIT_USER_AUTHORIZATION",
        "S14_decision_package_audit": scripts_between(81, 82) + scripts_between(84, 87) + scripts_between(101, 101),
        "execution_rule": "Resolve each numeric prefix against scripts/. Run only in a new work root with a new run_id; the frozen package is read-only evidence.",
    }
    write(STAGING / "scripts" / "STAGE_COMMANDS.json", json.dumps(stage_commands, indent=2, ensure_ascii=False))
    write(STAGING / "scripts" / "run_master_pipeline.ps1", "param([switch]$AuditOnly)\n$ErrorActionPreference = 'Stop'\nif (-not $AuditOnly) { throw 'La reejecución completa requiere un nuevo run_id y copia de trabajo; no sobrescriba la evidencia congelada.' }\n$python = 'C:\\Users\\yunim\\Documents\\BRIDGE\\pigno_dynamic_vscode_pipeline_v1_2\\.venv\\Scripts\\python.exe'\n& $python (Join-Path $PSScriptRoot '87_verify_frozen_final_package.py')\nif ($LASTEXITCODE -ne 0) { throw 'La verificación read-only del paquete falló.' }\n")
    write(STAGING / "legacy_links" / "HISTORICAL_ARTIFACTS.json", json.dumps({"negative_preflight_archive_preserved": True, "Rev7_scientific_authority": False, "Rev8_scientific_authority": False, "source_root": str(ROOT.parent)}, indent=2, ensure_ascii=False))

    required_figure_files = [STAGING / "figures" / f"F{number:02d}.{suffix}" for number in range(1, 46) for suffix in ("png", "pdf")]
    required_figure_files += [STAGING / "tables" / "figure_data" / f"F{number:02d}.csv" for number in range(1, 46)]
    required_figure_files += [STAGING / "figures" / "captions" / f"F{number:02d}.caption.json" for number in range(1, 46)]
    required_figure_files += [STAGING / "manifests" / "figures" / f"F{number:02d}.manifest.json" for number in range(1, 46)]
    missing_figure_files = [str(path.relative_to(STAGING)) for path in required_figure_files if not path.is_file() or path.stat().st_size == 0]
    arrowpoint_files = [str(path.relative_to(STAGING)) for path in STAGING.rglob("*") if path.is_file() and "arrowpoint" in path.name.lower()]
    expected_family_modules = set(family_sources)
    observed_family_modules = {path.name for path in (STAGING / "families").glob("R*.py")}
    family_module_count = len(observed_family_modules)
    source_module_count = len(list((STAGING / "src").rglob("*.py")))
    missing_reports = [name for name in REPORT_NAMES if not (reports / name).is_file()]
    structural_pass = not missing_figure_files and not arrowpoint_files and not missing_reports and observed_family_modules == expected_family_modules and source_module_count >= 10 and bool(binary_registry)
    package_qa = {"status": "PASS_FINAL_PACKAGE_STRUCTURAL_QA" if structural_pass else "FAIL_FINAL_PACKAGE_STRUCTURAL_QA", "required_figures": 45, "required_per_figure_artifacts": ["PNG", "PDF", "CSV", "caption JSON", "manifest JSON"], "missing_or_empty_figure_files": missing_figure_files, "ArrowPoint_files": arrowpoint_files, "mandatory_reports": len(REPORT_NAMES), "missing_reports": missing_reports, "family_module_count": family_module_count, "expected_family_modules": sorted(expected_family_modules), "observed_family_modules": sorted(observed_family_modules), "source_module_count": source_module_count, "binary_artifact_registry_rows": len(binary_registry)}
    write(STAGING / "manifests" / "PACKAGE_STRUCTURAL_QA.json", json.dumps(package_qa, indent=2, ensure_ascii=False))
    if package_qa["status"] != "PASS_FINAL_PACKAGE_STRUCTURAL_QA": raise RuntimeError(json.dumps(package_qa, ensure_ascii=False))

    artifacts = []
    for source in STAGING.rglob("*"):
        if source.is_file(): artifacts.append({"path": str(source.relative_to(STAGING)), "size_bytes": source.stat().st_size, "sha256": sha256(source)})
    write(STAGING / "manifests" / "ARTIFACT_MANIFEST.json", json.dumps({"status": "PASS_FINAL_PACKAGE_MANIFEST", "generated_utc": datetime.now(timezone.utc).isoformat(), "final_state": decision["final_state"], "artifact_count": len(artifacts), "artifacts": artifacts}, indent=2, ensure_ascii=False))
    missing_reports = [name for name in REPORT_NAMES if not (reports / name).is_file()]
    if missing_reports: raise RuntimeError(f"Mandatory reports missing: {missing_reports}")
    os.replace(STAGING, FINAL)
    print(json.dumps({"status": "PASS_FINAL_PORTFOLIO_PACKAGE", "path": str(FINAL), "final_state": decision["final_state"], "reports": len(REPORT_NAMES), "figures": 45, "binary_artifacts": len(binary_registry)}, indent=2))


if __name__ == "__main__":
    main()
