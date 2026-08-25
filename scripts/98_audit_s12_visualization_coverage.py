#!/usr/bin/env python3
"""Audit the frozen F01-F45 visualization contract without generating figures.

This is a design/readiness audit.  It deliberately does not train, tune, read
future results as selection feedback, or authorize a scientific decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
CONTRACT = S12 / "S12_VISUALIZATION_CONTRACT_V1.json"
OUT_JSON = S12 / "S12_VISUALIZATION_COVERAGE_AUDIT_V1.json"
OUT_CSV = S12 / "S12_VISUALIZATION_COVERAGE_AUDIT_V1.csv"
OUT_MD = ROOT / "reports" / "S12_VISUALIZATION_COVERAGE_AUDIT_V1.md"

GENERATOR_BY_ID = {
    **{f"F{i:02d}": "69_generate_s12_authority_graph_figures.py" for i in range(1, 7)},
    **{f"F{i:02d}": "71_generate_s12_historical_experiment_figures.py" for i in range(7, 17)},
    **{f"F{i:02d}": "65_generate_s12_core_oof_figures.py" for i in (17, 18, 20, 21, 23, 37, 38, 43)},
    **{f"F{i:02d}": "73_generate_s12_paired_oof_field_figures.py" for i in (19, 22, 24, 25, 26, 27, 28, 29)},
    **{f"F{i:02d}": "67_generate_s12_dynamic_spatial_figures.py" for i in range(30, 37)},
    **{f"F{i:02d}": "75_generate_s12_modal_diagnostics.py" for i in range(39, 42)},
    "F42": "78_generate_s12_graph_utility_figure.py",
    "F44": "82_generate_s14_final_decision_figures.py",
    "F45": "82_generate_s14_final_decision_figures.py",
}

OUTPUTS = {
    "png": ("figures", ".png"),
    "pdf": ("figures", ".pdf"),
    "source": ("figure_data", ".csv"),
    "caption": ("captions", ".caption.json"),
    "manifest": ("figure_manifests", ".manifest.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def output_path(figure_id: str, kind: str) -> Path:
    folder, suffix = OUTPUTS[kind]
    return S12 / folder / f"{figure_id}{suffix}"


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8-sig"))
    contract_ids = [entry["id"] for entry in contract["figures"]]
    expected_ids = [f"F{i:02d}" for i in range(1, 46)]
    archive = S12 / "rejected_historical_visuals_pre_r4_opinf_repair"
    rows: list[dict[str, object]] = []

    for figure_id in expected_ids:
        generator_name = GENERATOR_BY_ID.get(figure_id)
        generator = ROOT / "scripts" / generator_name if generator_name else None
        present = {kind: output_path(figure_id, kind).exists() for kind in OUTPUTS}
        complete = all(present.values())
        archived = bool(archive.exists() and list(archive.rglob(f"{figure_id}*")))
        if complete and figure_id in {f"F{i:02d}" for i in range(1, 7)}:
            state = "CURRENT_COMPLETE_AUTHORITY"
        elif complete:
            state = "CURRENT_COMPLETE_REQUIRES_FINAL_SOURCE_AUDIT"
        elif archived:
            state = "ARCHIVED_INVALID_PRE_R4_REPAIR_REGENERATE_AFTER_GATE"
        elif figure_id in {"F44", "F45"}:
            state = "PENDING_S14_FINAL_DECISION"
        else:
            state = "PENDING_UPSTREAM_GATE"
        rows.append(
            {
                "figure_id": figure_id,
                "in_contract": figure_id in contract_ids,
                "generator": generator_name or "",
                "generator_exists": bool(generator and generator.exists()),
                "generator_sha256": sha256(generator) if generator and generator.exists() else "",
                **{f"has_{kind}": value for kind, value in present.items()},
                "complete_current_bundle": complete,
                "archived_pre_r4_artifact_present": archived,
                "state": state,
            }
        )

    missing_contract = sorted(set(expected_ids) - set(contract_ids))
    extra_contract = sorted(set(contract_ids) - set(expected_ids))
    unmapped = [row["figure_id"] for row in rows if not row["generator_exists"]]
    current_complete = [row["figure_id"] for row in rows if row["complete_current_bundle"]]
    archived_invalid = [row["figure_id"] for row in rows if row["archived_pre_r4_artifact_present"]]
    design_complete = not missing_contract and not extra_contract and not unmapped
    status = (
        "PASS_S12_VISUALIZATION_DESIGN_COVERAGE_PENDING_UPSTREAM_GATES"
        if design_complete
        else "FAIL_S12_VISUALIZATION_DESIGN_COVERAGE"
    )
    payload = {
        "schema": "S12_VISUALIZATION_COVERAGE_AUDIT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(CONTRACT),
        "contract_sha256": sha256(CONTRACT),
        "expected_figure_count": 45,
        "mapped_generator_count": sum(bool(row["generator_exists"]) for row in rows),
        "current_complete_count": len(current_complete),
        "current_complete_ids": current_complete,
        "archived_invalid_ids": archived_invalid,
        "missing_contract_ids": missing_contract,
        "extra_contract_ids": extra_contract,
        "unmapped_or_missing_generator_ids": unmapped,
        "execution_gate": contract["execution_gate"],
        "claim_boundary": (
            "A mapped generator is design readiness only. F07-F45 remain non-current until "
            "their upstream repaired-R4 OOF or final-decision gates and source audits pass."
        ),
        "training_or_tuning_performed": False,
        "S11_authorized": False,
        "S14_authorized": False,
        "rows": rows,
    }

    atomic_text(OUT_JSON, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = OUT_CSV.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(OUT_CSV)

    markdown = [
        "# S12 visualization coverage audit",
        "",
        f"- Status: `{status}`",
        f"- Frozen contract figures: {len(contract_ids)}/45",
        f"- Figures with an existing mapped generator: {payload['mapped_generator_count']}/45",
        f"- Current complete bundles: {len(current_complete)}/45 ({', '.join(current_complete) or 'none'})",
        f"- Archived pre-repair figures requiring regeneration: {', '.join(archived_invalid) or 'none'}",
        "- No training, tuning, promotion, or final decision was performed.",
        "",
        "## Claim boundary",
        "",
        payload["claim_boundary"],
        "",
        "## Coverage",
        "",
        "| Figure | Generator | Current bundle | Archived invalid | State |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['figure_id']} | `{row['generator']}` | "
            f"{'yes' if row['complete_current_bundle'] else 'no'} | "
            f"{'yes' if row['archived_pre_r4_artifact_present'] else 'no'} | `{row['state']}` |"
        )
    atomic_text(OUT_MD, "\n".join(markdown) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "mapped_generator_count", "current_complete_count",
        "current_complete_ids", "archived_invalid_ids"
    )}, indent=2))


if __name__ == "__main__":
    main()
