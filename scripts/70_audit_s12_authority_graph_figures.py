#!/usr/bin/env python3
"""Audit F01-F06 after explicit visual QA and preserve rejected attempts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
EXPECTED_GENERATOR = (ROOT / "scripts" / "69_generate_s12_authority_graph_figures.py").resolve()
FIGURE_IDS = [f"F{index:02d}" for index in range(1, 7)]
VISUAL_FINDINGS = {
    "F01": "PASS: stage sequence and authorization boundary are legible",
    "F02": "PASS: six-route mechanism map is legible and is not presented as a ranking",
    "F03": "PASS: true-aspect Beam graph, observations and restrained nodes are legible",
    "F04": "PASS: local frames are legible in three true-aspect orthographic projections",
    "F05": "PASS: causal axle trajectories and active tracks are legible",
    "F06": "PASS: base, incremental and total fields share one symmetric scale and use true-aspect plan/elevation projections",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    failures = []
    for figure_id in FIGURE_IDS:
        manifest_path = S12 / "figure_manifests" / f"{figure_id}.manifest.json"
        caption_path = S12 / "captions" / f"{figure_id}.caption.json"
        png_path = S12 / "figures" / f"{figure_id}.png"
        pdf_path = S12 / "figures" / f"{figure_id}.pdf"
        csv_path = S12 / "figure_data" / f"{figure_id}.csv"
        required = [manifest_path, caption_path, png_path, pdf_path, csv_path]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            failures.append({"figure_id": figure_id, "missing_or_empty": missing})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = {
            "generator_exact": Path(manifest["script"]).resolve() == EXPECTED_GENERATOR,
            "generator_hash_exact": manifest["script_sha256"] == sha256(EXPECTED_GENERATOR),
            "png_hash_exact": manifest["png_sha256"] == sha256(png_path),
            "pdf_hash_exact": manifest["pdf_sha256"] == sha256(pdf_path),
            "csv_hash_exact": manifest["source_csv_sha256"] == sha256(csv_path),
            "caption_id_exact": json.loads(caption_path.read_text(encoding="utf-8"))["figure_id"] == figure_id,
        }
        if not all(checks.values()):
            failures.append({"figure_id": figure_id, "failed_checks": [key for key, value in checks.items() if not value]})
        rows.append({"figure_id": figure_id, "checks": checks, "visual_finding": VISUAL_FINDINGS[figure_id]})
    arrowpoint = [str(path) for path in S12.rglob("*") if path.is_file() and "arrowpoint" in path.name.lower()]
    if arrowpoint:
        failures.append({"forbidden_arrowpoint_files": arrowpoint})
    rejected = sorted(path.name for path in S12.glob("rejected_visual_qa_v*") if path.is_dir())
    report = {
        "status": "PASS_S12_AUTHORITY_GRAPH_VISUAL_QA_V1" if not failures else "FAIL_S12_AUTHORITY_GRAPH_VISUAL_QA_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": FIGURE_IDS,
        "manual_visual_review_recorded_after_original_resolution_inspection": True,
        "rows": rows,
        "rejected_attempt_archives": rejected,
        "forbidden_arrowpoint_count": len(arrowpoint),
        "failures": failures,
        "training_or_tuning_performed": False,
        "S12_final_decision_authorized": False,
    }
    output = S12 / "S12_AUTHORITY_GRAPH_VISUAL_QA_V1.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
