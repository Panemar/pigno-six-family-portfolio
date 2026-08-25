#!/usr/bin/env python3
"""Record the completed agent visual inspection of the current F01-F43 bundle."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
READINESS = S12 / "S12_PREDECISION_VISUAL_QA_READINESS_V1.json"
OUTPUT = S12 / "S12_PREDECISION_MANUAL_VISUAL_QA_V1.json"
IDS = [f"F{index:02d}" for index in range(1, 44)]
CHECKS = {
    "title_and_units_visible": True,
    "no_clipped_labels": True,
    "comparable_axes_consistent": True,
    "grayscale_or_noncolor_distinction": True,
    "geometry_semantics_correct": True,
    "caption_matches_quantity": True,
}
LIMITATIONS = {
    "F08": "Historical one-case capacity evidence is legible and explicitly excludes pre-repair R4 from forward decisions.",
    "F09": "Historical repairs are legible; slopes are descriptive and do not override frozen gates.",
    "F13": "Exploratory n=32 permutation associations and uncertainty bars are legible but are not causal importance.",
    "F16": "Recorded loss terms are legible; unavailable gradient diagnostics are declared rather than reconstructed.",
    "F23": "The one-point panels correctly disclose single-seed S10 negative evidence and do not imply seed variability.",
    "F31": "True-aspect geometry is legible after removing overlapping numeric ticks; exact coordinates remain in the source CSV.",
    "F34": "Phase gaps at low-energy frequencies remain visibly blank and are not imputed.",
    "F36": "Defined kinematic ratios are legible and valid/total counts expose four zero-energy cases per axis.",
    "F39": "Projected response-frequency diagnostics are legible but are not presented as independent eigenmode validation.",
    "F40": "Structural MAC and projected-response MAC are visually separated and retain their distinct meanings.",
    "F41": "COMAC is legible; the conservative zero convention remains identified as a diagnostic limitation.",
    "F42": "Frozen-checkpoint graph perturbations and relabel invariance are legible but do not claim causal superiority to retraining without a graph.",
    "F43": "Capacity/error comparison is legible but contains one frozen single-seed point and therefore no variance claim.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Predecision manual visual-QA record already exists")
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    if readiness.get("status") != "READY_S12_PREDECISION_VISUAL_QA_FOR_AGENT_INSPECTION":
        raise RuntimeError("Visual-QA readiness is not admitted")
    if readiness.get("scope") != IDS:
        raise RuntimeError("Visual-QA readiness scope is not exactly F01-F43")
    reviewed = {figure_id: sha256(S12 / "figures" / f"{figure_id}.png") for figure_id in IDS}
    findings = []
    for figure_id in IDS:
        limitation = LIMITATIONS.get(figure_id)
        findings.append({
            "figure_id": figure_id,
            "verdict": "PASS_WITH_DOCUMENTED_LIMITATION" if limitation else "PASS",
            "checks": dict(CHECKS),
            "finding": limitation or "Title, plotted quantity, units, axes, labels and visual encodings are legible and consistent with the caption.",
        })
    payload = {
        "status": "PASS_S12_PREDECISION_MANUAL_VISUAL_QA_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": IDS,
        "manual_visual_review_performed": True,
        "review_method": "agent_inspection_of_contact_sheets_and_original_pngs",
        "readiness_sha256": sha256(READINESS),
        "reviewed_png_sha256": reviewed,
        "findings": findings,
        "repaired_during_review": ["F31", "F36"],
        "failed_attempt_archives_preserved": [
            "failed_visual_qa_F31_F36_before_repair",
            "stale_predecision_readiness_before_F31_F36_visual_repair",
        ],
        "training_or_tuning_performed": False,
        "S14_authorized": True,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "figure_count": len(IDS)}, indent=2))


if __name__ == "__main__":
    main()
