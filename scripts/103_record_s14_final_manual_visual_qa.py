#!/usr/bin/env python3
"""Record the completed agent visual inspection of final figures F44-F45."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
S14 = ROOT / "s14_final_decision"
READINESS = S14 / "S14_FINAL_DECISION_VISUAL_QA_READINESS_V1.json"
OUTPUT = S14 / "S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1.json"
IDS = ["F44", "F45"]
CHECKS = {
    "title_and_units_visible": True,
    "no_clipped_labels": True,
    "comparable_axes_consistent": True,
    "grayscale_or_noncolor_distinction": True,
    "geometry_semantics_correct": True,
    "caption_matches_quantity": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Final manual visual-QA record already exists")
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    if readiness.get("status") != "READY_S14_FINAL_DECISION_VISUAL_QA_FOR_AGENT_INSPECTION" or readiness.get("scope") != IDS:
        raise RuntimeError("Final visual-QA readiness is not admitted")
    findings = [
        {
            "figure_id": "F44",
            "verdict": "PASS_WITH_DOCUMENTED_LIMITATION",
            "checks": dict(CHECKS),
            "finding": "All six routes and reached/not-reached gates are legible; common modal reference and scoped projected evidence remain distinct from learned eigenmode validation.",
        },
        {
            "figure_id": "F45",
            "verdict": "PASS_WITH_DOCUMENTED_LIMITATION",
            "checks": dict(CHECKS),
            "finding": "PASS, FAIL, FAIL/LIMITED and NOT RUN/OUT OF SCOPE are visually distinct; R4 gains are retained while failed primary noninferiority and the absence of five-seed confirmation prevent acceptance.",
        },
    ]
    payload = {
        "status": "PASS_S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": IDS,
        "manual_visual_review_performed": True,
        "review_method": "agent_inspection_of_contact_sheets_and_original_pngs",
        "readiness_sha256": sha256(READINESS),
        "reviewed_png_sha256": {figure_id: sha256(S12 / "figures" / f"{figure_id}.png") for figure_id in IDS},
        "findings": findings,
        "repaired_during_review": ["F45"],
        "failed_attempt_archive_preserved": "failed_final_visual_qa_before_F45_status_repair",
        "training_or_tuning_performed": False,
        "final_package_authorized": True,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "figure_count": len(IDS)}, indent=2))


if __name__ == "__main__":
    main()
