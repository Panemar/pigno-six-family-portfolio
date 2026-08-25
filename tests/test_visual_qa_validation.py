from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("visual_qa_validation", ROOT / "scripts" / "visual_qa_validation.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_manual_visual_qa_requires_exact_hashes_and_findings(tmp_path: Path) -> None:
    figures = tmp_path / "figures"; figures.mkdir()
    ids = ["F01", "F02"]
    for figure_id in ids:
        (figures / f"{figure_id}.png").write_bytes((figure_id + "-pixels").encode())
    readiness = tmp_path / "readiness.json"; readiness.write_text(json.dumps({"status": "READY"}), encoding="utf-8")
    checks = {
        "title_and_units_visible": True,
        "no_clipped_labels": True,
        "comparable_axes_consistent": True,
        "grayscale_or_noncolor_distinction": True,
        "geometry_semantics_correct": True,
        "caption_matches_quantity": True,
    }
    report = tmp_path / "manual.json"
    payload = {
        "status": "PASS_TEST_VISUAL_QA",
        "scope": ids,
        "manual_visual_review_performed": True,
        "review_method": "agent_inspection_of_contact_sheets_and_original_pngs",
        "readiness_sha256": MODULE.sha256(readiness),
        "reviewed_png_sha256": {figure_id: MODULE.sha256(figures / f"{figure_id}.png") for figure_id in ids},
        "findings": [{"figure_id": figure_id, "verdict": "PASS", "checks": checks, "finding": "Readable and semantically consistent."} for figure_id in ids],
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert MODULE.validate_manual_visual_qa(report, readiness, "PASS_TEST_VISUAL_QA", ids, figures)["scope"] == ids
    (figures / "F02.png").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Reviewed PNG hash mismatch"):
        MODULE.validate_manual_visual_qa(report, readiness, "PASS_TEST_VISUAL_QA", ids, figures)
