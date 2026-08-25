"""Shared validation for explicit agent visual-QA records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manual_visual_qa(
    report_path: Path,
    readiness_path: Path,
    expected_status: str,
    expected_ids: list[str],
    figure_directory: Path,
) -> dict:
    if not report_path.is_file() or not readiness_path.is_file():
        raise RuntimeError(f"Visual-QA report/readiness absent: {report_path}; {readiness_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if report.get("status") != expected_status:
        raise RuntimeError(f"Unadmitted visual-QA status: {report.get('status')}")
    if report.get("scope") != expected_ids or report.get("manual_visual_review_performed") is not True:
        raise RuntimeError("Visual-QA scope or explicit-review flag differs from contract")
    if report.get("review_method") != "agent_inspection_of_contact_sheets_and_original_pngs":
        raise RuntimeError("Visual-QA review method is not the frozen agent-inspection method")
    if report.get("readiness_sha256") != sha256(readiness_path):
        raise RuntimeError("Visual-QA readiness hash mismatch")
    reviewed = report.get("reviewed_png_sha256") or {}
    findings = report.get("findings") or []
    if set(reviewed) != set(expected_ids) or [row.get("figure_id") for row in findings] != expected_ids:
        raise RuntimeError("Visual-QA figure coverage is not exact and ordered")
    for figure_id in expected_ids:
        png = figure_directory / f"{figure_id}.png"
        if not png.is_file() or reviewed.get(figure_id) != sha256(png):
            raise RuntimeError(f"Reviewed PNG hash mismatch: {figure_id}")
    allowed_verdicts = {"PASS", "PASS_WITH_DOCUMENTED_LIMITATION"}
    required_checks = {
        "title_and_units_visible",
        "no_clipped_labels",
        "comparable_axes_consistent",
        "grayscale_or_noncolor_distinction",
        "geometry_semantics_correct",
        "caption_matches_quantity",
    }
    for row in findings:
        if row.get("verdict") not in allowed_verdicts:
            raise RuntimeError(f"Visual-QA verdict failed for {row.get('figure_id')}")
        checks = row.get("checks") or {}
        if set(checks) != required_checks or not all(checks.values()):
            raise RuntimeError(f"Visual-QA checks incomplete for {row.get('figure_id')}")
        if not str(row.get("finding", "")).strip():
            raise RuntimeError(f"Visual-QA finding absent for {row.get('figure_id')}")
    return report
