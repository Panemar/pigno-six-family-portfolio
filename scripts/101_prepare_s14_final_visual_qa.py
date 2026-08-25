#!/usr/bin/env python3
"""Prepare F44-F45 for explicit final visual inspection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
S14 = ROOT / "s14_final_decision"
OUT = S14 / "S14_FINAL_DECISION_VISUAL_QA_READINESS_V1.json"
SHEET = S14 / "S14_FINAL_DECISION_QA_SHEET.png"
IDS = ["F44", "F45"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() or SHEET.exists():
        raise FileExistsError("Final visual-QA readiness artifacts already exist")
    failures: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    images: list[tuple[str, Image.Image]] = []
    for figure_id in IDS:
        paths = {
            "png": S12 / "figures" / f"{figure_id}.png",
            "pdf": S12 / "figures" / f"{figure_id}.pdf",
            "csv": S12 / "figure_data" / f"{figure_id}.csv",
            "caption": S12 / "captions" / f"{figure_id}.caption.json",
            "manifest": S12 / "figure_manifests" / f"{figure_id}.manifest.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
        if missing:
            failures.append({"figure_id": figure_id, "missing_or_empty": missing})
            continue
        try:
            with Image.open(paths["png"]) as source:
                source.verify()
            with Image.open(paths["png"]) as source:
                width, height = source.size
                image = source.convert("RGB")
        except Exception as exc:
            failures.append({"figure_id": figure_id, "png_open_error": repr(exc)})
            continue
        if width < 1200 or height < 700:
            failures.append({"figure_id": figure_id, "insufficient_png_dimensions": [width, height]})
        records.append({"figure_id": figure_id, "width_px": width, "height_px": height, **{f"{key}_sha256": sha256(path) for key, path in paths.items()}})
        images.append((figure_id, image))
    if not failures:
        canvas = Image.new("RGB", (1800, 720), "white")
        draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
        for index, (figure_id, image) in enumerate(images):
            tile = image.copy(); tile.thumbnail((840, 620)); x = 30 + index * 880 + (840 - tile.width) // 2; y = 20 + (620 - tile.height) // 2
            canvas.paste(tile, (x, y)); draw.rectangle((30 + index * 880, 20, 870 + index * 880, 650), outline="#777777", width=2); draw.text((40 + index * 880, 670), figure_id, fill="black", font=font)
        canvas.save(SHEET, dpi=(150, 150))
    payload = {
        "status": "READY_S14_FINAL_DECISION_VISUAL_QA_FOR_AGENT_INSPECTION" if not failures else "FAIL_S14_FINAL_DECISION_VISUAL_QA_READINESS_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": IDS,
        "records": records,
        "contact_sheet": str(SHEET) if not failures else None,
        "contact_sheet_sha256": sha256(SHEET) if not failures else None,
        "failures": failures,
        "manual_visual_review_performed": False,
        "final_package_authorized": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
