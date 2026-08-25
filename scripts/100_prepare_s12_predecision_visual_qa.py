#!/usr/bin/env python3
"""Create F01-F43 contact sheets and structural readiness evidence for visual QA."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
SHEETS = S12 / "visual_qa_contact_sheets"
OUT = S12 / "S12_PREDECISION_VISUAL_QA_READINESS_V1.json"
IDS = [f"F{index:02d}" for index in range(1, 44)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Visual-QA readiness artifacts already exist")
    if SHEETS.exists():
        sheet_entries = list(SHEETS.iterdir())
        if any(entry.name.lower() != "desktop.ini" or not entry.is_file() for entry in sheet_entries):
            raise FileExistsError("Visual-QA readiness artifacts already exist")
        for entry in sheet_entries:
            entry.unlink()
        SHEETS.rmdir()
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    images: list[tuple[str, Image.Image]] = []
    for figure_id in IDS:
        png = S12 / "figures" / f"{figure_id}.png"
        pdf = S12 / "figures" / f"{figure_id}.pdf"
        csv = S12 / "figure_data" / f"{figure_id}.csv"
        caption = S12 / "captions" / f"{figure_id}.caption.json"
        manifest = S12 / "figure_manifests" / f"{figure_id}.manifest.json"
        required = [png, pdf, csv, caption, manifest]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            failures.append({"figure_id": figure_id, "missing_or_empty": missing})
            continue
        try:
            with Image.open(png) as source:
                source.verify()
            with Image.open(png) as source:
                width, height = source.size
                image = source.convert("RGB")
        except Exception as exc:  # diagnostic boundary
            failures.append({"figure_id": figure_id, "png_open_error": repr(exc)})
            continue
        if width < 1200 or height < 700:
            failures.append({"figure_id": figure_id, "insufficient_png_dimensions": [width, height]})
        records.append({
            "figure_id": figure_id,
            "width_px": width,
            "height_px": height,
            "png_sha256": sha256(png),
            "pdf_sha256": sha256(pdf),
            "source_sha256": sha256(csv),
            "caption_sha256": sha256(caption),
            "manifest_sha256": sha256(manifest),
        })
        images.append((figure_id, image))
    if failures:
        payload = {
            "status": "FAIL_S12_PREDECISION_VISUAL_QA_READINESS_V1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "failures": failures,
            "manual_visual_review_performed": False,
            "S14_authorized": False,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    SHEETS.mkdir(parents=True)
    font = ImageFont.load_default()
    sheet_paths: list[str] = []
    for sheet_number, start in enumerate(range(0, len(images), 9), 1):
        canvas = Image.new("RGB", (1800, 1260), "white")
        draw = ImageDraw.Draw(canvas)
        for local_index, (figure_id, image) in enumerate(images[start:start + 9]):
            row, column = divmod(local_index, 3)
            tile = image.copy()
            tile.thumbnail((560, 360))
            x = 20 + column * 590 + (560 - tile.width) // 2
            y = 30 + row * 410 + (360 - tile.height) // 2
            canvas.paste(tile, (x, y))
            draw.rectangle((20 + column * 590, 20 + row * 410, 580 + column * 590, 400 + row * 410), outline="#777777", width=2)
            draw.text((30 + column * 590, 382 + row * 410), figure_id, fill="black", font=font)
        path = SHEETS / f"S12_PREDECISION_QA_SHEET_{sheet_number:02d}.png"
        canvas.save(path, dpi=(150, 150))
        sheet_paths.append(str(path))
    payload = {
        "status": "READY_S12_PREDECISION_VISUAL_QA_FOR_AGENT_INSPECTION",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": IDS,
        "figure_count": len(records),
        "records": records,
        "contact_sheets": sheet_paths,
        "contact_sheet_sha256": {path: sha256(Path(path)) for path in sheet_paths},
        "manual_visual_review_performed": False,
        "required_manual_checks": [
            "titles and units visible",
            "no clipped labels",
            "comparable panels use consistent axes",
            "grayscale distinction remains interpretable",
            "true-aspect 3D geometry is not silently compressed",
            "caption and plotted quantity agree",
        ],
        "training_or_tuning_performed": False,
        "S14_authorized": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "figure_count": len(records), "contact_sheets": sheet_paths}, indent=2))


if __name__ == "__main__":
    main()
