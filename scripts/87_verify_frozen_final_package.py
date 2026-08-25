#!/usr/bin/env python3
"""Read-only verifier for the frozen thesis package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / "manifests" / "ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    if not MANIFEST.is_file():
        raise SystemExit("FAIL_FROZEN_PACKAGE_VERIFY: artifact manifest is absent")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_FINAL_PACKAGE_MANIFEST": failures.append("manifest status")
    for row in manifest.get("artifacts", []):
        path = PACKAGE / row["path"]
        if not path.is_file(): failures.append(f"missing:{row['path']}"); continue
        if path.stat().st_size != int(row["size_bytes"]): failures.append(f"size:{row['path']}"); continue
        if sha256(path) != row["sha256"]: failures.append(f"hash:{row['path']}")
    audit = PACKAGE / "manifests" / "FINAL_MASTER_COMPLETION_AUDIT.json"
    structural = PACKAGE / "manifests" / "PACKAGE_STRUCTURAL_QA.json"
    if not audit.is_file() or json.loads(audit.read_text(encoding="utf-8")).get("status") != "PASS_FINAL_MASTER_COMPLETION_AUDIT": failures.append("master completion audit")
    if not structural.is_file() or json.loads(structural.read_text(encoding="utf-8")).get("status") != "PASS_FINAL_PACKAGE_STRUCTURAL_QA": failures.append("structural QA")
    if len(list((PACKAGE / "families").glob("R*.py"))) != 6: failures.append("six family modules")
    if len(list((PACKAGE / "reports").glob("*.md"))) < 19: failures.append("nineteen reports")
    for number in range(1, 46):
        figure_id = f"F{number:02d}"
        for path in (PACKAGE / "figures" / f"{figure_id}.png", PACKAGE / "figures" / f"{figure_id}.pdf", PACKAGE / "tables" / "figure_data" / f"{figure_id}.csv", PACKAGE / "figures" / "captions" / f"{figure_id}.caption.json", PACKAGE / "manifests" / "figures" / f"{figure_id}.manifest.json"):
            if not path.is_file() or path.stat().st_size == 0: failures.append(f"figure artifact:{path.relative_to(PACKAGE)}")
    if any("arrowpoint" in path.name.lower() for path in PACKAGE.rglob("*") if path.is_file()): failures.append("ArrowPoint product")
    result = {"status": "PASS_FROZEN_FINAL_PACKAGE_VERIFY" if not failures else "FAIL_FROZEN_FINAL_PACKAGE_VERIFY", "manifest_artifacts": len(manifest.get("artifacts", [])), "failures": failures}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()
