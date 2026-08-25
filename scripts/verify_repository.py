"""Verify repository integrity, size gates, and external-data contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
MAX_GIT_FILE_BYTES = 100 * 1024 * 1024


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    failures: list[str] = []
    if not MANIFEST.exists():
        failures.append("missing MANIFEST.sha256")
    else:
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            target = ROOT / Path(relative)
            if not target.is_file():
                failures.append(f"missing: {relative}")
            elif digest(target) != expected:
                failures.append(f"hash mismatch: {relative}")

    oversized = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.stat().st_size >= MAX_GIT_FILE_BYTES
    ]
    failures.extend(f"GitHub size gate: {path}" for path in oversized)

    external = ROOT / "data" / "external" / "EXTERNAL_BINARY_MANIFEST.csv"
    if not external.exists():
        failures.append("missing external binary manifest")
    else:
        with external.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        required = {"artifact_id", "logical_path", "sha256", "size_bytes", "scientific_role"}
        if not rows:
            failures.append("external binary manifest is empty")
        elif not required.issubset(rows[0]):
            failures.append("external binary manifest schema mismatch")

    compressed_manifest = ROOT / "data" / "external" / "COMPRESSED_DATA_MANIFEST.json"
    if compressed_manifest.exists():
        record = json.loads(compressed_manifest.read_text(encoding="utf-8-sig"))
        packed = ROOT / record["compressed"]
        if not packed.exists() or digest(packed) != record["compressed_sha256"]:
            failures.append("compressed F27 artifact mismatch")

    if failures:
        print("FAIL_REPOSITORY_VERIFICATION")
        print("\n".join(f"- {item}" for item in failures))
        sys.exit(1)
    print("PASS_REPOSITORY_VERIFICATION")


if __name__ == "__main__":
    main()
