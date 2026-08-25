"""Materialize losslessly compressed figure-source tables and verify SHA-256."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "external" / "COMPRESSED_DATA_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    compressed = ROOT / record["compressed"]
    target = ROOT / record["source"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(compressed, "rb") as source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    actual = sha256(target)
    if actual != record["source_sha256"]:
        target.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch: {actual}")
    print(f"PASS {target.relative_to(ROOT)} sha256={actual}")


if __name__ == "__main__":
    main()
