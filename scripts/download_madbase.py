"""Download, checksum, extract, and validate the pinned official MADBase snapshot."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.madbase import DOWNLOADS, EXPECTED_SHA256, load_madbase  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "research_arabic_digits" / "research_data",
    )
    args = parser.parse_args()
    raw = args.data_root / "raw"
    extracted = args.data_root / "extracted"
    raw.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    for name, url in DOWNLOADS.items():
        target = raw / name
        expected = EXPECTED_SHA256[name]
        if not target.is_file() or sha256(target) != expected:
            part = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, part)
            actual = sha256(part)
            if actual != expected:
                part.unlink(missing_ok=True)
                raise IOError(f"{name}: SHA-256 {actual} != pinned {expected}")
            os.replace(part, target)

    tar = shutil.which("tar")
    if tar is None:
        raise RuntimeError("RAR extractor not found; install 7-Zip/libarchive or provide tar on PATH")
    for name in DOWNLOADS:
        subprocess.run([tar, "-xf", str(raw / name), "-C", str(extracted)], check=True)

    dataset = load_madbase(args.data_root)
    print(f"MADBase ready: {dataset.dataset_id} snapshot={dataset.snapshot_sha256}")


if __name__ == "__main__":
    main()
