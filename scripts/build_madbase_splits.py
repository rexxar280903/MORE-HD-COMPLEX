#!/usr/bin/env python3
"""Build and verify all frozen writer-aware MADBase split manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import constants as C  # noqa: E402
from core.artifact_io import save_json_atomic, sha256_file, sha256_json  # noqa: E402
from core.madbase import load_madbase  # noqa: E402
from core.splits import (  # noqa: E402
    build_madbase_split_manifest,
    madbase_split_manifest_path,
    validate_madbase_split_manifest,
)


def main() -> None:
    data = load_madbase(ROOT / C.MADBASE_ROOT)
    split_dir = ROOT / C.MADBASE_SPLITS_DIR
    jobs = [("PILOT", C.PILOT_SEED)] + [("CONFIRMATORY", s) for s in C.CONFIRMATORY_SEEDS]
    registry = {
        "schema_version": "1.0",
        "dataset_id": data.dataset_id,
        "dataset_snapshot_sha256": data.snapshot_sha256,
        "policy": "writer-disjoint train/validation; pilot holdout from train; fixed official-test subset",
        "manifests": [],
    }
    confirmatory_test_hashes = set()
    for mode, seed in jobs:
        manifest = build_madbase_split_manifest(seed, data, mode)
        validate_madbase_split_manifest(manifest, seed, data, mode)
        path = madbase_split_manifest_path(seed, mode, split_dir)
        save_json_atomic(manifest, path)
        test_hash = sha256_json(manifest["test_sample_ids_by_class"])
        if mode == "CONFIRMATORY":
            confirmatory_test_hashes.add(test_hash)
        registry["manifests"].append({
            "run_mode": mode,
            "seed": seed,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "test_sample_ids_sha256": test_hash,
            "test_origin": manifest["test_origin"],
        })
        print("ok", path.relative_to(ROOT))
    if len(confirmatory_test_hashes) != 1:
        raise RuntimeError("confirmatory manifests do not share one fixed final-test subset")
    registry["confirmatory_test_sample_ids_sha256"] = next(iter(confirmatory_test_hashes))
    save_json_atomic(registry, split_dir / "split_registry.json")
    print("ok", (split_dir / "split_registry.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
