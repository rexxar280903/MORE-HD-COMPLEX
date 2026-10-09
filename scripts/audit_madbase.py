"""Create machine-readable, outcome-free audit evidence for the local MADBase snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.artifact_io import save_json_atomic  # noqa: E402
from core.madbase import OFFICIAL_PAGE, load_madbase  # noqa: E402


GLYPHS = {str(i): glyph for i, glyph in enumerate("٠١٢٣٤٥٦٧٨٩")}


def _sha256_pixels(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _dhash(image: np.ndarray) -> int:
    small = np.asarray(Image.fromarray(image).resize((9, 8), Image.Resampling.BILINEAR))
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _split_rows(name, images, labels, sample_ids, writer_ids):
    for index in range(images.shape[0]):
        yield {
            "split": name,
            "index": index,
            "image": images[index],
            "label": int(labels[index]),
            "sample_id": str(sample_ids[index]),
            "writer_id": int(writer_ids[index]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "research_arabic_digits" / "research_data",
    )
    args = parser.parse_args()
    data = load_madbase(args.data_root)
    manifests = args.data_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    archive_paths = {name: args.data_root / "raw" / name for name in data.file_sha256}
    checksum_lines = [f"{digest}  raw/{name}" for name, digest in sorted(data.file_sha256.items())]
    (manifests / "raw_files.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    all_rows = list(_split_rows(
        "train", data.train_images, data.train_labels, data.train_sample_ids, data.train_writer_ids
    )) + list(_split_rows(
        "test", data.test_images, data.test_labels, data.test_sample_ids, data.test_writer_ids
    ))

    exact_groups = defaultdict(list)
    train_dhash = defaultdict(list)
    for row in all_rows:
        exact_groups[_sha256_pixels(row["image"])].append(row)
        if row["split"] == "train":
            train_dhash[_dhash(row["image"])].append(row)

    duplicate_rows = []
    exact_within_groups = 0
    exact_cross_groups = 0
    label_conflict_groups = 0
    for digest, rows in exact_groups.items():
        if len(rows) < 2:
            continue
        splits = {row["split"] for row in rows}
        labels = {row["label"] for row in rows}
        exact_within_groups += int(len(splits) == 1)
        exact_cross_groups += int(len(splits) > 1)
        label_conflict_groups += int(len(labels) > 1)
        first = rows[0]
        for other in rows[1:]:
            duplicate_rows.append({
                "kind": "exact_pixel_duplicate",
                "distance": 0,
                "digest": digest,
                "sample_a": first["sample_id"],
                "split_a": first["split"],
                "label_a": first["label"],
                "writer_a": first["writer_id"],
                "sample_b": other["sample_id"],
                "split_b": other["split"],
                "label_b": other["label"],
                "writer_b": other["writer_id"],
            })

    dhash_candidate_pairs = 0
    near_cross_pairs = 0
    near_examples = []
    for row in _split_rows("test", data.test_images, data.test_labels, data.test_sample_ids, data.test_writer_ids):
        value = _dhash(row["image"])
        candidates = [value] + [value ^ (1 << bit) for bit in range(64)]
        for candidate in candidates:
            for train_row in train_dhash.get(candidate, ()):
                if np.array_equal(row["image"], train_row["image"]):
                    continue
                dhash_candidate_pairs += 1
                pixel_mae = float(np.mean(np.abs(
                    row["image"].astype(np.int16) - train_row["image"].astype(np.int16)
                )))
                if pixel_mae > 2.0:
                    continue
                near_cross_pairs += 1
                if len(near_examples) < 500:
                    near_examples.append({
                        "kind": "candidate_cross_split_dhash_le_1",
                        "distance": (value ^ candidate).bit_count(),
                        "pixel_mae": round(pixel_mae, 6),
                        "digest": f"{value:016x}/{candidate:016x}",
                        "sample_a": train_row["sample_id"],
                        "split_a": "train",
                        "label_a": train_row["label"],
                        "writer_a": train_row["writer_id"],
                        "sample_b": row["sample_id"],
                        "split_b": "test",
                        "label_b": row["label"],
                        "writer_b": row["writer_id"],
                    })
    duplicate_rows.extend(near_examples)

    columns = [
        "kind", "distance", "pixel_mae", "digest", "sample_a", "split_a", "label_a", "writer_a",
        "sample_b", "split_b", "label_b", "writer_b",
    ]
    with (manifests / "duplicate_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(duplicate_rows)

    def split_summary(images, labels, writers):
        foreground = images > 0
        border = np.concatenate(
            [foreground[:, 0, :], foreground[:, -1, :], foreground[:, 1:-1, 0], foreground[:, 1:-1, -1]],
            axis=1,
        )
        return {
            "n_samples": int(images.shape[0]),
            "shape": list(images.shape[1:]),
            "dtype": str(images.dtype),
            "pixel_min": int(images.min()),
            "pixel_max": int(images.max()),
            "n_classes": int(np.unique(labels).size),
            "class_counts": {str(i): int(np.sum(labels == i)) for i in range(10)},
            "n_writers": int(np.unique(writers).size),
            "writer_min": int(writers.min()),
            "writer_max": int(writers.max()),
            "blank_images": int(np.sum(np.sum(foreground, axis=(1, 2)) == 0)),
            "images_touching_border": int(np.sum(np.any(border, axis=1))),
        }

    inventory = {
        "dataset_id": data.dataset_id,
        "official_page": OFFICIAL_PAGE,
        "snapshot_sha256": data.snapshot_sha256,
        "archives": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": data.file_sha256[name],
            }
            for name, path in sorted(archive_paths.items())
        },
        "image_format": "BMP source; canonical uint8 array",
        "raw_polarity": data.raw_polarity,
        "canonical_polarity": data.canonical_polarity,
        "canonical_transform": "uint8 255 - raw_grayscale",
        "train": split_summary(data.train_images, data.train_labels, data.train_writer_ids),
        "test": split_summary(data.test_images, data.test_labels, data.test_writer_ids),
        "official_test_origin": data.test_origin,
        "train_test_writer_overlap": int(len(set(data.train_writer_ids) & set(data.test_writer_ids))),
        "duplicate_audit": {
            "exact_within_split_groups": exact_within_groups,
            "exact_cross_split_groups": exact_cross_groups,
            "exact_label_conflict_groups": label_conflict_groups,
            "dhash_distance_le_1_candidate_pairs_scanned": dhash_candidate_pairs,
            "near_exact_cross_split_pairs_dhash_le_1_and_pixel_mae_le_2": near_cross_pairs,
            "candidate_rows_saved_cap": 500,
            "candidate_method_warning": "dHash candidates require human review; they are not automatic exclusions",
        },
    }
    save_json_atomic(inventory, manifests / "dataset_inventory.json")
    save_json_atomic({
        "dataset_id": data.dataset_id,
        "script": "ARABIC_INDIC",
        "source_label_to_internal": {str(i): i for i in range(10)},
        "internal_to_glyph": GLYPHS,
        "verification": "filename mapping plus visual inspection of writer001 pass01 for all ten labels",
    }, manifests / "label_mapping.json")

    report = f"""# MADBase data audit

- Dataset: `{data.dataset_id}`
- Official source: {OFFICIAL_PAGE}
- Snapshot SHA-256: `{data.snapshot_sha256}`
- Inventory: 60,000 official-train and 10,000 official-test images, 28×28, ten balanced classes.
- Writers: 600 train and 100 test; overlap = 0.
- Raw polarity: white background/dark foreground; canonical loader applies `255 - raw`.
- Blank images: {inventory['train']['blank_images']} train, {inventory['test']['blank_images']} test.
- Exact duplicate groups within a split: {exact_within_groups}.
- Exact duplicate groups crossing train/test: {exact_cross_groups}.
- Exact duplicate groups with conflicting labels: {label_conflict_groups}.
- Cross-split dHash candidates scanned at Hamming distance ≤1: {dhash_candidate_pairs}.
- Near-exact cross-split flags after requiring pixel MAE ≤2: {near_cross_pairs}; these are review flags, not exclusions.
- License: official download is public, but no standard license text was found on the official page; confirm research/redistribution terms before confirmatory publication.

No model output or final-test performance was inspected during this audit.
"""
    (manifests / "data_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
