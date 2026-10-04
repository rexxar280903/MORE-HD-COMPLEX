"""Clustering pair protocol (pseudocode §4.6; G1-03, G1-08).

Five TRAIN samples per class are drawn without replacement from a class-specific
seeded substream (nested-K, shared across feature methods and architectures);
all C(5K, 2) unordered unique pairs are used once, with the natural
same/different composition (no balancing, no reweighting).
"""

from __future__ import annotations

from math import comb

import numpy as np

from .constants import N_CLUSTER_PAIR_SAMPLES, PAIR_BALANCE_POLICY, PAIR_WEIGHTING
from .seeds import derive_subseed


def build_pairing(y_train: np.ndarray, train_mnist_index: np.ndarray, classes, seed: int) -> dict:
    classes = [int(c) for c in classes]
    selected_by_class = {}
    sel_positions = []
    sel_classes = []
    for c in classes:
        positions = np.flatnonzero(y_train == c)
        if positions.size < N_CLUSTER_PAIR_SAMPLES:
            raise ValueError(f"class {c} has fewer than {N_CLUSTER_PAIR_SAMPLES} training samples")
        class_seed = derive_subseed(seed, f"cluster_pairs:{c}")
        rng = np.random.default_rng(class_seed)
        chosen = np.sort(rng.choice(positions, size=N_CLUSTER_PAIR_SAMPLES, replace=False))
        selected_by_class[str(c)] = {
            "class_pair_seed": class_seed,
            "train_positions": [int(p) for p in chosen],
            "mnist_indices": [int(train_mnist_index[p]) for p in chosen],
        }
        sel_positions.extend(int(p) for p in chosen)
        sel_classes.extend([c] * N_CLUSTER_PAIR_SAMPLES)
    # 'selected' is sorted by (class, train_position) by construction
    sel_positions = np.asarray(sel_positions, dtype=np.int64)
    sel_classes = np.asarray(sel_classes, dtype=np.int64)
    n_sel = sel_positions.size
    ia, ib = np.triu_indices(n_sel, 1)                     # unordered unique pairs, a < b
    class_i = sel_classes[ia]
    class_j = sel_classes[ib]
    k = len(classes)
    n_same = int(np.sum(class_i == class_j))
    n_total = int(ia.size)
    n_diff = n_total - n_same
    keys = set(zip(sel_positions[ia].tolist(), sel_positions[ib].tolist()))
    n_dup = n_total - len(keys)
    n_self = int(np.sum(sel_positions[ia] == sel_positions[ib]))
    assert n_total == comb(N_CLUSTER_PAIR_SAMPLES * k, 2)
    assert n_same == k * comb(N_CLUSTER_PAIR_SAMPLES, 2)
    assert n_diff == comb(k, 2) * N_CLUSTER_PAIR_SAMPLES ** 2
    assert n_dup == 0 and n_self == 0
    manifest = {
        "master_seed": int(seed),
        "pair_seed": derive_subseed(seed, "cluster_pairs"),
        "n_samples_per_class": N_CLUSTER_PAIR_SAMPLES,
        "selected_by_class": selected_by_class,
        "pair_order": "unordered_i_lt_j",
        "sampling": "without_replacement",
        "sampler": "numpy.random.default_rng(DERIVE_SUBSEED(seed, 'cluster_pairs:<class>')).choice(replace=False), sorted",
        "frozen_before_optimizer": True,
        "balance_policy": PAIR_BALANCE_POLICY,
        "pair_weighting": PAIR_WEIGHTING,
    }
    stats = {
        "K": k,
        "n_selected_samples": int(n_sel),
        "n_pairs_total": n_total,
        "n_pairs_same_class": n_same,
        "n_pairs_different_class": n_diff,
        "same_class_ratio": n_same / n_total,
        "different_class_ratio": n_diff / n_total,
        "duplicate_pairs": n_dup,
        "self_pairs": n_self,
    }
    return {
        "manifest": manifest,
        "stats": stats,
        "selected_positions": sel_positions,      # (5K,) rows of X_train, sorted by (class, position)
        "selected_classes": sel_classes,
        "pair_a": ia,                             # indices into selected_positions
        "pair_b": ib,
        "pair_class_i": class_i,
        "pair_class_j": class_j,
    }


def cluster_positions_from_manifest(manifest: dict, classes) -> tuple[np.ndarray, np.ndarray]:
    """CLUSTER_SAMPLES_FROM_MANIFEST: (positions, classes) in (class, position) order."""
    pos, cls = [], []
    for c in classes:
        p = manifest["selected_by_class"][str(int(c))]["train_positions"]
        pos.extend(p)
        cls.extend([int(c)] * len(p))
    return np.asarray(pos, dtype=np.int64), np.asarray(cls, dtype=np.int64)
