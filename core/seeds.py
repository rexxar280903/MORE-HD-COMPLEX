"""Deterministic sub-seed propagation (pseudocode §0.3, G0-04)."""

from __future__ import annotations

import hashlib

import numpy as np

from .constants import CONFIRMATORY_SEEDS, PILOT_SEED


def derive_subseed(master_seed: int, namespace: str) -> int:
    """DERIVE_SUBSEED: uint32 from the first 8 hex digits of SHA256("<seed>:<namespace>")."""
    digest = hashlib.sha256(f"{int(master_seed)}:{namespace}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def rng_for(master_seed: int, namespace: str) -> np.random.Generator:
    """RNG(DERIVE_SUBSEED(seed, namespace)) as numpy.random.default_rng (PCG64)."""
    return np.random.default_rng(derive_subseed(master_seed, namespace))


def validate_seed_protocol(seed: int, run_mode: str) -> None:
    """VALIDATE_SEED_PROTOCOL (G0-04)."""
    if run_mode == "PILOT":
        if seed != PILOT_SEED:
            raise ValueError(f"PILOT runs must use seed {PILOT_SEED}, got {seed}")
    elif run_mode == "CONFIRMATORY":
        if seed not in CONFIRMATORY_SEEDS:
            raise ValueError(f"CONFIRMATORY seed must be one of {CONFIRMATORY_SEEDS}, got {seed}")
    else:
        raise ValueError(f"unknown run_mode {run_mode!r}")


def manifest_seeds(seed: int) -> dict:
    """Seed fields recorded in every run manifest (workbook 02_Run_Config)."""
    return {
        "master_seed": int(seed),
        "data_seed": derive_subseed(seed, "data"),
        "ry_core_seed": derive_subseed(seed, "init:ry_core"),
        "rz_phase_seed": derive_subseed(seed, "init:rz_phase"),
        "ry_extra_seed": derive_subseed(seed, "ablation:init:ry_extra"),
        "more_init_seed": derive_subseed(seed, "init:more_qcnn"),
        "pair_seed": derive_subseed(seed, "cluster_pairs"),
        "label_seed": derive_subseed(seed, "quantum_labels"),
    }
