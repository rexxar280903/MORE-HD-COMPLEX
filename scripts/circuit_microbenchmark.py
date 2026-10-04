#!/usr/bin/env python3
"""Controlled circuit-cost microbenchmark A/B/C/D/M (pseudocode §1.2.1, SAP §4.5).

Adapted to the batched engine (decision 2026-10-04): the production cost of one
objective evaluation is one U(theta) construction plus one forward pass over the
evaluated samples. One *repeat* is therefore one forward pass of a fixed batch of
``--batch`` samples (default 1000); per-sample times are pass time / batch.
Three blocks run in separate processes with rotated model order; one process,
one thread, no other runs active (n_parallel_declared = 1).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import constants as C  # noqa: E402
from core.artifact_io import save_json_atomic  # noqa: E402
from core.circuits import BatchedStatevector, build_initialization_bundle, build_model, initial_params  # noqa: E402
from core.environment import package_versions  # noqa: E402
from core.timing import cpu_model, now_iso, thread_env  # noqa: E402

ORDERS = {1: ["A", "B", "C", "D", "M"], 2: ["B", "C", "D", "M", "A"], 3: ["C", "D", "M", "A", "B"]}
CODE_TO_ARCH = {v: k for k, v in C.MODEL_CODE.items()}


def bench_inputs(batch: int) -> np.ndarray:
    """Fixed inputs: the seed-42 K=10 PCA X_train_scaled of the timing pilot when present."""
    for cand in sorted((ROOT / "runs").glob("cls-0-1-2-3-4-5-6-7-8-9_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed42*/artifacts/X_train_scaled.npy")):
        x = np.load(cand)
        return x[np.linspace(0, x.shape[0] - 1, batch).astype(int)]
    rng = np.random.default_rng(42)
    return rng.uniform(0, np.pi, (batch, 8))


def run_block(block: int, batch: int, repeats: int, warmup: int, out_dir: Path) -> None:
    x = bench_inputs(batch)
    init = build_initialization_bundle(42)
    rows = []
    for code in ORDERS[block]:
        arch = CODE_TO_ARCH[code]
        model = build_model(arch, init)
        eng = BatchedStatevector(model)
        theta = initial_params(arch, init)
        phi = eng.encode(x)
        for _ in range(warmup):
            eng.expectations(phi, eng.unitary_columns(theta))
        wall, cpu, uni = [], [], []
        for _ in range(repeats):
            t0, c0 = time.perf_counter(), time.process_time()
            u = eng.unitary_columns(theta)
            t1 = time.perf_counter()
            eng.expectations(phi, u)
            wall.append((time.perf_counter() - t0) / batch)
            cpu.append((time.process_time() - c0) / batch)
            uni.append(t1 - t0)
        res = model.resource_summary()
        rows.append({
            "block_id": block, "model": arch, "model_code": code, "n_params": model.n_trainable,
            "gate_count": res["gate_count_total"], "depth": res["depth"], "n_warmup": warmup, "n_repeat": repeats,
            "batch_size": batch, "wall_median_sec": float(np.median(wall)), "wall_q1_sec": float(np.percentile(wall, 25)),
            "wall_q3_sec": float(np.percentile(wall, 75)), "wall_p95_sec": float(np.percentile(wall, 95)),
            "cpu_median_sec": float(np.median(cpu)), "unitary_median_sec": float(np.median(uni)),
        })
    v = package_versions()
    save_json_atomic({
        "rows": rows, "per_sample_basis": f"forward pass of {batch} samples incl. one U(theta) build, divided by {batch}",
        "device_name": C.DEVICE_NAME, "sim_dtype": C.SIM_DTYPE, "pennylane_version": v.get("pennylane"),
        "numpy_version": v.get("numpy"), "hostname": os.uname().nodename, "cpu_model": cpu_model(),
        "thread_env": thread_env(), "timestamp": now_iso(),
    }, out_dir / f"circuit_microbenchmark_block{block}.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", type=int, choices=[1, 2, 3])
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out-dir", default=str(ROOT / "benchmarks"))
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.block:
        run_block(a.block, a.batch, a.repeats, a.warmup, out)
        return
    for block in (1, 2, 3):                      # separate processes, as required by §1.2.1
        subprocess.run([sys.executable, __file__, "--block", str(block), "--batch", str(a.batch),
                        "--repeats", str(a.repeats), "--warmup", str(a.warmup), "--out-dir", str(out)], check=True)
    print(f"benchmark blocks written to {out}")


if __name__ == "__main__":
    main()
