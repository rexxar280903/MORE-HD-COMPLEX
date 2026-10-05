#!/usr/bin/env python3
"""Launch the seed-42 PILOT ONLY stages (pseudocode §1.1, §1.2): smoke, timing, convergence.

Every run is still an independent process with its own run folder (one run =
one condition); this helper only spares typing the commands. It refuses any
seed other than 42 and never starts CONFIRMATORY runs.
"""

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import constants as C  # noqa: E402

ENTRY = {
    C.ARCH_MORE_HD: ("main_train.py", "--architecture"), C.ARCH_MORE_HD_C: ("main_train.py", "--architecture"),
    C.ARCH_MORE_HD_60P: ("main_ablation.py", "--model"), C.ARCH_MORE_HD_C_FIXED_RZ: ("main_ablation.py", "--model"),
    C.ARCH_MORE_REPRO: ("main_more_reference.py", None),
}
ALL5 = [C.ARCH_MORE_HD, C.ARCH_MORE_HD_C, C.ARCH_MORE_HD_60P, C.ARCH_MORE_HD_C_FIXED_RZ, C.ARCH_MORE_REPRO]


def plan(stage: str) -> list[tuple]:
    if stage == "smoke":       # every code path, cheapest condition
        return [(a, "PCA", 3, C.SMOKE_MAX_NFEV) for a in ALL5]
    if stage == "timing":      # most expensive class count, all models
        return [(a, "PCA", 10, C.SMOKE_MAX_NFEV) for a in ALL5]
    if stage == "convergence":  # frozen G1-01 design (+ MORE-REPRO, decision 2026-10-04)
        runs = [(a, f, k, C.PILOT_MAX_NFEV_CAP) for k, f, a in
                itertools.product((3, 10), ("PCA", "ZERNIKE"), (C.ARCH_MORE_HD, C.ARCH_MORE_HD_C, C.ARCH_MORE_REPRO))]
        runs += [(C.ARCH_MORE_HD_60P, f, 10, C.PILOT_MAX_NFEV_CAP) for f in ("PCA", "ZERNIKE")]
        first = (C.ARCH_MORE_HD_C, "PCA", 3, C.PILOT_MAX_NFEV_CAP)       # checked alone first (§1.1)
        runs.remove(first)
        return [first] + runs
    raise ValueError(stage)


def command(arch, feature, k, nfev, tag, n_parallel):
    script, flag = ENTRY[arch]
    cmd = [sys.executable, str(ROOT / script)]
    if flag:
        cmd += [flag, arch]
    cmd += ["--feature", feature, "--k", str(k), "--seed", "42", "--run-mode", "PILOT",
            "--max-nfev-clustering", str(nfev), "--max-nfev-supervised", str(nfev),
            "--pilot-tag", tag, "--n-parallel-declared", str(n_parallel), "--project-root", str(ROOT)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["smoke", "timing", "convergence"])
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--first-only", action="store_true", help="convergence: run only the first check run")
    p.add_argument("--skip-first", action="store_true", help="convergence: run all but the first run")
    a = p.parse_args()
    runs = plan(a.stage)
    if a.stage == "convergence" and a.first_only:
        runs = runs[:1]
    if a.stage == "convergence" and a.skip_first:
        runs = runs[1:]
    log_dir = ROOT / "runs" / "pilot_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def launch(r):
        arch, feature, k, nfev = r
        name = f"{a.stage}_{arch}_{feature}_K{k}"
        t0 = time.time()
        with open(log_dir / f"{name}.log", "w") as fh:
            rc = subprocess.run(command(arch, feature, k, nfev, a.stage, a.parallel), stdout=fh, stderr=subprocess.STDOUT).returncode
        return name, rc, time.time() - t0

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for name, rc, dt in ex.map(launch, runs):
            print(f"{'OK  ' if rc == 0 else 'FAIL'} {name} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
