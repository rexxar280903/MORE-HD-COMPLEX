#!/usr/bin/env python3
"""Print the frozen confirmatory run plan (450 executions + 90 Jalur B) as shell commands.

The protocol runs every condition as its own process and folder ("controlled
manual per run"). This script only lists the commands and the current status of
each run_uid; it does not launch anything. Use --pending to list only runs that
have no COMPLETED attempt yet.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import constants as C  # noqa: E402
from core.config import RunConfig  # noqa: E402
from core.run_status import existing_attempts  # noqa: E402


def entries():
    for seed in C.CONFIRMATORY_SEEDS:
        for k in C.K_VALUES:
            for fm in C.FEATURE_METHODS:
                for arch in C.PRIMARY_ARCHITECTURES:
                    yield ("PRIMARY", "main_train.py", f"--architecture {arch}", arch, fm, k, seed)
    for seed in C.CONFIRMATORY_SEEDS:
        for k in C.ABLATION_K_VALUES:
            for fm in C.FEATURE_METHODS:
                for model in C.ABLATION_MODELS:
                    yield ("ABLATION", "main_ablation.py", f"--model {model}", model, fm, k, seed)
    for seed in C.CONFIRMATORY_SEEDS:
        for k in C.K_VALUES:
            for fm in C.FEATURE_METHODS:
                yield ("MORE_REFERENCE", "main_more_reference.py", "", C.ARCH_MORE_REPRO, fm, k, seed)


def status_of(cfg):
    states = [st.get("status") if st else "NO_STATUS" for _, _, st in existing_attempts(ROOT / "runs" / cfg.run_name)]
    if C.STATUS_COMPLETED in states:
        return C.STATUS_COMPLETED
    return states[-1] if states else "NOT_STARTED"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pending", action="store_true")
    p.add_argument("--track", choices=["PRIMARY", "ABLATION", "MORE_REFERENCE"], default=None)
    p.add_argument("--n-parallel", type=int, default=1, help="value written into --n-parallel-declared")
    a = p.parse_args()
    n = done = 0
    for track, script, flag, arch, fm, k, seed in entries():
        if a.track and track != a.track:
            continue
        cfg = RunConfig(arch, fm, k, seed, run_mode="CONFIRMATORY", n_parallel_declared=a.n_parallel)
        st = status_of(cfg)
        n += 1
        done += st == C.STATUS_COMPLETED
        if a.pending and st == C.STATUS_COMPLETED:
            continue
        cmd = f"python {script} {flag} --feature {fm} --k {k} --seed {seed} --run-mode CONFIRMATORY --n-parallel-declared {a.n_parallel}"
        print(f"# {cfg.run_uid:<12} {st:<12} runs/{cfg.run_name}\n{' '.join(cmd.split())}")
    print(f"# {done}/{n} COMPLETED")
    if not a.track or a.track == "PRIMARY":
        print("# Jalur B (secondary, after the source run is COMPLETED): K in {3,6,10}, A and D, 5 seeds = 90")
        print("#   python main_selected_clustering.py --source-run-dir runs/<primary run> --n-parallel-declared N")


if __name__ == "__main__":
    main()
