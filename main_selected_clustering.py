#!/usr/bin/env python3
"""Jalur B: secondary path from a deterministic validation-selected clustering checkpoint (pseudocode §10)."""

from core.environment import lock_threads

lock_threads()  # G4-01

import argparse  # noqa: E402
import json  # noqa: E402

from core.jalur_b import run_jalur_b  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run-dir", required=True, help="COMPLETED primary Jalur A run folder")
    p.add_argument("--n-parallel-declared", type=int, default=None)
    p.add_argument("--project-root", default=".")
    args = p.parse_args()
    m = run_jalur_b(args.source_run_dir, args.n_parallel_declared, args.project_root)
    print(json.dumps({"run_name": m["run_name"], "selected_eval_id": m["selected_eval_id"],
                      "n_eligible_eval_ids": m["n_eligible_eval_ids"],
                      "accuracy": m["final_metrics"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
