"""Shared command-line handling for the run entry points."""

from __future__ import annotations

import argparse
import json
import sys

from . import constants as C


def add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--feature", required=True, choices=C.FEATURE_METHODS)
    p.add_argument("--k", required=True, type=int, choices=C.K_VALUES)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--run-mode", required=True, choices=C.RUN_MODES)
    p.add_argument("--max-nfev-clustering", type=int, default=None,
                   help="PILOT only; CONFIRMATORY uses the frozen final budget")
    p.add_argument("--max-nfev-supervised", type=int, default=None,
                   help="PILOT only; CONFIRMATORY uses the frozen final budget")
    p.add_argument("--n-parallel-declared", type=int, default=None,
                   help="number of runs deliberately running at the same time (required for CONFIRMATORY)")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--mnist-download", action="store_true", help="download MNIST (first pilot only)")
    p.add_argument("--pilot-tag", default="", help="free label for PILOT runs, e.g. smoke / timing / convergence")
    p.add_argument("--n-train-per-class", type=int, default=C.N_TRAIN_PER_CLASS, help="PILOT only")
    p.add_argument("--n-val-per-class", type=int, default=C.N_VAL_PER_CLASS, help="PILOT only")
    p.add_argument("--n-test-per-class", type=int, default=C.N_TEST_PER_CLASS, help="PILOT only")
    p.add_argument("--project-root", default=".")
    p.add_argument("--no-workbook", action="store_true", help="skip the per-run run_result.xlsx")


def budget_from_args(args) -> tuple[int, int]:
    if args.run_mode == "CONFIRMATORY":
        if C.FINAL_MAX_NFEV_CLUSTERING is None or C.FINAL_MAX_NFEV_SUPERVISED is None:
            sys.exit("final COBYLA budget not frozen yet (G1-01); confirmatory runs are blocked")
        for given, final in ((args.max_nfev_clustering, C.FINAL_MAX_NFEV_CLUSTERING),
                             (args.max_nfev_supervised, C.FINAL_MAX_NFEV_SUPERVISED)):
            if given is not None and given != final:
                sys.exit(f"CONFIRMATORY budget is frozen at {final}; got {given}")
        return C.FINAL_MAX_NFEV_CLUSTERING, C.FINAL_MAX_NFEV_SUPERVISED
    return (args.max_nfev_clustering or C.SMOKE_MAX_NFEV, args.max_nfev_supervised or C.SMOKE_MAX_NFEV)


def make_config(args, architecture: str):
    from .config import RunConfig

    mc, ms = budget_from_args(args)
    return RunConfig(
        architecture=architecture, feature_method=args.feature, k=args.k, seed=args.seed,
        run_mode=args.run_mode, max_nfev_clustering=mc, max_nfev_supervised=ms,
        n_train_per_class=args.n_train_per_class, n_val_per_class=args.n_val_per_class,
        n_test_per_class=args.n_test_per_class, attempt=args.attempt,
        n_parallel_declared=args.n_parallel_declared, project_root=args.project_root,
        mnist_download=args.mnist_download, pilot_tag=args.pilot_tag,
    )


def run_and_report(cfg, write_workbook: bool) -> None:
    from .pipeline import run_condition

    print(f"[{cfg.run_uid}] {cfg.run_name}", flush=True)
    manifest = run_condition(cfg, write_workbook=write_workbook)
    out = {
        "run_uid": manifest["run_uid"], "run_dir": f"runs/{manifest['run_name']}",
        "accuracy": manifest["final_metrics"]["accuracy"], "f1_macro": manifest["final_metrics"]["f1_score"],
        "clustering_nfev": manifest["optimizer_results"]["clustering"]["nfev"],
        "supervised_nfev": manifest["optimizer_results"]["supervised"]["nfev"],
        "total_runtime_sec": round(manifest["total_runtime_sec"], 1),
    }
    print(json.dumps(out, indent=2))
