"""Shared command-line handling for the run entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import constants as C


def add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET_MNIST,
                   help="dataset eksplisit; gunakan MADBASE untuk digit Arab")
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
        if args.dataset == C.DATASET_MADBASE:
            decision_path = (Path(args.project_root).resolve() / "research_arabic_digits" /
                             "research_data" / "pilot" / "convergence_budget_decision.json")
            if not decision_path.is_file():
                sys.exit(
                    "MADBase confirmatory runs are blocked: run the convergence pilot and "
                    "scripts/analyze_pilot_budget.py --dataset MADBASE first"
                )
            with decision_path.open(encoding="utf-8") as fh:
                decision = json.load(fh)
            if decision.get("status") != "FROZEN" or decision.get("dataset") != C.DATASET_MADBASE:
                sys.exit(f"invalid MADBase convergence decision: {decision_path}")
            final_c = int(decision["clustering"]["final_max_nfev"])
            final_s = int(decision["supervised"]["final_max_nfev"])
        else:
            final_c, final_s = C.FINAL_MAX_NFEV_CLUSTERING, C.FINAL_MAX_NFEV_SUPERVISED
        if final_c is None or final_s is None:
            sys.exit("final COBYLA budget not frozen yet (G1-01); confirmatory runs are blocked")
        for given, final in ((args.max_nfev_clustering, final_c),
                             (args.max_nfev_supervised, final_s)):
            if given is not None and given != final:
                sys.exit(f"CONFIRMATORY budget is frozen at {final}; got {given}")
        return final_c, final_s
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
        mnist_download=args.mnist_download, pilot_tag=args.pilot_tag, dataset=args.dataset,
    )


def run_and_report(cfg, write_workbook: bool) -> None:
    from .pipeline import run_condition

    print(f"[{cfg.run_uid}] {cfg.run_name}", flush=True)
    manifest = run_condition(cfg, write_workbook=write_workbook)
    out = {
        "run_uid": manifest["run_uid"], "run_dir": f"runs/{manifest['run_name']}",
        "dataset_id": manifest["dataset_id"],
        "accuracy": manifest["final_metrics"]["accuracy"], "f1_macro": manifest["final_metrics"]["f1_score"],
        "clustering_nfev": manifest["optimizer_results"]["clustering"]["nfev"],
        "supervised_nfev": manifest["optimizer_results"]["supervised"]["nfev"],
        "total_runtime_sec": round(manifest["total_runtime_sec"], 1),
    }
    print(json.dumps(out, indent=2))
