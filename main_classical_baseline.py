#!/usr/bin/env python3
"""Classical reference baselines G1-10 (chance, Nearest Centroid, Logistic Regression), pseudocode §12."""

from core.environment import lock_threads

lock_threads()  # G4-01

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from core import constants as C  # noqa: E402
from core.artifact_io import save_json_atomic  # noqa: E402
from core.classical_baselines import baseline_config, run_baseline_cell, write_results_csv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-mode", choices=C.RUN_MODES, default="CONFIRMATORY")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--features", nargs="*", default=list(C.FEATURE_METHODS))
    p.add_argument("--ks", type=int, nargs="*", default=list(C.K_VALUES))
    p.add_argument("--sizes", type=int, nargs=3, default=[C.N_TRAIN_PER_CLASS, C.N_VAL_PER_CLASS, C.N_TEST_PER_CLASS],
                   help="per-class train/val/test sizes of the source runs (PILOT only)")
    p.add_argument("--pilot-tag", default="", help="PILOT only: tag of the source pilot runs")
    p.add_argument("--project-root", default=".")
    args = p.parse_args()
    seeds = args.seeds or (list(C.CONFIRMATORY_SEEDS) if args.run_mode == "CONFIRMATORY" else [C.PILOT_SEED])
    if args.run_mode == "CONFIRMATORY" and (set(seeds) - set(C.CONFIRMATORY_SEEDS) or
                                            tuple(args.sizes) != (C.N_TRAIN_PER_CLASS, C.N_VAL_PER_CLASS, C.N_TEST_PER_CLASS)):
        raise SystemExit("confirmatory baselines use seeds 101..505 and the 1000/100/200 protocol")
    root = Path(args.project_root).resolve()
    out_root = root / C.RUNS_DIR / ("baselines" if args.run_mode == "CONFIRMATORY" else "baselines_pilot")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = {**baseline_config(), "seeds": seeds, "features": args.features, "K_values": args.ks,
           "run_mode": args.run_mode}
    save_json_atomic(cfg, out_root / "baseline_config.json")
    rows, errors = [], []
    for seed in seeds:
        for k in args.ks:
            for feature in args.features:
                try:
                    rows += run_baseline_cell(root, seed, feature, k, out_root, tuple(args.sizes),
                                              run_mode=args.run_mode, pilot_tag=args.pilot_tag)
                except Exception as exc:  # cell skipped and logged; rerun later in a new folder
                    errors.append({"seed": seed, "K": k, "feature_method": feature, "error": repr(exc)})
    write_results_csv(rows, out_root / "baseline_results.csv")
    save_json_atomic(errors, out_root / "baseline_errors.json")
    n_fit = sum(1 for r in rows if r["baseline"] in ("NC", "LR"))
    print(f"{n_fit} baseline fits written to {out_root}; {len(errors)} cells skipped")


if __name__ == "__main__":
    main()
