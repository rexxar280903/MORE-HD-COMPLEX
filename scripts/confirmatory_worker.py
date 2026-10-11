"""One resumable classical-baseline cell or secondary MADBase run."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import constants as C
for variable in C.THREAD_ENV_VARS:
    os.environ[variable] = "1"
from core.artifact_io import load_json, save_json_atomic
from core.classical_baselines import run_baseline_cell
from core.config import RunConfig
from core.jalur_b import run_jalur_b, select_from_source
from core.run_status import existing_attempts, read_status


def completed_source(architecture, feature, k, seed):
    cfg = RunConfig(architecture, feature, k, seed, dataset="MADBASE", run_mode="CONFIRMATORY", project_root=str(ROOT))
    completed = [path for _, path, status in existing_attempts(cfg.run_dir)
                 if status and status.get("status") == "COMPLETED"]
    if len(completed) != 1:
        raise RuntimeError(f"Expected one completed source: {cfg.run_uid}")
    return completed[0]


def baseline_directory(seed, feature, k):
    return ROOT / "runs/arabic_digits/baselines" / f"S{seed}_cls-{'-'.join(str(c) for c in range(k))}_{feature}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("baseline", "jalurb", "export"))
    p.add_argument("--seed", type=int)
    p.add_argument("--feature")
    p.add_argument("--k", type=int)
    p.add_argument("--architecture", default="MORE-HD")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--run-dir")
    args = p.parse_args()
    if args.kind == "export":
        import shutil
        from core.workbook import update_local_run_spreadsheet
        directory = Path(args.run_dir)
        cfg = load_json(directory / "config.json")
        templates = {"PRIMARY": C.MADBASE_MASTER_SPREADSHEET_PATH,
                     "ABLATION": C.MADBASE_ABLATION_SPREADSHEET_PATH,
                     "MORE_REFERENCE": C.MADBASE_MORE_REFERENCE_SPREADSHEET_PATH}
        shutil.copyfile(ROOT / templates[cfg["track"]], directory / C.LOCAL_RUN_SPREADSHEET_NAME)
        update_local_run_spreadsheet(directory)
    elif args.kind == "baseline":
        # Both paired source models must be finished before their official-test arrays are read.
        completed_source("MORE-HD", args.feature, args.k, args.seed)
        completed_source("MORE-HD-C", args.feature, args.k, args.seed)
        directory = baseline_directory(args.seed, args.feature, args.k)
        rows_path = directory / "baseline_rows.json"
        if rows_path.exists():
            rows = load_json(rows_path)
            assert len(rows) == 3 and {r["baseline"] for r in rows} == {"NC", "LR", "CHANCE"}
            print("Completed baseline cell retained", flush=True)
            return
        rows = run_baseline_cell(ROOT, args.seed, args.feature, args.k, directory.parent, dataset="MADBASE")
        save_json_atomic(rows, rows_path)
        print(f"Completed two fits and chance: S{args.seed} {args.feature} K{args.k}", flush=True)
    else:
        source = completed_source(args.architecture, args.feature, args.k, args.seed)
        selected, _ = select_from_source(source)
        directory = source.with_name(source.name + f"_jalurB_auto_eval{selected:04d}")
        if directory.exists():
            status = read_status(directory)
            if status and status.get("status") == "COMPLETED":
                print("Completed secondary run retained", flush=True)
                return
            raise RuntimeError(f"Incomplete secondary attempt needs review: {directory}")
        run_jalur_b(source, args.parallel, ROOT)
        print(f"Completed Jalur B: {directory.name}", flush=True)


if __name__ == "__main__":
    main()
