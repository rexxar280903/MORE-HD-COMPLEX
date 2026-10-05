#!/usr/bin/env python3
"""Frozen analysis (MORE_HD_STATISTICAL_ANALYSIS_PLAN.md v2.0) from run artifacts.

Outputs (research_data/analysis/ by default):
* seed_aggregation.csv            SAP §7  five-seed summaries per track/model/feature/K
* paired_primary_D_minus_A.csv    SAP §6, §8, §9  primary estimand with CI, Hedges g, sign-flip
* ablation_contrasts.csv          SAP §6.1  A-B, A-C, B-D, C-D with Holm over the four contrasts
* more_comparison.csv             SAP §10.3  MORE-REPRO vs A and D (paired), plus Table I (descriptive)
* mcnemar_primary.csv             SAP §9  secondary instance-level test, Holm within seed
* g1_11_six_nine.csv              SAP §10.4  6<->9 confusion and merged-6/9 accuracy at K=10
* baseline_comparison.csv         SAP §10.2  quantum minus NC/LR (descriptive)
* MORE_HD_analysis_tables.xlsx    the same tables as sheets

``--run-mode PILOT`` analyses seed-42 pilot runs only to verify the machinery;
pilot numbers are never confirmatory results.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402

from core import constants as C  # noqa: E402
from core.artifact_io import load_json, load_npy  # noqa: E402
from core.stats import describe, holm, mcnemar_exact, paired_summary  # noqa: E402

METRICS = {
    "accuracy": ("final_metrics", "accuracy"),
    "f1_macro": ("final_metrics", "f1_score"),
    "min_separation_ratio": ("cluster_row", "min_separation_ratio"),
    "min_separation": ("cluster_row", "min_separation"),
    "min_separation_val_ratio": ("cluster_row", "min_separation_val_ratio"),
    "yodd_norm_fraction": ("structural", "yodd_norm_fraction"),
}
PRIMARY_INFERENTIAL = ("accuracy", "f1_macro", "min_separation_ratio")


def load_runs(runs_root: Path, run_mode: str, pilot_tag: str | None = None) -> list[dict]:
    runs = []
    for cfg_path in sorted(runs_root.rglob("config.json")):
        m = load_json(cfg_path)
        if "track" not in m or m.get("track") == "JALUR_B" or m.get("run_mode") != run_mode:
            continue
        if run_mode == "PILOT" and pilot_tag is not None and m.get("pilot_tag") != pilot_tag:
            continue
        d = cfg_path.parent
        st = load_json(d / "logs" / "run_status.json")
        if st["status"] != C.STATUS_COMPLETED:
            continue
        opt = load_json(d / "logs" / "clustering_optimizer_result.json")
        rows = [line for line in (d / "logs" / "clustering_log.jsonl").read_text().splitlines() if line.strip()]
        import json

        crow = json.loads(rows[opt["final_point_eval_id"]])
        sd = load_json(d / "artifacts" / "structural_diagnostics.json")["checkpoints"]["clustering_final"]
        runs.append({"dir": d, "manifest": m, "cluster_row": crow, "structural": sd,
                     "final_metrics": m["final_metrics"]})
    uids = [r["manifest"]["run_uid"] for r in runs]
    if len(uids) != len(set(uids)):
        raise SystemExit("duplicate COMPLETED run_uid; resolve attempts first (G3-04)")
    return runs


def metric(run: dict, name: str):
    src, key = METRICS[name]
    return run[src].get(key)


def index_runs(runs):
    idx = {}
    for r in runs:
        m = r["manifest"]
        idx[(m["model_code"], m["feature_method"], m["n_classes"], m["seed"])] = r
    return idx


def seed_aggregation(runs):
    groups = defaultdict(list)
    for r in runs:
        m = r["manifest"]
        groups[(m["track"], m["model_code"], m["architecture"], m["feature_method"], m["n_classes"])].append(r)
    out = []
    for (track, code, arch, fm, k), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][4])):
        row = {"track": track, "model_code": code, "architecture": arch, "feature_method": fm, "K": k,
               "n_seeds": len(rs), "seeds": ",".join(str(r["manifest"]["seed"]) for r in rs)}
        for name in METRICS:
            d = describe([metric(r, name) for r in rs])
            for stat in ("mean", "sd", "median", "iqr"):
                row[f"{name}_{stat}"] = d[stat]
        rt = describe([r["manifest"]["total_runtime_sec"] for r in rs])
        row.update({"runtime_mean_sec": rt["mean"], "runtime_median_sec": rt["median"], "runtime_iqr_sec": rt["iqr"]})
        out.append(row)
    return out


def paired_rows(idx, code_a, code_b, metrics, seeds, label):
    out = []
    cells = sorted({(fm, k) for (c, fm, k, s) in idx if c in (code_a, code_b)})
    for fm, k in cells:
        for name in metrics:
            a_vals, b_vals, used = [], [], []
            for s in seeds:
                ra, rb = idx.get((code_a, fm, k, s)), idx.get((code_b, fm, k, s))
                if ra is None or rb is None:
                    continue
                a_vals.append(metric(ra, name))
                b_vals.append(metric(rb, name))
                used.append(s)
            if not used:
                continue
            s_ = paired_summary(a_vals, b_vals)
            out.append({"contrast": label, "feature_method": fm, "K": k, "metric": name, "seeds": ",".join(map(str, used)),
                        **{kk: v for kk, v in s_.items() if kk != "deltas"},
                        "deltas": ",".join(f"{x:.6g}" for x in s_["deltas"])})
    return out


def ablation_contrasts(idx, seeds):
    rows = []
    for a, b, lab in (("A", "B", "A-B (B-A)"), ("A", "C", "A-C (C-A)"), ("B", "D", "B-D (D-B)"), ("C", "D", "C-D (D-C)")):
        rows += [r for r in paired_rows(idx, a, b, PRIMARY_INFERENTIAL + ("yodd_norm_fraction",), seeds, lab)
                 if r["K"] in C.ABLATION_K_VALUES]
    fam = defaultdict(list)
    for i, r in enumerate(rows):
        if r["metric"] in PRIMARY_INFERENTIAL:
            fam[(r["feature_method"], r["K"], r["metric"])].append(i)
    for members in fam.values():
        adj = holm([rows[i]["sign_flip_p"] for i in members])
        for i, p in zip(members, adj):
            rows[i]["sign_flip_p_holm"] = p
    return rows


def more_comparison(idx, seeds):
    rows = paired_rows(idx, "M", "A", ("accuracy", "f1_macro", "min_separation", "min_separation_ratio"), seeds, "M-A (A-M)")
    rows += paired_rows(idx, "M", "D", ("accuracy", "f1_macro", "min_separation", "min_separation_ratio"), seeds, "M-D (D-M)")
    for r in rows:
        r["MORE_TableI_without_R_pct"], r["MORE_TableI_with_R_pct"] = C.MORE_TABLE_I[r["K"]]
    return rows


def mcnemar_primary(idx, seeds):
    rows = []
    for s in seeds:
        fam = []
        for fm in C.FEATURE_METHODS:
            for k in C.K_VALUES:
                ra, rd = idx.get(("A", fm, k, s)), idx.get(("D", fm, k, s))
                if ra is None or rd is None:
                    continue
                ya = load_npy(ra["dir"] / "artifacts" / "y_test.npy")
                pa = load_npy(ra["dir"] / "logs" / "test_predictions.npy")
                pd_ = load_npy(rd["dir"] / "logs" / "test_predictions.npy")
                if not np.array_equal(load_npy(rd["dir"] / "logs" / "test_mnist_index.npy"),
                                      load_npy(ra["dir"] / "logs" / "test_mnist_index.npy")):
                    raise SystemExit("A/D official-test identities differ")
                res = mcnemar_exact(pa == ya, pd_ == ya)
                fam.append({"seed": s, "feature_method": fm, "K": k, **res})
        adj = holm([r["p_value"] for r in fam])
        for r, p in zip(fam, adj):
            r["p_value_holm_within_seed"] = p
        rows += fam
    return rows


def six_nine(runs, baseline_root: Path | None):
    rows = []
    for r in runs:
        m = r["manifest"]
        if m["n_classes"] != 10:
            continue
        cm = load_npy(r["dir"] / "logs" / "confusion_matrix.npy")
        rows.append(_six_nine_row(cm, m["model_code"], m["architecture"], m["feature_method"], m["seed"]))
    if baseline_root and baseline_root.exists():
        for cell in sorted(baseline_root.glob("S*_cls-0-1-2-3-4-5-6-7-8-9_*")):
            seed = int(cell.name.split("_")[0][1:])
            fm = cell.name.split("_")[-1]
            for name in ("NC", "LR"):
                cm = load_npy(cell / f"{name}_confusion_matrix.npy")
                rows.append(_six_nine_row(cm, name, name, fm, seed))
    return rows


def _six_nine_row(cm, code, arch, fm, seed):
    total = cm.sum()
    n69 = int(cm[6, 9] + cm[9, 6])
    support = int(cm[6].sum() + cm[9].sum())
    errors = int(total - np.trace(cm))
    return {"model_code": code, "architecture": arch, "feature_method": fm, "seed": seed,
            "accuracy": float(np.trace(cm) / total), "accuracy_merged_6_9": float((np.trace(cm) + n69) / total),
            "six_nine_confusions": n69, "six_nine_confusion_rate": n69 / support if support else None,
            "share_of_errors_6_9": n69 / errors if errors else None}


def baseline_comparison(idx, baseline_root: Path | None, seeds):
    if not baseline_root or not (baseline_root / "baseline_results.csv").exists():
        return []
    rows = []
    with open(baseline_root / "baseline_results.csv", encoding="utf-8") as fh:
        base = list(csv.DictReader(fh))
    for b in base:
        if b["baseline"] not in ("NC", "LR"):
            continue
        fm, k, s = b["feature_method"], int(b["K"]), int(b["seed"])
        if s not in seeds:
            continue
        for code in ("A", "D", "M"):
            q = idx.get((code, fm, k, s))
            if q is None:
                continue
            rows.append({"feature_method": fm, "K": k, "seed": s, "baseline": b["baseline"], "quantum_model": code,
                         "delta_accuracy": metric(q, "accuracy") - float(b["accuracy"]),
                         "delta_f1": metric(q, "f1_macro") - float(b["macro_f1"])})
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-mode", choices=C.RUN_MODES, default="CONFIRMATORY")
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--pilot-tag", default=None, help="PILOT only: analyse one pilot stage (e.g. convergence)")
    a = p.parse_args()
    root = Path(a.project_root)
    runs = load_runs(root / C.RUNS_DIR, a.run_mode, a.pilot_tag)
    seeds = list(C.CONFIRMATORY_SEEDS) if a.run_mode == "CONFIRMATORY" else [C.PILOT_SEED]
    out = Path(a.out_dir) if a.out_dir else root / "research_data" / ("analysis" if a.run_mode == "CONFIRMATORY" else "analysis_pilot")
    out.mkdir(parents=True, exist_ok=True)
    idx = index_runs(runs)
    broot = root / C.RUNS_DIR / ("baselines" if a.run_mode == "CONFIRMATORY" else "baselines_pilot")
    tables = {
        "seed_aggregation": seed_aggregation(runs),
        "paired_primary_D_minus_A": paired_rows(idx, "A", "D", tuple(METRICS), seeds, "A-D (D-A)"),
        "ablation_contrasts": ablation_contrasts(idx, seeds),
        "more_comparison": more_comparison(idx, seeds),
        "mcnemar_primary": mcnemar_primary(idx, seeds),
        "g1_11_six_nine": six_nine(runs, broot),
        "baseline_comparison": baseline_comparison(idx, broot, seeds),
    }
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in tables.items():
        write_csv(rows, out / f"{name}.csv")
        ws = wb.create_sheet(name[:31])
        if rows:
            hdr = list(rows[0].keys())
            ws.append(hdr)
            for r in rows:
                ws.append([r.get(h) if not isinstance(r.get(h), (list, dict)) else str(r.get(h)) for h in hdr])
    wb.save(out / "MORE_HD_analysis_tables.xlsx")
    print(f"{len(runs)} runs analysed ({a.run_mode}); tables in {out}")
    for name, rows in tables.items():
        print(f"  {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
