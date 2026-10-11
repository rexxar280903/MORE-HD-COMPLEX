"""Kemas input analisis mekanistik eksploratif MADBase (180 run sel ablasi, K in {3,6,10}).

Isi arsip: core/*.py, per run: config.json, logs/metrics_final.json, artifacts/{supervised_params_final.npy,
quantum_labels.json}; per sel (fitur, K, seed): X_train_scaled, y_train, X_test_scaled, y_test (identik
untuk A/B/C/D karena paired_input_check = MATCH). Ditambah ringkasan lintasan loss (convergence.csv).
Arsip ini hanya berisi fitur 8 kanal terskala, bukan citra mentah MADBase.
"""
import csv, io, json, tarfile, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = REPO / "runs" / "arabic_digits"
ARCH = {"MORE-HD", "MORE-HD-C", "MORE-HD-60P", "MORE-HD-C-FixedRZ"}


def runs():
    out = []
    for cfgp in list(RUNS.glob("cls-*/config.json")) + list((RUNS / "ablation").glob("ABL*/config.json")):
        d = cfgp.parent
        if "jalurB" in d.name or "smoke" in d.name or "convergence" in d.name:
            continue
        cfg = json.loads(cfgp.read_text())
        if cfg.get("run_mode") != "CONFIRMATORY" or cfg.get("architecture") not in ARCH:
            continue
        if int(cfg["n_classes"]) not in (3, 6, 10):
            continue
        out.append((d, cfg))
    return out


def traj(path):
    loss = [json.loads(l)["train_loss"] for l in path.read_text().splitlines() if l.strip()]
    best = np.minimum.accumulate(np.array(loss))
    total = best[0] - best[-1]
    n = len(best)
    last = best[max(0, n - 101)] - best[-1]
    return n, bool(total > 0 and last / total > 0.02), float(last / total) if total > 0 else 0.0


rs = runs()
print(len(rs), "runs")
rows = []
tar_path = HERE / "mech_inputs.tar.gz"
cells_done = set()
with tarfile.open(tar_path, "w:gz") as tar:
    for f in sorted((REPO / "core").glob("*.py")):
        tar.add(f, arcname=f"core/{f.name}")
    for d, cfg in sorted(rs, key=lambda t: str(t[0])):
        key = f"{cfg['architecture']}__{cfg['feature_method']}__K{cfg['n_classes']}__S{cfg['seed']}"
        for rel in ("config.json", "logs/metrics_final.json", "artifacts/supervised_params_final.npy", "artifacts/quantum_labels.json"):
            tar.add(d / rel, arcname=f"runs/{key}/{rel}")
        cell = f"{cfg['feature_method']}__K{cfg['n_classes']}__S{cfg['seed']}"
        hashes = {}
        for a in ("X_train_scaled.npy", "y_train.npy", "X_test_scaled.npy", "y_test.npy"):
            hashes[a] = hashlib.sha256((d / "artifacts" / a).read_bytes()).hexdigest()
            if cell not in cells_done:
                tar.add(d / "artifacts" / a, arcname=f"cells/{cell}/{a}")
        cells_done.add(cell)
        sn, si, sf = traj(d / "logs" / "supervised_log.jsonl")
        cn, ci, cf = traj(d / "logs" / "clustering_log.jsonl")
        rows.append({"key": key, "architecture": cfg["architecture"], "feature": cfg["feature_method"], "K": cfg["n_classes"],
                     "seed": cfg["seed"], "cell": cell, "xtrain_sha": hashes["X_train_scaled.npy"][:16],
                     "xtest_sha": hashes["X_test_scaled.npy"][:16], "sup_n_eval": sn, "sup_still_improving_last100": si,
                     "sup_frac_improvement_last100": sf, "clu_n_eval": cn, "clu_still_improving_last100": ci,
                     "clu_frac_improvement_last100": cf, "run_dir": str(d.relative_to(REPO))})
with (HERE / "convergence.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
# paired input check across architectures within a cell
import collections
cc = collections.defaultdict(set)
for r in rows:
    cc[r["cell"]].add((r["xtrain_sha"], r["xtest_sha"]))
print("cells", len(cc), "cells with mismatched inputs", sum(len(v) > 1 for v in cc.values()))
print("tar MB", round(tar_path.stat().st_size / 1e6, 1))
