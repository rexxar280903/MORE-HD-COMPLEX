"""Kumpulkan hasil Jalur B (analisis sensitivitas sekunder) MADBase ke satu CSV.

Sumber: runs/arabic_digits/*_jalurB_*/{config.json, logs/metrics_final.json, logs/run_status.json}
Keluaran: research_arabic_digits/paper/build/data/jalurB_runs.csv
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "runs" / "arabic_digits"
FIELDS = ["run_uid", "source_run_uid", "source_condition_id", "architecture", "feature_method", "n_classes", "seed",
          "run_mode", "selected_eval_id", "n_eligible_eval_ids", "selection_method", "test_used_for_selection",
          "max_nfev_supervised", "degenerate_quantum_label", "status", "attempt", "accuracy", "f1_score",
          "precision", "recall", "n_test", "n_correct", "n_degenerate_test_outputs", "total_runtime_sec", "run_dir"]
rows = []
for d in sorted(RUNS.glob("cls-*_jalurB_*")):
    cfg = json.loads((d / "config.json").read_text())
    met = json.loads((d / "logs" / "metrics_final.json").read_text())
    st = json.loads((d / "logs" / "run_status.json").read_text())
    r = {**{k: cfg.get(k) for k in FIELDS}, **{k: met.get(k) for k in FIELDS if k in met},
         "status": st.get("status"), "attempt": st.get("attempt"), "run_dir": d.name}
    rows.append(r)
out = HERE / "data" / "jalurB_runs.csv"
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(len(rows), "rows ->", out)
