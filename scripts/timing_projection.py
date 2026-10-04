#!/usr/bin/env python3
"""Timing pilot projection and the G3-05 FULL_VAL feasibility verdict (pseudocode §1.2).

Uses the seed-42 smoke (K=3) and timing (K=10) runs of every model. Per model and
loop, the mean objective-evaluation time is modelled as t(n) = a + b * n where n
is the number of circuit executions per evaluation (clustering: 5K + 100K,
supervised: 1100K under FULL_VAL), fitted through the K=3 and K=10 points.
Fixed per-run stage overhead (data pipeline, setup, labels, diagnostics, final
evaluation) is interpolated the same way.

Criterion written into gate G3-05 before this pilot (2026-10-04): FULL_VAL is
infeasible iff, at the pilot cap for both loops, (i) the projected single-thread
compute for all confirmatory executions (240 primary + 90 ablation + 120 MORE
reference) plus the pre-specified Jalur B subset (90) exceeds 240 CPU-hours on the
reference machine, or (ii) validation monitoring exceeds 50% of the projected
objective-evaluation time.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import constants as C  # noqa: E402
from core.artifact_io import load_json, read_jsonl, save_json_atomic  # noqa: E402

LIMIT_CPU_HOURS = 240.0
LIMIT_VAL_SHARE = 0.50


def n_exec(loop, k):
    return 5 * k + 100 * k if loop == "clustering" else 1100 * k


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--cap", type=int, default=C.PILOT_MAX_NFEV_CAP)
    a = p.parse_args()
    root = Path(a.project_root)
    pts = defaultdict(dict)        # (arch, loop) -> {K: mean eval time}
    over = defaultdict(dict)       # arch -> {K: overhead sec}
    for cfg_path in sorted((root / C.RUNS_DIR).rglob("config.json")):
        m = load_json(cfg_path)
        if m.get("run_mode") != "PILOT" or m.get("pilot_tag") not in ("smoke", "timing") or m.get("track") == "JALUR_B":
            continue
        if m["n_train_per_class"] != C.N_TRAIN_PER_CLASS or m["feature_method"] != "PCA":
            continue
        d = cfg_path.parent
        k, arch = m["n_classes"], m["architecture"]
        for loop in ("clustering", "supervised"):
            rows = read_jsonl(d / "logs" / f"{loop}_log.jsonl")
            pts[(arch, loop)][k] = float(np.mean([r["eval_runtime_sec"] for r in rows[1:]]))
        sw = m["timing"]["stage_wall_sec"]
        over[arch][k] = sum(v for s, v in sw.items() if s not in ("clustering_loop", "supervised_loop"))
    fits = {}
    for (arch, loop), byk in pts.items():
        if not {3, 10} <= set(byk):
            raise SystemExit(f"need K=3 and K=10 timing for {arch} {loop}")
        n3, n10 = n_exec(loop, 3), n_exec(loop, 10)
        b = (byk[10] - byk[3]) / (n10 - n3)
        fits[(arch, loop)] = (byk[3] - b * n3, b)
    plan = []
    for k in C.K_VALUES:
        for fm in C.FEATURE_METHODS:
            plan += [(C.ARCH_MORE_HD, k), (C.ARCH_MORE_HD_C, k), (C.ARCH_MORE_REPRO, k)]
    for k in C.ABLATION_K_VALUES:
        for fm in C.FEATURE_METHODS:
            plan += [(C.ARCH_MORE_HD_60P, k), (C.ARCH_MORE_HD_C_FIXED_RZ, k)]
    plan = plan * len(C.CONFIRMATORY_SEEDS)
    jalur_b = [(arch, k) for k in C.JALUR_B_K_VALUES for fm in C.FEATURE_METHODS
               for arch in (C.ARCH_MORE_HD, C.ARCH_MORE_HD_C)] * len(C.CONFIRMATORY_SEEDS)
    total = val = 0.0
    per_arch = defaultdict(float)
    for arch, k in plan:
        o3, o10 = over[arch][3], over[arch][10]
        sec = o3 + (o10 - o3) * (k - 3) / 7
        for loop in ("clustering", "supervised"):
            a0, b = fits[(arch, loop)]
            sec += a.cap * (a0 + b * n_exec(loop, k))
            val += a.cap * b * 100 * k
        total += sec
        per_arch[arch] += sec
    jb = 0.0
    for arch, k in jalur_b:
        a0, b = fits[(arch, "supervised")]
        jb += a.cap * (a0 + b * n_exec("supervised", k))
        val += a.cap * b * 100 * k
    cpu_hours = (total + jb) / 3600.0
    share = val / (total + jb)
    verdict = {
        "cap": a.cap, "n_confirmatory_executions": len(plan), "n_jalur_b": len(jalur_b),
        "projected_cpu_hours_confirmatory": total / 3600.0, "projected_cpu_hours_jalur_b": jb / 3600.0,
        "projected_cpu_hours_total": cpu_hours, "projected_val_monitoring_share": share,
        "per_architecture_cpu_hours": {k: v / 3600 for k, v in per_arch.items()},
        "fits_sec": {f"{arch}|{loop}": {"a": f[0], "b_per_execution": f[1]} for (arch, loop), f in fits.items()},
        "criterion": {"limit_cpu_hours": LIMIT_CPU_HOURS, "limit_val_share": LIMIT_VAL_SHARE},
        "full_val_infeasible": bool(cpu_hours > LIMIT_CPU_HOURS or share > LIMIT_VAL_SHARE),
        "machine_note": "single-thread projection on the machine that ran the timing pilot; rerun on the confirmatory machine",
    }
    out = root / "research_data" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    save_json_atomic(verdict, out / "timing_projection.json")
    print(json.dumps({k: v for k, v in verdict.items() if k != "fits_sec"}, indent=2))


if __name__ == "__main__":
    main()
