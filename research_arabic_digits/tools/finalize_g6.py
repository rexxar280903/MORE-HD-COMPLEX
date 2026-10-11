"""Audit penutupan G6 dan manifest reproduksi studi MADBase (hanya membaca; tidak mengubah run/workbook).

Keluaran:
  research_arabic_digits/research_data/reports/g6_final_audit.json
  research_arabic_digits/REPRODUCIBILITY_MANIFEST.json
  research_arabic_digits/REPRODUCIBILITY_MANIFEST.md
Usage: python research_arabic_digits/tools/finalize_g6.py
Hanya pustaka standar Python; aman dijalankan di Windows maupun Linux.
"""
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent
RD = STUDY / "research_data"
RUNS = REPO / "runs" / "arabic_digits"
WIB = timezone(timedelta(hours=7))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status_counts():
    main, jb, pilot = Counter(), Counter(), Counter()
    attempts = Counter()
    for st in list(RUNS.glob("*/logs/run_status.json")) + list(RUNS.glob("*/*/logs/run_status.json")):
        d = st.parents[1]
        cfgp = d / "config.json"
        if not cfgp.exists():
            continue
        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        s = json.loads(st.read_text(encoding="utf-8"))
        if cfg.get("run_mode") != "CONFIRMATORY":
            pilot[s.get("status")] += 1
            continue
        if "jalurB" in d.name or cfg.get("track") == "JALUR_B":
            jb[s.get("status")] += 1
        else:
            main[(cfg.get("track"), s.get("status"))] += 1
            attempts[s.get("attempt")] += 1
    return main, jb, pilot, attempts


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def main():
    main_c, jb_c, pilot_c, attempts = status_counts()
    adm = json.loads((RD / "reports" / "final_admission.json").read_text(encoding="utf-8"))
    budget = json.loads((RD / "pilot" / "convergence_budget_decision.json").read_text(encoding="utf-8"))
    base_rows = list(csv.DictReader((STUDY / "paper" / "build" / "data" / "primary__15_Classical_Baselines.csv").open(encoding="utf-8")))
    base_status = Counter(r["status"] for r in base_rows if r["baseline"] in ("NC", "LR"))
    chance_rows = sum(1 for r in base_rows if r["baseline"] not in ("NC", "LR"))
    planned = {"PRIMARY": 240, "ABLATION": 90, "MORE_REFERENCE": 120}
    by_track = {t: {s: n for (tt, s), n in main_c.items() if tt == t} for t in planned}
    checks = {
        "main_runs_completed": sum(v.get("COMPLETED", 0) for v in by_track.values()),
        "main_runs_planned": sum(planned.values()),
        "main_by_track": by_track,
        "main_attempts": dict(attempts),
        "jalurB": dict(jb_c),
        "classical_fits": dict(base_status),
        "chance_rows_analytic": chance_rows,
        "pilot_and_smoke_runs_excluded": dict(pilot_c),
        "final_admission": adm,
        "budget_decision": {k: budget.get(k) for k in ("status", "dataset", "cap", "n_runs")} |
                           {"clustering": budget["clustering"]["final_max_nfev"], "supervised": budget["supervised"]["final_max_nfev"]},
    }
    ok = (checks["main_runs_completed"] == 450 and jb_c.get("COMPLETED", 0) == 90 and base_status.get("COMPLETED", 0) == 240
          and adm.get("accepted") is True and not adm.get("errors") and set(attempts) == {1})
    artifacts = {
        "dataset_manifest": ["research_data/manifests/raw_files.sha256", "research_data/manifests/dataset_inventory.json",
                             "research_data/manifests/label_mapping.json", "research_data/manifests/duplicate_audit.csv",
                             "research_data/manifests/data_audit_report.md", "DATASET_CARD.md"],
        "split_manifests": sorted(str(p.relative_to(STUDY)).replace("\\", "/") for p in (STUDY / "splits").glob("*.json")),
        "protocol": ["PROTOCOL_FREEZE.json", "RESEARCH_PROTOCOL.md", "STATISTICAL_ANALYSIS_PLAN.md", "DEVIATION_LOG.md"],
        "results": ["RESULTS.md"],
        "run_plan": ["research_data/reports/confirmatory_run_plan.json", "research_data/reports/queue8_plan.json"],
        "budget_decision": ["research_data/pilot/convergence_budget_decision.json", "research_data/pilot/convergence_check.json"],
        "admission_report": ["research_data/reports/final_admission.json", "research_data/reports/full_readiness_audit.json"],
        "consolidated_workbooks": sorted(str(p.relative_to(STUDY)).replace("\\", "/") for p in (RD / "consolidated").glob("*.xlsx")),
        "analysis_tables": sorted(str(p.relative_to(STUDY)).replace("\\", "/") for p in (RD / "analysis").glob("*")),
        "environment": ["requirements-lock-py313.txt", "research_data/confirmatory_machine/environment_pretraining.json",
                        "research_data/confirmatory_machine/hardware_pretraining.json"],
        "paper_numbers": ["paper/build/numbers.json", "paper/build/tables.json", "paper/build/runs_all.csv"],
    }
    manifest = {}
    for group, rels in artifacts.items():
        manifest[group] = []
        for rel in rels:
            p = STUDY / rel
            manifest[group].append({"path": "research_arabic_digits/" + rel, "exists": p.exists(),
                                    "bytes": p.stat().st_size if p.exists() else None,
                                    "sha256": sha256(p) if p.exists() else None})
    code = {"run_git_commit": "188bab6a068afcb02d76b18bd23cf5449943ed2d",
            "head_now": git("rev-parse", "HEAD"),
            "core_engine_diff_vs_commit": git("diff", "--stat", "188bab6a068afcb02d76b18bd23cf5449943ed2d", "--", "core") or "NONE",
            "untracked_or_modified_now": git("status", "--short").splitlines()}
    now = datetime.now(WIB).isoformat(timespec="seconds")
    audit = {"generated_at": now, "dataset": "MADBASE-AUC-V1", "g6_status": "CLOSED" if ok else "OPEN", "checks": checks,
             "code": code}
    (RD / "reports" / "g6_final_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (STUDY / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps({"generated_at": now, "code": code, "artifacts": manifest},
                                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Manifest Reproduksi — Studi Digit Arab (MADBase)", "",
          f"Dibuat otomatis oleh `research_arabic_digits/tools/finalize_g6.py` pada {now}. Status G6: "
          f"**{audit['g6_status']}**.", "",
          "## Kode", "",
          f"- Commit yang direkam semua run: `{code['run_git_commit']}`",
          f"- HEAD saat manifest dibuat: `{code['head_now']}`",
          f"- Diff engine `core/` terhadap commit run: {code['core_engine_diff_vs_commit']}",
          "- File belum di-commit/termodifikasi saat manifest dibuat: lihat `REPRODUCIBILITY_MANIFEST.json` "
          "(`code.untracked_or_modified_now`) dan `DEVIATION_LOG.md` DEV-02.", "",
          "## Ringkasan run", "",
          f"- Run utama COMPLETED: {checks['main_runs_completed']}/{checks['main_runs_planned']} "
          f"(attempt: {checks['main_attempts']})",
          f"- Jalur B: {checks['jalurB']}; fit baseline klasik: {checks['classical_fits']} "
          f"(+{checks['chance_rows_analytic']} baris chance analitik)",
          f"- Budget: clustering {checks['budget_decision']['clustering']}, supervised "
          f"{checks['budget_decision']['supervised']} ({checks['budget_decision']['status']})", "",
          "## Artefak dan SHA-256", ""]
    for group, rows in manifest.items():
        md += [f"### {group}", "", "| Path | Bytes | SHA-256 |", "|---|---:|---|"]
        md += [f"| `{r['path']}` | {r['bytes'] if r['bytes'] is not None else '—'} | `{r['sha256'] or 'MISSING'}` |" for r in rows]
        md.append("")
    md += ["Data mentah MADBase tidak termasuk paket ini (kebijakan `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`); "
           "verifikasi unduhan resmi dengan `research_data/manifests/raw_files.sha256`.", ""]
    (STUDY / "REPRODUCIBILITY_MANIFEST.md").write_text("\n".join(md), encoding="utf-8")
    print("G6", audit["g6_status"], json.dumps(checks, ensure_ascii=False)[:900])
    print("core diff:", code["core_engine_diff_vs_commit"][:300])


if __name__ == "__main__":
    main()
