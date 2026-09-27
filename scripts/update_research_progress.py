#!/usr/bin/env python3
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "MORE_HD_RESEARCH_READINESS_GATES (1).md"
README = ROOT / "README.md"
HISTORY = ROOT / "research_progress_history.json"

P_START = "<!-- RESEARCH_PROGRESS_START -->"
P_END = "<!-- RESEARCH_PROGRESS_END -->"
S_START = "<!-- READINESS_SUMMARY_START -->"
S_END = "<!-- READINESS_SUMMARY_END -->"

WEIGHTS = {
    "CLOSED": 1.0,
    "READY FOR VERIFICATION": 0.75,
    "IN PROGRESS": 0.5,
    "PILOT ONLY": 0.25,
    "BLOCKED": 0.0,
}
STATUS_ORDER = [
    "READY FOR VERIFICATION",
    "IN PROGRESS",
    "PILOT ONLY",
    "CLOSED",
    "BLOCKED",
]
GROUPS = {
    "G0": "Validitas data & evaluasi",
    "G1": "Optimasi & desain eksperimen",
    "G2": "Preprocessing & definisi numerik",
    "G3": "Artefak & spreadsheet",
    "G4": "Reproducibility & verifikasi sirkuit",
    "D": "Ketidakkonsistenan dokumen (BAB 1, FRD-09, dll.)",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def parse_gate():
    found = {}
    pattern = re.compile(r"^(G[0-4]-\d+|D-\d+)$")
    for line in GATE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [x.strip().replace("**", "").replace("`", "") for x in line.split("|")[1:-1]]
        if len(cells) < 2 or not pattern.match(cells[0]) or cells[0] in found:
            continue
        upper = cells[1].upper()
        status = next((s for s in STATUS_ORDER if s in upper), None)
        if status:
            found[cells[0]] = status
    if not found:
        raise SystemExit("No readiness items found.")
    return found


def stats(items):
    by_group = {}
    for group in GROUPS:
        subset = {k: v for k, v in items.items() if ("D" if k.startswith("D-") else k.split("-")[0]) == group}
        counts = Counter(subset.values())
        score = sum(WEIGHTS[v] for v in subset.values())
        by_group[group] = {
            "items": len(subset),
            "counts": counts,
            "percent": round(100 * score / len(subset), 1) if subset else 0.0,
        }
    overall = round(100 * sum(WEIGHTS[v] for v in items.values()) / len(items), 1)
    return overall, by_group


def load_history():
    return json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else []


def save_history(data):
    HISTORY.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_history(history, progress, item_count):
    if os.getenv("PROGRESS_RECORD_HISTORY", "").lower() != "true":
        return history
    sha = os.getenv("PROGRESS_COMMIT_SHA") or git("rev-parse", "HEAD")
    if any(x["commit"] == sha for x in history):
        return history
    date = os.getenv("PROGRESS_DATE") or git("show", "-s", "--format=%cs", sha)
    summary = os.getenv("PROGRESS_SUMMARY") or git("show", "-s", "--format=%s", sha)
    previous = float(history[-1]["progress"]) if history else None
    history.append({
        "date": date,
        "commit": sha,
        "progress": progress,
        "delta_pp": None if previous is None else round(progress - previous, 1),
        "items": item_count,
        "summary": summary,
    })
    save_history(history)
    return history


def delta_text(value):
    if value is None:
        return "—"
    value = float(value)
    return f"{'+' if value > 0 else ''}{value:.1f} pp"


def escape_md(text):
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def progress_block(progress, item_count, history):
    repo = os.getenv("GITHUB_REPOSITORY", "rexxar280903/MORE-HD-COMPLEX")
    filled = max(0, min(20, round(progress / 5)))
    bar = "█" * filled + "░" * (20 - filled)
    rows = []
    visible = history[-15:]
    for i, entry in enumerate(visible):
        sha = entry["commit"]
        p = f"{float(entry['progress']):.1f}%"
        d = delta_text(entry.get("delta_pp"))
        if i == len(visible) - 1:
            p, d = f"**{p}**", f"**{d}**"
        rows.append(
            f"| {escape_md(entry['date'])} | [`{sha[:7]}`](https://github.com/{repo}/commit/{sha}) | "
            f"{p} | {d} | {escape_md(entry.get('summary', ''))} |"
        )
    trend = " → ".join(f"{float(x['progress']):.1f}%" for x in history[-10:])
    return "\n".join([
        P_START,
        "## Research Progress",
        "",
        f"![Research Progress](https://img.shields.io/badge/Research%20Progress-{progress:.1f}%25-blue)",
        "",
        f"**Current research readiness: {progress:.1f}%**",
        "",
        f"`{bar} {progress:.1f}%`",
        "",
        f"Progress dihitung otomatis dari **{item_count} item Research Readiness Gate** dengan bobot:",
        "`CLOSED = 100%`, `READY FOR VERIFICATION = 75%`, `IN PROGRESS = 50%`, `PILOT ONLY = 25%`, dan `BLOCKED = 0%`.",
        "",
        "### Progress History",
        "",
        "| Tanggal | Commit acuan | Progress | Perubahan | Ringkasan |",
        "|---|---|---:|---:|---|",
        *rows,
        "",
        f"**Trend terbaru:** `{trend}`",
        "",
        "Riwayat lengkap tersimpan di [`research_progress_history.json`](research_progress_history.json).",
        "",
        "> Bagian ini dikelola otomatis oleh GitHub Actions. Nilai dapat turun bila scope/kriteria penelitian bertambah; tracker tidak memaksa progress selalu meningkat.",
        P_END,
    ])


def summary_block(progress, item_count, grouped, date):
    total = Counter()
    lines = [
        S_START,
        f"### Ringkasan Kesiapan (Readiness Gate) — per {date}",
        "",
        "| Gate | Cakupan | Item | Closed | Ready for Verification | In Progress | Pilot Only | Blocked | Kesiapan* |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, scope in GROUPS.items():
        s, c = grouped[group], grouped[group]["counts"]
        total.update(c)
        lines.append(
            f"| {group} | {scope} | {s['items']} | {c.get('CLOSED',0)} | "
            f"{c.get('READY FOR VERIFICATION',0)} | {c.get('IN PROGRESS',0)} | "
            f"{c.get('PILOT ONLY',0)} | {c.get('BLOCKED',0)} | **{s['percent']:.1f}%** |"
        )
    lines += [
        f"| **Total** |  | **{item_count}** | **{total.get('CLOSED',0)}** | "
        f"**{total.get('READY FOR VERIFICATION',0)}** | **{total.get('IN PROGRESS',0)}** | "
        f"**{total.get('PILOT ONLY',0)}** | **{total.get('BLOCKED',0)}** | **{progress:.1f}%** |",
        "",
        "\\* Kesiapan dihitung sebagai rata-rata bobot per item: `CLOSED=100%`, "
        "`READY FOR VERIFICATION=75%`, `IN PROGRESS=50%`, `PILOT ONLY=25%`, dan `BLOCKED=0%`.",
        S_END,
    ]
    return "\n".join(lines)


def replace(text, start, end, new):
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"README markers missing: {start} ... {end}")
    return pattern.sub(new, text, count=1)


items = parse_gate()
progress, grouped = stats(items)
history = record_history(load_history(), progress, len(items))
if not history:
    raise SystemExit("Progress history is empty.")
date = history[-1]["date"]

readme = README.read_text(encoding="utf-8")
readme = replace(readme, P_START, P_END, progress_block(progress, len(items), history))
readme = replace(readme, S_START, S_END, summary_block(progress, len(items), grouped, date))
README.write_text(readme, encoding="utf-8")
print(f"Research progress updated: {progress:.1f}% across {len(items)} items.")
