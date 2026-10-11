"""Baris tabel paper MADBase (angka dari tables.json, dibuat oleh analyze_madbase.py)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "tables.json", encoding="utf-8"))
FEAT = {"PCA": "PCA", "HU": "Hu", "ZERNIKE": "Zernike"}


def num(x, d=1):
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−").replace(".", ",")


def signed(x, d=1):
    s = num(x, d)
    return s if s.startswith("−") or float(f"{x:.{d}f}") == 0 else "+" + s


def ci(lo, hi):
    return f"[{num(lo)}; {num(hi)}]"


def star(lo, hi):
    return "*" if (lo > 0 or hi < 0) else ""


def primary_rows():
    rows, last = [], None
    for f, k, am, asd, dm, dsd, dl, lo, hi, g, p in D["T2"]:
        rows.append([FEAT[f] if f != last else "", str(k), f"{num(am)} ± {num(asd)}", f"{num(dm)} ± {num(dsd)}",
                     signed(dl) + star(lo, hi), ci(lo, hi), num(g, 2)])
        last = f
    return rows


def ablation_rows():
    rows, last = [], None
    for r in D["T3"]:
        f, k, vals = r[0], r[1], r[2:]
        cells = []
        for i in range(4):
            m, lo, hi, _p = vals[4 * i: 4 * i + 4]
            cells.append(f"{signed(m)}{star(lo, hi)}\n{ci(lo, hi)}")
        rows.append([FEAT[f] if f != last else "", str(k)] + cells)
        last = f
    return rows


def more_rows():
    rows, last = [], None
    for f, k, mm, msd, am, alo, ahi, dm, dlo, dhi in D["T4"]:
        rows.append([FEAT[f] if f != last else "", str(k), f"{num(mm)} ± {num(msd)}",
                     f"{signed(am)}{star(alo, ahi)} {ci(alo, ahi)}", f"{signed(dm)}{star(dlo, dhi)} {ci(dlo, dhi)}"])
        last = f
    return rows


def baseline_rows(mean_label="Rata-rata"):
    data = {}
    for f, k, ch, nc, lr, m, a, d in D["T5"]:
        data.setdefault(f, {})[k] = (ch, nc, lr, m, a, d)
    rows = []
    for f in ("PCA", "HU", "ZERNIKE"):
        first = True
        for k in (3, 10):
            rows.append([FEAT[f] if first else "", f"K = {k}"] + [num(x) for x in data[f][k]])
            first = False
        means = [sum(data[f][k][i] for k in range(3, 11)) / 8 for i in range(6)]
        rows.append(["", mean_label] + [num(x) for x in means])
    return rows


def mechanism_rows():
    m = D.get("T6")
    if not m:
        return []
    trainable = {"A": "30", "C": "30", "D": "60", "B": "60"}
    rows = []
    for code in ("A", "C", "D", "B"):
        v = m[code]
        p9 = num(v["probe9"]) if code in ("C", "D") else "—"
        rows.append([code, trainable[code], f"{v['rank']:.0f}", num(v["eff"]), num(v["probe"]), p9,
                     num(v["acc"]), num(v["gap"]), num(v["sup"], 0) + "%"])
    return rows


def jalurb_rows():
    rows, last = [], None
    for f, k, aB, dB, daA, daB, lo, hi in D["TJ"]:
        rows.append([FEAT[f] if f != last else "", str(k), num(aB), num(dB), signed(daA), f"{signed(daB)}{star(lo, hi)} {ci(lo, hi)}"])
        last = f
    return rows


def cross_rows():
    x = D["TX"]
    mn, ad = x["mnist"], x["madbase"]

    def cnt(s):
        return f"{s['n_pos']}/{s['n']}"

    rows = [
        ["Akurasi rata-rata A / D (%)", f"{num(mn['acc']['A'])} / {num(mn['acc']['D'])}", f"{num(ad['acc']['A'])} / {num(ad['acc']['D'])}"],
        ["D − A akurasi (pp); sel positif", f"{signed(mn['DA']['mean'])}; {cnt(mn['DA'])}", f"{signed(ad['DA']['mean'])}; {cnt(ad['DA'])}"],
        ["D − A macro-F1 (pp); sel positif", f"{signed(mn['DA_f1']['mean'])}; {cnt(mn['DA_f1'])}", f"{signed(ad['DA_f1']['mean'])}; {cnt(ad['DA_f1'])}"],
        ["SK 95% D − A di atas / di bawah nol", f"{mn['DA']['ci_pos']} / {mn['DA']['ci_neg']}", f"{ad['DA']['ci_pos']} / {ad['DA']['ci_neg']}"],
        ["McNemar signifikan (dari 120)", str(mn["mcnemar_sig"]), str(ad["mcnemar_sig"])],
    ]
    for lab in ("B−A", "C−A", "D−B", "D−C"):
        rows.append([f"Ablasi {lab.replace('−', ' − ')} (pp); sel positif", f"{signed(mn['ablation'][lab]['mean'])}; {cnt(mn['ablation'][lab])}",
                     f"{signed(ad['ablation'][lab]['mean'])}; {cnt(ad['ablation'][lab])}"])
    rows += [
        ["D − M (pp); sel positif", f"{signed(mn['D_minus_M']['mean'])}; {cnt(mn['D_minus_M'])}", f"{signed(ad['D_minus_M']['mean'])}; {cnt(ad['D_minus_M'])}"],
        ["Akurasi rata-rata M / NC / LR (%)", f"{num(mn['acc']['M'])} / {num(mn['NC'])} / {num(mn['LR'])}",
         f"{num(ad['acc']['M'])} / {num(ad['NC'])} / {num(ad['LR'])}"],
        ["Porsi galat 6↔9 pada K = 10, model A–D", num(mn["six_nine_share_ABCD"] * 100) + "%", num(ad["six_nine_share_ABCD"] * 100) + "%"],
    ]
    return rows
