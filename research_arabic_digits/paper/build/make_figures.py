"""Gambar paper MADBase (label bahasa Indonesia) dari hasil konfirmatori digit Arab.

Input : research_arabic_digits/paper/build/{runs_all.csv, numbers.json, cross_dataset_cells.csv, confusion_pairs_k10.csv,
        mechanism/runs_mechanism.csv} dan data/analysis/paired_primary_D_minus_A.csv
Output: research_arabic_digits/paper/figures/id/fig*.png (300 dpi, lebar teks 5 in template VJCS)
Studi MNIST tidak diubah; angka MNIST hanya dipakai di Gambar 8 (perbandingan deskriptif).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AN = HERE / "data" / "analysis"
OUT = HERE.parent / "figures" / "id"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
GLYPH_FONT = {"family": "DejaVu Sans"}
GLYPH = "٠١٢٣٤٥٦٧٨٩"

COL = {"A": "#0072B2", "D": "#D55E00", "B": "#009E73", "C": "#CC79A7", "M": "#7F7F7F"}
FEATS = ["PCA", "HU", "ZERNIKE"]
FEAT_LAB = {"PCA": "PCA", "HU": "Momen Hu", "ZERNIKE": "Momen Zernike"}
NAMES = {"A": "A: MORE-HD (30 RY)", "D": "D: MORE-HD-C (30 RY + 30 RZ)", "B": "B: MORE-HD-60P (60 RY)",
         "C": "C: FixedRZ (30 RY + 30 RZ beku)", "M": "M: MORE-REPRO (91)"}
T = dict(acc="Akurasi uji", k="Jumlah kelas $K$", chance="Peluang $1/K$",
         delta="$\\Delta$ akurasi, D $-$ A (poin persentase)", yodd="Fraksi norma Y-ganjil",
         sep="Rasio separasi minimum", data="data", readout="readout", enc="Enkoding",
         layer="Lapisan variasional (diulang 3 kali)", ent="Blok entangling", meas="15 ekspektasi\nPauli",
         rz_note="RZ: hanya MORE-HD-C", rho="$\\rho$ Spearman")
PIPE = ["Digit Arab MADBase\nkelas $0..K-1$\nsplit writer-disjoint", "Fitur\nPCA / Hu / Zernike\n8 kanal dalam $[0,\\pi]$",
        "Fase 1: clustering\n5 sampel/kelas\nsemua $\\binom{5K}{2}$ pasangan",
        "Label kuantum\nnormalisasi-median-\nnormalisasi", "Fase 2: supervised\nloss kosinus ke label\n1.000 sampel/kelas",
        "Prediksi\nlabel terdekat\n(uji resmi, 20 penulis)"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def dec(x, d=2):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


# ---------------------------------------------------------------- Gambar 1 sirkuit (identik dengan studi MNIST)
def fig_circuit():
    fig, ax = plt.subplots(figsize=(5.0, 2.55))
    ax.set_xlim(-1.6, 17.6)
    ax.set_ylim(-0.9, 10.2)
    ax.axis("off")
    ax.invert_yaxis()
    ys = list(range(10))
    for q in ys:
        ax.plot([0, 16.3], [q, q], color="black", lw=0.5, zorder=0)
        lab = f"$q_{q}$" if q < 8 else f"$r_{q - 8}$"
        ax.text(-0.35, q, lab, ha="right", va="center", fontsize=6.5)
    ax.text(-1.25, 3.5, T["data"], rotation=90, ha="center", va="center", fontsize=6.5, color="#444")
    ax.text(-1.25, 8.5, T["readout"], rotation=90, ha="center", va="center", fontsize=6.5, color="#444")

    def box(x, y, txt, fc="white", ec="black", w=0.95):
        ax.add_patch(Rectangle((x - w / 2, y - 0.36), w, 0.72, fc=fc, ec=ec, lw=0.5, zorder=2))
        ax.text(x, y, txt, ha="center", va="center", fontsize=5.2, zorder=3)

    for q in range(8):
        box(0.9, q, "$R_Y(x_{%d})$" % q, fc="#E8E8E8", w=1.15)
    ax.add_patch(FancyBboxPatch((1.85, -0.75), 13.25, 10.4, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="none", ec="#555", lw=0.6, ls="--", zorder=1))
    for q in ys:
        box(2.75, q, "$R_Y(\\theta)$", fc="#DCEBF7")
        box(3.95, q, "$R_Z(\\phi)$", fc="#FBE3D6", ec=COL["D"])
    pairs = [(i, i + 1) for i in range(7)] + [(7, 0), (7, 8), (7, 9), (8, 9)]
    x0 = 5.0
    for k, (c, tg) in enumerate(pairs):
        x = x0 + k * 0.9
        ax.plot([x, x], [c, tg], color="black", lw=0.6, zorder=1)
        ax.add_patch(Circle((x, c), 0.13, color="black", zorder=3))
        ax.add_patch(Circle((x, tg), 0.27, fc="white", ec="black", lw=0.6, zorder=3))
        ax.plot([x, x], [tg - 0.27, tg + 0.27], color="black", lw=0.6, zorder=4)
        ax.plot([x - 0.27, x + 0.27], [tg, tg], color="black", lw=0.6, zorder=4)
    for q in (8, 9):
        ax.add_patch(Rectangle((15.6, q - 0.36), 0.75, 0.72, fc="#F2F2F2", ec="black", lw=0.5, zorder=2))
        ax.add_patch(matplotlib.patches.Arc((15.97, q + 0.18), 0.5, 0.45, theta1=0, theta2=180, lw=0.5, zorder=3))
        ax.plot([15.97, 16.17], [q + 0.18, q - 0.2], color="black", lw=0.5, zorder=3)
    ax.text(16.5, 8.5, T["meas"], ha="left", va="center", fontsize=5.8)
    ax.text(0.9, -0.75, T["enc"], ha="center", va="bottom", fontsize=6)
    ax.text(8.5, -0.85, T["layer"], ha="center", va="bottom", fontsize=6)
    ax.text(9.5, 9.85, T["ent"], ha="center", va="top", fontsize=5.8, color="#333")
    ax.text(3.95, 9.85, T["rz_note"], ha="center", va="top", fontsize=5.8, color=COL["D"])
    save(fig, "fig1_circuit.png")


# ---------------------------------------------------------------- Gambar 2 alur
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(5.0, 1.75))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    fills = ["#F2F2F2", "#F2F2F2", "#DCEBF7", "#FBE3D6", "#DCEBF7", "#F2F2F2"]
    pos = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
    w, h = 0.78, 0.8
    for i, (txt, fc) in enumerate(zip(PIPE, fills)):
        cx, cy = pos[i][0] + 0.5, pos[i][1] + 0.5
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                    fc=fc, ec="#333", lw=0.5))
        ax.text(cx - w / 2 + 0.04, cy + h / 2 - 0.05, f"({i + 1})", ha="left", va="top", fontsize=5.5, color="#555")
        ax.text(cx, cy - 0.03, txt, ha="center", va="center", fontsize=6.0, linespacing=1.2,
                fontfamily=["Liberation Serif", "DejaVu Sans"])
    arrows = [((0.89, 1.5), (1.11, 1.5)), ((1.89, 1.5), (2.11, 1.5)), ((2.5, 1.1), (2.5, 0.9)),
              ((2.11, 0.5), (1.89, 0.5)), ((1.11, 0.5), (0.89, 0.5))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333", mutation_scale=7))
    save(fig, "fig2_pipeline.png")


# ---------------------------------------------------------------- Gambar 3 akurasi vs K (A, D, M)
def fig_accuracy(runs):
    fig, axes = plt.subplots(1, 3, figsize=(5.0, 1.95), sharey=True)
    Ks = np.arange(3, 11)
    for ax, f in zip(axes, FEATS):
        for m in ["A", "D", "M"]:
            g = runs[(runs.m == m) & (runs.feature_method == f)].groupby("K").accuracy
            mu, sd = g.mean().reindex(Ks) * 100, g.std().reindex(Ks) * 100
            ax.plot(Ks, mu, "-o", color=COL[m], ms=2.6, lw=1.0, label=NAMES[m])
            ax.fill_between(Ks, mu - sd, mu + sd, color=COL[m], alpha=0.14, lw=0)
        ax.plot(Ks, 100 / Ks, ":", color="#888", lw=0.8, label=T["chance"])
        ax.set_title(FEAT_LAB[f])
        ax.set_xticks(Ks)
        ax.set_xlabel(T["k"])
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel(T["acc"] + " (%)")
    axes[0].set_ylim(0, 100)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_accuracy_vs_K.png")


# ---------------------------------------------------------------- Gambar 4 forest D−A
def fig_forest(prim):
    p = prim[prim.metric == "accuracy"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(5.0, 2.1), sharey=True)
    Ks = np.arange(3, 11)
    for ax, f in zip(axes, FEATS):
        q = p[p.feature_method == f].set_index("K").reindex(Ks)
        y = np.arange(len(Ks))
        mu, lo, hi = q.mean_delta * 100, q.ci95_low * 100, q.ci95_high * 100
        for yi, k in zip(y, Ks):
            deltas = [float(v) * 100 for v in str(q.loc[k, "deltas"]).split(",")]
            ax.scatter(deltas, [yi] * 5, s=4, color="#999", zorder=1, lw=0)
        excl = (lo > 0) | (hi < 0)
        ax.errorbar(mu, y, xerr=[mu - lo, hi - mu], fmt="none", ecolor=COL["D"], elinewidth=0.9, capsize=1.5, zorder=2)
        ax.scatter(mu, y, s=14, marker="D", color=np.where(excl, COL["D"], "white"), edgecolor=COL["D"], lw=0.8, zorder=3)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"K={k}" for k in Ks])
        ax.set_title(FEAT_LAB[f])
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].invert_yaxis()
    fig.supxlabel(T["delta"], fontsize=8, y=0.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_forest_D_minus_A.png")


# ---------------------------------------------------------------- Gambar 5 ablasi
def fig_ablation(runs):
    fig, axes = plt.subplots(1, 3, figsize=(5.0, 2.0), sharey=True)
    order = ["A", "C", "D", "B"]
    width = 0.2
    for ax, f in zip(axes, FEATS):
        for j, m in enumerate(order):
            for i, k in enumerate([3, 6, 10]):
                v = runs[(runs.m == m) & (runs.feature_method == f) & (runs.K == k)].accuracy * 100
                x = i + (j - 1.5) * width
                ax.bar(x, v.mean(), width * 0.9, color=COL[m], alpha=0.85, label=NAMES[m] if i == 0 else None)
                ax.scatter(np.full(len(v), x), v, s=3, color="black", zorder=3, lw=0)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["K=3", "K=6", "K=10"])
        ax.set_title(FEAT_LAB[f])
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel(T["acc"] + " (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig5_ablation.png")


# ---------------------------------------------------------------- Gambar 6 diagnostik readout
def fig_readout(runs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.0, 2.0), gridspec_kw=dict(width_ratios=[1, 1.35]))
    order = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = runs[runs.m == m].yodd_norm_fraction
        a1.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=3, color=COL[m], alpha=0.6, lw=0)
        a1.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="black", lw=1.0)
    a1.set_xticks(range(4))
    a1.set_xticklabels(order)
    a1.set_ylabel(T["yodd"])
    a1.set_ylim(-0.03, 1.02)
    a1.set_title("(a)")
    a1.grid(axis="y", lw=0.3, alpha=0.5)
    for m in ["A", "D"]:
        q = runs[runs.m == m]
        r = stats.spearmanr(q.min_separation_ratio, q.accuracy).statistic
        a2.scatter(q.min_separation_ratio, q.accuracy * 100, s=5, color=COL[m], alpha=0.65, lw=0,
                   label=f"{m} ({T['rho']} = {dec(r)})")
    a2.set_xscale("symlog", linthresh=0.01)
    a2.set_xlabel(T["sep"])
    a2.set_ylabel(T["acc"] + " (%)")
    a2.set_title("(b)")
    a2.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", markerscale=2.5)
    a2.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig6_readout.png")


# ---------------------------------------------------------------- Gambar 7 analisis mekanistik (eksploratif)
def fig_capacity():
    f = HERE / "mechanism" / "runs_mechanism.csv"
    if not f.exists():
        print("lewati Gambar 7: runs_mechanism.csv belum ada")
        return
    c = pd.read_csv(f)
    order = ["A", "C", "D", "B"]
    nominal = {"A": "30", "C": "30", "D": "60 (52)", "B": "60"}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(5.0, 2.05), gridspec_kw=dict(width_ratios=[1.25, 1, 1.15]))
    x = np.arange(len(order))
    for i, m in enumerate(order):
        q = c[c.model == m]
        pr, ac = q.probe_all15 * 100, q.acc_reproduced * 100
        a1.bar(i - 0.19, pr.mean(), 0.36, color=COL[m], alpha=0.35, edgecolor=COL[m], lw=0.6, hatch="////",
               label="Linear probe (15 dimensi)" if i == 0 else None)
        a1.bar(i + 0.19, ac.mean(), 0.36, color=COL[m], alpha=0.9, label="Label kuantum terdekat" if i == 0 else None)
        a1.errorbar([i - 0.19, i + 0.19], [pr.mean(), ac.mean()], yerr=[pr.sem(), ac.sem()], fmt="none",
                    ecolor="black", elinewidth=0.6, capsize=1.2)
        if m in ("C", "D"):
            a1.scatter([i - 0.19], [q.probe_yeven9.mean() * 100], marker="_", s=60, color="black", lw=1.2,
                       label="Linear probe (9 dimensi Y-genap)" if m == "C" else None, zorder=4)
    a1.set_xticks(x)
    a1.set_xticklabels(order)
    a1.set_ylim(30, 100)
    a1.set_ylabel("Akurasi uji (%)")
    a1.set_title("(a)")
    a1.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.2)
    a1.grid(axis="y", lw=0.3, alpha=0.5)
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = c[c.model == m].jac_eff_rank
        a2.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=4, color=COL[m], alpha=0.7, lw=0)
        a2.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="black", lw=1.0)
    a2.set_xticks(x)
    a2.set_xticklabels([f"{m}\n{nominal[m]}" for m in order], fontsize=6.3)
    a2.set_ylabel("Effective rank Jacobian")
    a2.set_ylim(0, 62)
    a2.set_title("(b)")
    a2.grid(axis="y", lw=0.3, alpha=0.5)
    c["dacc"] = (c.acc_reproduced - c.groupby(["feature", "K"]).acc_reproduced.transform("mean")) * 100
    for m in order:
        q = c[c.model == m]
        a3.scatter(q.jac_eff_rank, q.dacc, s=5, color=COL[m], alpha=0.7, lw=0, label=m)
    rk = c.assign(ra=c.groupby(["feature", "K"]).acc_reproduced.rank(), rj=c.groupby(["feature", "K"]).jac_eff_rank.rank())
    r = stats.spearmanr(rk.rj, rk.ra).statistic
    a3.axhline(0, color="#888", lw=0.5)
    a3.set_xlabel("Effective rank Jacobian")
    a3.set_ylabel("Akurasi dikurangi rata-rata sel (pp)")
    a3.set_title("(c)")
    a3.text(0.03, 0.97, f"$\\rho$ Spearman di dalam sel = {dec(r)}", transform=a3.transAxes, ha="left", va="top", fontsize=6)
    a3.legend(frameon=False, fontsize=5.8, loc="lower right", markerscale=2, handletextpad=0.1, ncol=2, columnspacing=0.6)
    a3.grid(lw=0.3, alpha=0.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig7_capacity.png")


# ---------------------------------------------------------------- Gambar 8 pasangan kelas tertukar (eksploratif)
def fig_confusion():
    p = pd.read_csv(HERE / "confusion_pairs_k10.csv")
    fig, axes = plt.subplots(1, 3, figsize=(5.0, 1.9), sharex=True)
    for ax, f in zip(axes, FEATS):
        q = p[p.feature_method == f].nsmallest(6, "rank").iloc[::-1]
        labels = []
        for pr in q.pair:
            a, b = (int(s) for s in pr.split("↔"))
            labels.append(f"{a}↔{b}  (\u200e{GLYPH[a]}\u200e ↔ \u200e{GLYPH[b]}\u200e)")
        hl = [pr in ("7↔8", "0↔5") for pr in q.pair]
        ax.barh(range(len(q)), q.share * 100, color=[COL["D"] if h else "#9AA4AE" for h in hl], height=0.65)
        ax.axvline(100 / 45, color="black", lw=0.6, ls=":")
        ax.set_yticks(range(len(q)))
        ax.set_yticklabels(labels, fontdict={"family": "DejaVu Sans", "size": 6.3})
        ax.set_title(FEAT_LAB[f])
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.supxlabel("Porsi dari seluruh galat model A–D pada $K$ = 10 (%)", fontsize=7.5, y=0.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig8_confusion_pairs.png")


# ---------------------------------------------------------------- Gambar 9 perbandingan deskriptif dengan MNIST
def fig_cross(num):
    cells = pd.read_csv(HERE / "cross_dataset_cells.csv")
    cr = num["cross"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.0, 2.1), gridspec_kw=dict(width_ratios=[1, 1.25]))
    mk = {"PCA": "o", "HU": "s", "ZERNIKE": "^"}
    for f in FEATS:
        q = cells[cells.feature == f]
        a1.scatter(q.mnist, q.madbase, s=12, marker=mk[f], color="#555", alpha=0.8, lw=0, label=FEAT_LAB[f])
    lim = [-12, 28]
    a1.plot(lim, lim, color="#999", lw=0.5, ls="--")
    a1.axhline(0, color="black", lw=0.4)
    a1.axvline(0, color="black", lw=0.4)
    a1.set_xlim(lim)
    a1.set_ylim(lim)
    a1.set_xlabel("D − A pada MNIST (pp)")
    a1.set_ylabel("D − A pada MADBase (pp)")
    a1.set_title("(a)")
    a1.legend(frameon=False, fontsize=5.8, loc="lower right", handletextpad=0.2)
    labs = ["D − A", "B − A", "C − A", "D − B", "D − C"]
    mvals = [cr["mnist"]["DA"]["mean"]] + [cr["mnist"]["ablation"][k]["mean"] for k in ("B−A", "C−A", "D−B", "D−C")]
    avals = [cr["madbase"]["DA"]["mean"]] + [cr["madbase"]["ablation"][k]["mean"] for k in ("B−A", "C−A", "D−B", "D−C")]
    y = np.arange(len(labs))
    a2.barh(y - 0.19, mvals, 0.36, color="#B8C4CE", label="MNIST")
    a2.barh(y + 0.19, avals, 0.36, color=COL["D"], label="MADBase")
    a2.axvline(0, color="black", lw=0.5)
    a2.set_yticks(y)
    a2.set_yticklabels(labs)
    a2.invert_yaxis()
    a2.set_xlabel("Rata-rata selisih akurasi (pp)")
    a2.set_title("(b)")
    a2.legend(frameon=False, fontsize=6, loc="lower right")
    a2.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig9_cross_dataset.png")


if __name__ == "__main__":
    runs = pd.read_csv(HERE / "runs_all.csv")
    prim = pd.read_csv(AN / "paired_primary_D_minus_A.csv")
    num = json.loads((HERE / "numbers.json").read_text())
    fig_circuit()
    fig_pipeline()
    fig_accuracy(runs)
    fig_forest(prim)
    fig_ablation(runs)
    fig_readout(runs)
    fig_capacity()
    fig_confusion()
    fig_cross(num)
    print("figures written to", OUT)
