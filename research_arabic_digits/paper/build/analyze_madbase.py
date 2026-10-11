"""Analisis final studi MADBase (digit Arab) untuk RESULTS.md, paper, dan dashboard.

Semua angka dihitung ulang dari workbook konsolidasi (diekstrak oleh extract_workbooks.py) dan dicek silang
terhadap tabel resmi analysis/run_analysis.py (research_arabic_digits/research_data/analysis/*.csv).
Perbandingan dengan MNIST bersifat DESKRIPTIF (SAP MADBase §8): angka MNIST diambil dari
research_data/data_from_laptop/analysis/*.csv dan paper/build/runs_all.csv dengan rumus yang sama.

Keluaran (folder ini):
  runs_all.csv            satu baris per run konfirmatori (A/B/C/D/M)
  tables.json             baris tabel paper (T2 primer, T3 ablasi, T4 MORE-REPRO, T5 baseline, TX lintas dataset, TJ Jalur B)
  numbers.json            semua angka yang dikutip di teks paper dan RESULTS.md
  cross_dataset_cells.csv D−A per sel untuk MNIST dan MADBase
  jalurB_summary.csv      ringkasan analisis sensitivitas Jalur B
  confusion_pairs_k10.csv pasangan kelas yang paling sering tertukar pada K = 10 (eksploratif)

Usage: python research_arabic_digits/paper/build/analyze_madbase.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE / "data"
AN = DATA / "analysis"
MNIST_AN = ROOT / "research_data" / "data_from_laptop" / "analysis"
MNIST_RUNS = ROOT / "paper" / "build" / "runs_all.csv"
MECH = HERE / "mechanism"

CODE = {"MORE-HD": "A", "MORE-HD-C": "D", "MORE-HD-60P": "B", "MORE-HD-C-FixedRZ": "C", "MORE-REPRO": "M"}
FEATS = ["PCA", "HU", "ZERNIKE"]
KS = list(range(3, 11))
SEEDS = [101, 202, 303, 404, 505]
T975 = stats.t.ppf(0.975, 4)


def paired(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if sd > 0 else 0.0
    g = (1 - 3 / (4 * n - 5)) * m / sd if sd > 0 else 0.0
    obs = abs(m)
    cnt = sum(abs(np.mean(d * np.array(s))) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=n))
    return {"mean": m, "sd": sd, "lo": m - half, "hi": m + half, "g": g, "p": cnt / 2 ** n}


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(ps) - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj


def load_runs():
    frames = []
    for t in ("primary", "ablation", "more"):
        r = pd.read_csv(DATA / f"{t}__01_Run_Summary.csv")
        uid = [c for c in r.columns if c.endswith("run_uid")][0]
        r = r.rename(columns={uid: "run_uid"})
        r["track"] = t.upper()
        frames.append(r)
    r = pd.concat(frames, ignore_index=True)
    r["m"] = r.architecture.map(CODE)
    return r


def mnist_runs():
    r = pd.read_csv(MNIST_RUNS)
    r["m"] = r.architecture.map(CODE)
    return r


def cell_contrast(runs, a, b, metric="accuracy", feats=FEATS, ks=KS, scale=100.0):
    """b − a per (feature, K), paired by seed."""
    out = []
    for f in feats:
        for k in ks:
            x = runs[(runs.m == a) & (runs.feature_method == f) & (runs.K == k)].set_index("seed")[metric]
            y = runs[(runs.m == b) & (runs.feature_method == f) & (runs.K == k)].set_index("seed")[metric]
            if len(x) == 0 or len(y) == 0:
                continue
            d = (y.reindex(SEEDS) - x.reindex(SEEDS)).values * scale
            s = paired(d)
            out.append({"feature": f, "K": k, "a": a, "b": b, "metric": metric, **s, "deltas": d.tolist(),
                        "a_mean": x.mean() * scale, "a_sd": x.std() * scale, "b_mean": y.mean() * scale, "b_sd": y.std() * scale})
    return pd.DataFrame(out)


def excl(df):
    return int((df.lo > 0).sum()), int((df.hi < 0).sum())


def summarize_contrast(df):
    plus, minus = excl(df)
    return {"mean": float(df["mean"].mean()), "n_pos": int((df["mean"] > 0).sum()), "n": int(len(df)),
            "ci_pos": plus, "ci_neg": minus, "min": float(df["mean"].min()), "max": float(df["mean"].max()),
            "n_g_gt_08": int((df.g > 0.8).sum()), "n_g_lt_m08": int((df.g < -0.8).sum())}


def main():
    runs = load_runs()
    mn = mnist_runs()
    num: dict = {}
    tables: dict = {}

    # ------------------------------------------------------------------ admission / data quality
    adm = json.loads((DATA / "reports" / "final_admission.json").read_text())
    num["admission"] = adm
    num["runs_by_model"] = runs.groupby("m").size().to_dict()
    num["quality"] = {
        "status_completed": int((runs.status == "COMPLETED").sum()),
        "attempt_max": int(runs.attempt.max()),
        "duplicate_uid": int(runs.run_uid.duplicated().sum()),
        "degenerate_quantum_label": int(runs.degenerate_quantum_label.astype(int).sum()),
        "degenerate_test_outputs": int(runs.n_degenerate_test_outputs.sum()),
        "backend_max_abs_diff": float(runs.backend_max_abs_diff.max()),
        "cluster_nfev": sorted(runs.cluster_nfev.unique().tolist()),
        "supervised_nfev": sorted(runs.supervised_nfev.unique().tolist()),
        "git_commit": sorted(runs.git_commit.unique().tolist()),
        "started": runs.started_at.min(), "finished": runs.finished_at.max(),
        "active_dims": {m: sorted(g.active_dimensions.unique().tolist()) for m, g in runs.groupby("m")},
        "paired_input_check": {t: runs[runs.track == t].paired_input_check.value_counts().to_dict()
                               for t in ("ABLATION", "MORE") if "paired_input_check" in runs},
    }
    keep = ["track", "run_uid", "K", "feature_method", "architecture", "m", "seed", "accuracy", "f1_macro",
            "min_separation_ratio", "min_separation", "yodd_norm_fraction", "final_cluster_loss",
            "final_supervised_train_loss", "final_supervised_val_loss", "active_dimensions", "total_runtime_sec", "n_params"]
    runs[keep].sort_values(["track", "m", "feature_method", "K", "seed"]).to_csv(HERE / "runs_all.csv", index=False)

    # ------------------------------------------------------------------ primary D − A
    prim = cell_contrast(runs, "A", "D")
    off = pd.read_csv(AN / "paired_primary_D_minus_A.csv")
    oa = off[off.metric == "accuracy"].set_index(["feature_method", "K"])
    chk = max(abs(r["mean"] / 100 - oa.loc[(r.feature, r.K), "mean_delta"]) for _, r in prim.iterrows())
    num["crosscheck_primary_max_abs_diff"] = float(chk)
    tables["T2"] = [[r.feature, int(r.K), r.a_mean, r.a_sd, r.b_mean, r.b_sd, r["mean"], r.lo, r.hi, r.g, r.p]
                    for _, r in prim.iterrows()]
    num["primary_acc"] = summarize_contrast(prim)
    num["primary_acc_by_feature"] = prim.groupby("feature")["mean"].mean().to_dict()
    num["primary_acc_ci_pos_cells"] = [f"{r.feature} K={r.K}" for _, r in prim.iterrows() if r.lo > 0]
    num["primary_acc_ci_neg_cells"] = [f"{r.feature} K={r.K}" for _, r in prim.iterrows() if r.hi < 0]
    num["primary_acc_neg_cells"] = [f"{r.feature} K={r.K}" for _, r in prim.iterrows() if r["mean"] < 0]
    num["primary_acc_sign_test_p"] = float(stats.binomtest(num["primary_acc"]["n_pos"], 24).pvalue)
    holm_p = holm(prim.p.values)
    num["primary_acc_holm_min"] = float(holm_p.min())
    num["primary_acc_min_raw_p"] = float(prim.p.min())
    for metric, scale in (("f1_macro", 100.0), ("min_separation_ratio", 1.0)):
        c = cell_contrast(runs, "A", "D", metric, scale=scale)
        num[f"primary_{metric}"] = summarize_contrast(c)
    # all seed-level deltas positive in a cell
    num["primary_cells_all5_pos"] = int(sum(all(x > 0 for x in d) for d in prim.deltas))
    num["primary_seed_deltas_pos"] = int(sum(sum(x > 0 for x in d) for d in prim.deltas))

    # McNemar
    mcn = pd.read_csv(AN / "mcnemar_primary.csv")
    sig = mcn.p_value_holm_within_seed < 0.05
    dbet = mcn.a_wrong_b_right > mcn.a_right_b_wrong
    num["mcnemar"] = {"n": int(len(mcn)), "sig": int(sig.sum()), "d_fewer_errors": int(dbet.sum()),
                      "sig_d_better": int((sig & dbet).sum()), "sig_a_better": int((sig & ~dbet).sum())}

    # model means
    num["model_mean_acc"] = (runs.groupby("m").accuracy.mean() * 100).to_dict()
    num["model_mean_acc_primary_cells"] = (runs[runs.track == "PRIMARY"].groupby("m").accuracy.mean() * 100).to_dict()
    abl_cells = runs[runs.K.isin([3, 6, 10]) & runs.m.isin(list("ABCD"))]
    num["model_mean_acc_ablation_cells"] = (abl_cells.groupby("m").accuracy.mean() * 100).to_dict()
    num["model_mean_sep_ablation_cells"] = abl_cells.groupby("m").min_separation_ratio.mean().to_dict()
    num["best_model_count_ablation_cells"] = (
        abl_cells.groupby(["feature_method", "K", "m"]).accuracy.mean().unstack().idxmax(axis=1).value_counts().to_dict())

    # ------------------------------------------------------------------ ablation
    abl = []
    for a, b, lab in (("A", "B", "B−A"), ("A", "C", "C−A"), ("B", "D", "D−B"), ("C", "D", "D−C")):
        c = cell_contrast(runs, a, b, ks=[3, 6, 10])
        c["contrast"] = lab
        abl.append(c)
    abl = pd.concat(abl, ignore_index=True)
    abl["p_holm"] = abl.groupby(["feature", "K"]).p.transform(lambda s: holm(s.values))
    offa = pd.read_csv(AN / "ablation_contrasts.csv")
    offa = offa[offa.metric == "accuracy"]
    mapc = {"A-B (B-A)": "B−A", "A-C (C-A)": "C−A", "B-D (D-B)": "D−B", "C-D (D-C)": "D−C"}
    offa = offa.assign(contrast=offa.contrast.map(mapc)).set_index(["contrast", "feature_method", "K"])
    num["crosscheck_ablation_max_abs_diff"] = float(max(
        abs(r["mean"] / 100 - offa.loc[(r.contrast, r.feature, r.K), "mean_delta"]) for _, r in abl.iterrows()))
    num["crosscheck_ablation_holm_max_abs_diff"] = float(max(
        abs(r.p_holm - offa.loc[(r.contrast, r.feature, r.K), "sign_flip_p_holm"]) for _, r in abl.iterrows()))
    num["ablation"] = {lab: summarize_contrast(g) for lab, g in abl.groupby("contrast")}
    num["ablation_holm_min"] = float(abl.p_holm.min())
    t3 = []
    for f in FEATS:
        for k in (3, 6, 10):
            row = [f, k]
            for lab in ("B−A", "C−A", "D−B", "D−C"):
                r = abl[(abl.contrast == lab) & (abl.feature == f) & (abl.K == k)].iloc[0]
                row += [r["mean"], r.lo, r.hi, r.p_holm]
            t3.append(row)
    tables["T3"] = t3

    # ------------------------------------------------------------------ MORE-REPRO (within-protocol MORE)
    am = cell_contrast(runs, "M", "A")
    dm = cell_contrast(runs, "M", "D")
    tables["T4"] = [[ra.feature, int(ra.K), ra.a_mean, ra.a_sd, ra["mean"], ra.lo, ra.hi, rd["mean"], rd.lo, rd.hi]
                    for (_, ra), (_, rd) in zip(am.iterrows(), dm.iterrows())]
    num["more_A_minus_M"] = summarize_contrast(am)
    num["more_D_minus_M"] = summarize_contrast(dm)
    num["more_M_mean_acc"] = float(runs[runs.m == "M"].accuracy.mean() * 100)
    num["more_M_by_feature"] = (runs[runs.m == "M"].groupby("feature_method").accuracy.mean() * 100).to_dict()
    num["more_M_K3_PCA"] = float(runs[(runs.m == "M") & (runs.K == 3) & (runs.feature_method == "PCA")].accuracy.mean() * 100)
    num["more_M_K10_mean"] = float(runs[(runs.m == "M") & (runs.K == 10)].accuracy.mean() * 100)
    num["more_neg_cells_A"] = [f"{r.feature} K={r.K}" for _, r in am.iterrows() if r["mean"] < 0]
    num["more_neg_cells_D"] = [f"{r.feature} K={r.K}" for _, r in dm.iterrows() if r["mean"] < 0]

    # ------------------------------------------------------------------ classical baselines
    cb = pd.read_csv(DATA / "primary__15_Classical_Baselines.csv")
    base = cb.groupby(["feature_method", "K", "baseline"]).accuracy.mean().unstack() * 100
    qm = runs[runs.m.isin(["A", "D", "M"])].groupby(["feature_method", "K", "m"]).accuracy.mean().unstack() * 100
    t5 = []
    for f in FEATS:
        for k in KS:
            t5.append([f, k, 100.0 / k, base.loc[(f, k), "NC"], base.loc[(f, k), "LR"], qm.loc[(f, k), "M"],
                       qm.loc[(f, k), "A"], qm.loc[(f, k), "D"]])
    tables["T5"] = t5
    t5d = pd.DataFrame(t5, columns=["f", "K", "chance", "NC", "LR", "M", "A", "D"])
    num["baseline_mean"] = t5d[["chance", "NC", "LR", "M", "A", "D"]].mean().to_dict()
    num["baseline_gap"] = {"A-LR": float((t5d.A - t5d.LR).mean()), "D-LR": float((t5d.D - t5d.LR).mean()),
                           "A-NC": float((t5d.A - t5d.NC).mean()), "D-NC": float((t5d.D - t5d.NC).mean())}
    num["baseline_cells_D_ge_NC"] = [f"{r.f} K={r.K}" for _, r in t5d.iterrows() if r.D >= r.NC]
    num["baseline_cells_D_ge_LR"] = [f"{r.f} K={r.K}" for _, r in t5d.iterrows() if r.D >= r.LR]
    num["baseline_by_feature"] = t5d.groupby("f")[["NC", "LR", "A", "D", "M"]].mean().to_dict()
    num["lr_converged_all"] = bool(cb[cb.baseline == "LR"].lr_converged.astype(str).str.lower().eq("true").all())

    # ------------------------------------------------------------------ 6↔9 (pre-specified) and exploratory pairs
    six = pd.read_csv(AN / "g1_11_six_nine.csv")
    q = six[six.model_code.isin(list("ABCD"))]
    num["six_nine"] = {
        "share_of_errors_mean_ABCD": float(q.share_of_errors_6_9.mean()),
        "share_by_feature": q.groupby("feature_method").share_of_errors_6_9.mean().to_dict(),
        "share_by_model": six.groupby("model_code").share_of_errors_6_9.mean().to_dict(),
        "merged_gain_ABCD_pp": float((q.accuracy_merged_6_9 - q.accuracy).mean() * 100),
        "merged_gain_by_model_pp": (six.assign(g=(six.accuracy_merged_6_9 - six.accuracy) * 100).groupby("model_code").g.mean()).to_dict(),
        "D_acc": float(six[six.model_code == "D"].accuracy.mean()),
        "D_acc_merged": float(six[six.model_code == "D"].accuracy_merged_6_9.mean()),
    }
    cf = pd.concat([pd.read_csv(DATA / f"{t}__10_Confusion_Data.csv") for t in ("primary", "ablation", "more")])
    cf = cf[(cf.K == 10)].copy()
    cf["m"] = cf.architecture.map(CODE)
    offd = cf[(cf.actual_class != cf.predicted_class) & cf.m.isin(list("ABCD"))].copy()
    offd["pair"] = [f"{min(a, b)}↔{max(a, b)}" for a, b in zip(offd.actual_class.astype(int), offd.predicted_class.astype(int))]
    pairs = (offd.groupby(["feature_method", "pair"])["count"].sum() /
             offd.groupby("feature_method")["count"].sum()).rename("share").reset_index()
    pairs["rank"] = pairs.groupby("feature_method").share.rank(ascending=False, method="first").astype(int)
    pairs.sort_values(["feature_method", "rank"]).to_csv(HERE / "confusion_pairs_k10.csv", index=False)
    num["confusion_top3"] = {f: [(r.pair, float(r.share)) for _, r in g.nsmallest(3, "rank").iterrows()]
                             for f, g in pairs.groupby("feature_method")}
    num["confusion_pair_78"] = pairs[pairs.pair == "7↔8"].set_index("feature_method").share.to_dict()
    num["confusion_pair_05"] = pairs[pairs.pair == "0↔5"].set_index("feature_method").share.to_dict()
    num["confusion_pair_69"] = pairs[pairs.pair == "6↔9"].set_index("feature_method").share.to_dict()
    num["confusion_uniform_share"] = 1 / 45
    pc = pd.concat([pd.read_csv(DATA / f"{t}__09_Per_Class_Metrics.csv") for t in ("primary", "ablation")])
    pc = pc[pc.K == 10]
    num["recall_k10_by_feature_class"] = pc.groupby(["feature_method", "class"]).recall.mean().unstack().round(4).to_dict(orient="index")

    # ------------------------------------------------------------------ readout diagnostics (exploratory correlations)
    num["yodd_mean"] = runs[runs.m.isin(list("ABCD"))].groupby("m").yodd_norm_fraction.mean().to_dict()
    d = runs[runs.m == "D"].copy()
    d["ra"] = d.groupby(["feature_method", "K"]).accuracy.rank()
    d["ry"] = d.groupby(["feature_method", "K"]).yodd_norm_fraction.rank()
    r_ = stats.spearmanr(d.ry, d.ra)
    num["spearman_yodd_acc_D_within"] = {"rho": float(r_.statistic), "p": float(r_.pvalue), "n": int(len(d))}
    sep = {}
    for m in "ABCD":
        q2 = runs[runs.m == m]
        s_ = stats.spearmanr(q2.min_separation_ratio, q2.accuracy)
        sep[m] = {"rho": float(s_.statistic), "p": float(s_.pvalue), "n": int(len(q2))}
    q3 = runs[runs.m.isin(list("ABCD"))].copy()
    q3["ra"] = q3.groupby(["feature_method", "K"]).accuracy.rank()
    q3["rs"] = q3.groupby(["feature_method", "K"]).min_separation_ratio.rank()
    s_ = stats.spearmanr(q3.rs, q3.ra)
    sep["within_all"] = {"rho": float(s_.statistic), "p": float(s_.pvalue), "n": int(len(q3))}
    num["spearman_sep_acc"] = sep

    # ------------------------------------------------------------------ runtime and microbenchmark
    num["runtime_median_sec"] = runs.groupby("m").total_runtime_sec.median().to_dict()
    num["runtime_mean_sec"] = runs.groupby("m").total_runtime_sec.mean().to_dict()
    bm = pd.read_csv(DATA / "primary__18_Circuit_Benchmark.csv")
    num["benchmark_unitary_ms"] = (bm.groupby("model").unitary_median_sec.median() * 1000).to_dict()
    num["benchmark_wall_ms"] = (bm.groupby("model").wall_median_sec.median() * 1000).to_dict()
    num["benchmark_cpu"] = sorted(bm.cpu_model.dropna().unique().tolist()) if "cpu_model" in bm else []
    hw = json.loads((DATA / "reports" / "hardware_pretraining.json").read_text())
    num["hardware"] = hw
    num["n_parallel_declared"] = sorted(runs.n_parallel_declared.dropna().astype(int).unique().tolist()) if "n_parallel_declared" in runs else []

    # ------------------------------------------------------------------ Jalur B (secondary sensitivity)
    jb = pd.read_csv(DATA / "jalurB_runs.csv")
    jb = jb[jb.run_mode == "CONFIRMATORY"].copy()
    jb["m"] = jb.architecture.map(CODE)
    jb = jb.rename(columns={"n_classes": "K"})
    num["jalurB_n"] = int(len(jb))
    num["jalurB_status"] = jb.status.value_counts().to_dict()
    num["jalurB_test_used_for_selection"] = jb.test_used_for_selection.astype(str).value_counts().to_dict()
    jrows, tj = [], []
    for f in FEATS:
        for k in (3, 6, 10):
            sel = lambda m_, src: (src[(src.m == m_) & (src.feature_method == f) & (src.K == k)]
                                   .set_index("seed").accuracy.reindex(SEEDS).values * 100)
            aA, dA = sel("A", runs), sel("D", runs)
            aB, dB = sel("A", jb), sel("D", jb)
            pa, pb = paired(dA - aA), paired(dB - aB)
            sa, sd_ = paired(aB - aA), paired(dB - dA)
            jrows.append({"feature": f, "K": k, "A_jalurA": aA.mean(), "A_jalurB": aB.mean(), "D_jalurA": dA.mean(),
                          "D_jalurB": dB.mean(), "DA_jalurA": pa["mean"], "DA_jalurA_lo": pa["lo"], "DA_jalurA_hi": pa["hi"],
                          "DA_jalurB": pb["mean"], "DA_jalurB_lo": pb["lo"], "DA_jalurB_hi": pb["hi"],
                          "A_B_minus_A": sa["mean"], "D_B_minus_A": sd_["mean"]})
            tj.append([f, k, aB.mean(), dB.mean(), pa["mean"], pb["mean"], pb["lo"], pb["hi"]])
    jrows = pd.DataFrame(jrows)
    jrows.to_csv(HERE / "jalurB_summary.csv", index=False)
    tables["TJ"] = tj
    num["jalurB"] = {"DA_mean_jalurA": float(jrows.DA_jalurA.mean()), "DA_mean_jalurB": float(jrows.DA_jalurB.mean()),
                     "DA_pos_jalurB": int((jrows.DA_jalurB > 0).sum()), "DA_pos_jalurA": int((jrows.DA_jalurA > 0).sum()),
                     "same_sign": int((np.sign(jrows.DA_jalurA) == np.sign(jrows.DA_jalurB)).sum()),
                     "A_change_mean": float(jrows.A_B_minus_A.mean()), "D_change_mean": float(jrows.D_B_minus_A.mean()),
                     "ci_pos_jalurB": int((jrows.DA_jalurB_lo > 0).sum()), "ci_neg_jalurB": int((jrows.DA_jalurB_hi < 0).sum())}

    # ------------------------------------------------------------------ cross-dataset (descriptive)
    mprim = cell_contrast(mn, "A", "D")
    cells = prim[["feature", "K", "mean", "lo", "hi"]].rename(columns={"mean": "madbase", "lo": "madbase_lo", "hi": "madbase_hi"})
    cells = cells.merge(mprim[["feature", "K", "mean", "lo", "hi"]].rename(
        columns={"mean": "mnist", "lo": "mnist_lo", "hi": "mnist_hi"}), on=["feature", "K"])
    cells.to_csv(HERE / "cross_dataset_cells.csv", index=False)
    rho = stats.spearmanr(cells.mnist, cells.madbase)
    mabl = []
    for a, b, lab in (("A", "B", "B−A"), ("A", "C", "C−A"), ("B", "D", "D−B"), ("C", "D", "D−C")):
        c = cell_contrast(mn, a, b, ks=[3, 6, 10])
        c["contrast"] = lab
        mabl.append(c)
    mabl = pd.concat(mabl, ignore_index=True)
    mam, mdm = cell_contrast(mn, "M", "A"), cell_contrast(mn, "M", "D")
    mbase = pd.read_csv(MNIST_AN / "baseline_comparison.csv")
    mcn_m = pd.read_csv(MNIST_AN / "mcnemar_primary.csv")
    six_m = pd.read_csv(MNIST_AN / "g1_11_six_nine.csv")

    def base_means(bdf, rr):
        # mean accuracy of NC and LR = quantum accuracy − delta (delta = quantum − baseline)
        out = {}
        for bl in ("NC", "LR"):
            x = bdf[(bdf.baseline == bl) & (bdf.quantum_model == "A")].merge(
                rr[rr.m == "A"][["feature_method", "K", "seed", "accuracy"]], on=["feature_method", "K", "seed"])
            out[bl] = float((x.accuracy - x.delta_accuracy).mean() * 100)
        return out

    mb = base_means(mbase, mn)
    xb = base_means(pd.read_csv(AN / "baseline_comparison.csv"), runs)
    num["crosscheck_baseline_LR_mean"] = {"from_deltas": xb["LR"], "from_sheet": float(num["baseline_mean"]["LR"])}
    sigm = mcn_m.p_value_holm_within_seed < 0.05
    cross = {
        "mnist": {"acc": {m: float(mn[mn.m == m].accuracy.mean() * 100) for m in "ABCDM"},
                  "acc_primary_cells": {m: float(mn[(mn.m == m) & mn.K.isin(KS)].accuracy.mean() * 100) for m in "AD"},
                  "acc_ablation_cells": {m: float(mn[(mn.m == m) & mn.K.isin([3, 6, 10])].accuracy.mean() * 100) for m in "ABCD"},
                  "DA": summarize_contrast(mprim), "DA_by_feature": mprim.groupby("feature")["mean"].mean().to_dict(),
                  "DA_f1": summarize_contrast(cell_contrast(mn, "A", "D", "f1_macro")),
                  "DA_sep": summarize_contrast(cell_contrast(mn, "A", "D", "min_separation_ratio", scale=1.0)),
                  "ablation": {lab: summarize_contrast(g) for lab, g in mabl.groupby("contrast")},
                  "A_minus_M": summarize_contrast(mam), "D_minus_M": summarize_contrast(mdm),
                  "NC": mb["NC"], "LR": mb["LR"],
                  "mcnemar_sig": int(sigm.sum()), "mcnemar_d_fewer": int((mcn_m.a_wrong_b_right > mcn_m.a_right_b_wrong).sum()),
                  "six_nine_share_ABCD": float(six_m[six_m.model_code.isin(list("ABCD"))].share_of_errors_6_9.mean()),
                  "chance_mean": float(np.mean([100 / k for k in KS]))},
        "madbase": {"acc": {m: float(runs[runs.m == m].accuracy.mean() * 100) for m in "ABCDM"},
                    "acc_primary_cells": {m: float(runs[(runs.m == m)].accuracy.mean() * 100) for m in "AD"},
                    "acc_ablation_cells": {m: float(abl_cells[abl_cells.m == m].accuracy.mean() * 100) for m in "ABCD"},
                    "DA": num["primary_acc"], "DA_by_feature": num["primary_acc_by_feature"],
                    "DA_f1": num["primary_f1_macro"], "DA_sep": num["primary_min_separation_ratio"],
                    "ablation": num["ablation"], "A_minus_M": num["more_A_minus_M"], "D_minus_M": num["more_D_minus_M"],
                    "NC": xb["NC"], "LR": xb["LR"], "mcnemar_sig": num["mcnemar"]["sig"],
                    "mcnemar_d_fewer": num["mcnemar"]["d_fewer_errors"],
                    "six_nine_share_ABCD": num["six_nine"]["share_of_errors_mean_ABCD"],
                    "chance_mean": float(np.mean([100 / k for k in KS]))},
        "cell_DA_spearman": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
        "cell_DA_same_sign": int((np.sign(cells.mnist) == np.sign(cells.madbase)).sum()),
        "feature_acc": {
            "mnist": (mn[mn.m.isin(["A", "D"])].groupby(["feature_method", "m"]).accuracy.mean() * 100).unstack().to_dict(orient="index"),
            "madbase": (runs[runs.m.isin(["A", "D"])].groupby(["feature_method", "m"]).accuracy.mean() * 100).unstack().to_dict(orient="index")},
    }
    num["cross"] = cross
    tables["TX"] = cross

    # ------------------------------------------------------------------ mechanism (exploratory, if available)
    mf = MECH / "runs_mechanism.csv"
    if mf.exists():
        mech = pd.read_csv(mf)
        conv = pd.read_csv(MECH / "convergence.csv")
        conv["model"] = conv.architecture.map(CODE)
        mech = mech.merge(conv[["model", "feature", "K", "seed", "sup_still_improving_last100", "clu_still_improving_last100",
                                "sup_frac_improvement_last100"]], on=["model", "feature", "K", "seed"], how="left")
        mech.to_csv(MECH / "runs_mechanism_with_convergence.csv", index=False)
        g = mech.groupby("model")
        tm = {}
        for m_ in "ACDB":
            q4 = mech[mech.model == m_]
            tm[m_] = {"rank": float(q4.jac_rank.mean()), "eff": float(q4.jac_eff_rank.mean()),
                      "probe": float(q4.probe_all15.mean() * 100), "probe9": float(q4.probe_yeven9.mean() * 100),
                      "acc": float(q4.acc_reproduced.mean() * 100),
                      "gap": float((q4.probe_all15 - q4.acc_reproduced).mean() * 100),
                      "sup": float(q4.sup_still_improving_last100.astype(bool).mean() * 100),
                      "clu": float(q4.clu_still_improving_last100.astype(bool).mean() * 100),
                      "oracle": float(q4.acc_oracle_labels.mean() * 100)}
        tables["T6"] = tm
        piv = lambda col: mech.pivot_table(index=["feature", "K", "seed"], columns="model", values=col)
        P, E_, O = piv("probe_all15") * 100, piv("jac_eff_rank"), piv("acc_oracle_labels") * 100
        pairs_ = {}
        for lab, (x, y) in {"D-B": ("D", "B"), "C-A": ("C", "A"), "B-A": ("B", "A"), "D-C": ("D", "C"), "D-A": ("D", "A")}.items():
            dd = (P[x] - P[y]).values
            s_ = stats.ttest_1samp(dd, 0)
            ci = s_.confidence_interval(0.95)
            pairs_[lab] = {"mean": float(dd.mean()), "lo": float(ci.low), "hi": float(ci.high)}
        dd9 = mech[mech.model == "D"]
        d9 = (dd9.probe_all15 - dd9.probe_yeven9).values * 100
        ci9 = stats.ttest_1samp(d9, 0).confidence_interval(0.95)
        dl = (dd9.acc_reproduced - dd9.acc_labels_yeven_only).values * 100
        cil = stats.ttest_1samp(dl, 0).confidence_interval(0.95)
        ed = (E_["D"] - E_["B"]).values
        cie = stats.ttest_1samp(ed, 0).confidence_interval(0.95)
        mech["ra"] = mech.groupby(["feature", "K"]).acc_reproduced.rank()
        mech["re"] = mech.groupby(["feature", "K"]).jac_eff_rank.rank()
        mech["rg"] = mech.groupby(["feature", "K"]).apply(lambda q5: (q5.probe_all15 - q5.acc_reproduced).rank()).reset_index(level=[0, 1], drop=True)
        s1, s2 = stats.spearmanr(mech.re, mech.ra), stats.spearmanr(mech.re, mech.rg)
        bsup = mech[mech.model == "B"].sup_still_improving_last100.astype(bool)
        dsup = mech[mech.model == "D"].sup_still_improving_last100.astype(bool)
        bclu = mech[mech.model == "B"].clu_still_improving_last100.astype(bool)
        dclu = mech[mech.model == "D"].clu_still_improving_last100.astype(bool)
        f_sup = stats.fisher_exact([[bsup.sum(), (~bsup).sum()], [dsup.sum(), (~dsup).sum()]])
        f_clu = stats.fisher_exact([[bclu.sum(), (~bclu).sum()], [dclu.sum(), (~dclu).sum()]])
        rnd = pd.read_csv(MECH / "jacobian_random.csv") if (MECH / "jacobian_random.csv").exists() else None
        num["mechanism"] = {
            "n_runs": int(len(mech)), "max_abs_acc_repro_diff": float((mech.acc_logged - mech.acc_reproduced).abs().max()),
            "table": tm, "probe_pairs": pairs_,
            "probe_drop_without_yodd_D": {"mean": float(d9.mean()), "lo": float(ci9.low), "hi": float(ci9.high)},
            "label_acc_drop_without_yodd_D": {"mean": float(dl.mean()), "lo": float(cil.low), "hi": float(cil.high)},
            "eff_rank_D_minus_B": {"mean": float(ed.mean()), "lo": float(cie.low), "hi": float(cie.high)},
            "rank_D_values": sorted(mech[mech.model == "D"].jac_rank.unique().tolist()),
            "rank_B_values": sorted(mech[mech.model == "B"].jac_rank.unique().tolist()),
            "rank_C_values": sorted(mech[mech.model == "C"].jac_rank.unique().tolist()),
            "rank_A_values": sorted(mech[mech.model == "A"].jac_rank.unique().tolist()),
            "last_layer_data_rz_colnorm_max": float(mech[mech.model == "D"].last_layer_data_rz_colnorm_max.max())
            if "last_layer_data_rz_colnorm_max" in mech else None,
            "rz_colnorm_share_D": float(mech[mech.model == "D"].rz_colnorm_share.mean()),
            "spearman_effrank_acc_within": {"rho": float(s1.statistic), "p": float(s1.pvalue)},
            "spearman_effrank_gap_within": {"rho": float(s2.statistic), "p": float(s2.pvalue)},
            "fisher_sup_B_vs_D": {"B": float(bsup.mean() * 100), "D": float(dsup.mean() * 100), "p": float(f_sup.pvalue)},
            "fisher_clu_B_vs_D": {"B": float(bclu.mean() * 100), "D": float(dclu.mean() * 100), "p": float(f_clu.pvalue)},
            "oracle_order": {m_: tm[m_]["oracle"] for m_ in tm},
            "random_eff_rank": rnd.groupby("model").jac_eff_rank.mean().to_dict() if rnd is not None else None,
            "random_rank": rnd.groupby("model").jac_rank.mean().to_dict() if rnd is not None else None,
        }
        mnm = pd.read_csv(ROOT / "paper" / "build" / "mechanism" / "runs_mechanism.csv")
        mnm = mnm[mnm.K.isin([3, 6, 10])]
        num["cross"]["mnist"]["mechanism"] = {m_: {"probe": float(mnm[mnm.model == m_].probe_all15.mean() * 100),
                                                   "eff": float(mnm[mnm.model == m_].jac_eff_rank.mean()),
                                                   "acc": float(mnm[mnm.model == m_].acc_reproduced.mean() * 100)}
                                              for m_ in "ACDB" if "jac_eff_rank" in mnm}

    (HERE / "tables.json").write_text(json.dumps(tables, indent=1, ensure_ascii=False, default=float))
    (HERE / "numbers.json").write_text(json.dumps(num, indent=1, ensure_ascii=False, default=float))
    print("crosscheck primary", num["crosscheck_primary_max_abs_diff"], "ablation", num["crosscheck_ablation_max_abs_diff"],
          "holm", num["crosscheck_ablation_holm_max_abs_diff"])
    print("wrote tables.json, numbers.json, runs_all.csv")


if __name__ == "__main__":
    main()
