"""Isi research_arabic_digits/RESULTS.md (struktur RESULTS_TEMPLATE.md) dari artefak MADBase yang lolos admission.

Input : numbers.json, tables.json, runs_all.csv, data/*.csv (hasil extract_workbooks.py / analyze_madbase.py),
        ../../research_data/reports/g6_final_audit.json (dari tools/finalize_g6.py)
Output: research_arabic_digits/RESULTS.md
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
N = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))
T = json.loads((HERE / "tables.json").read_text(encoding="utf-8"))
DATA = HERE / "data"
FEAT = {"PCA": "PCA", "HU": "Hu", "ZERNIKE": "Zernike"}
NAME = {"A": "MORE-HD", "B": "MORE-HD-60P", "C": "MORE-HD-C-FixedRZ", "D": "MORE-HD-C", "M": "MORE-REPRO"}


def f(x, d=1):
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−").replace(".", ",")


def sg(x, d=1):
    s = f(x, d)
    return s if s.startswith("−") else "+" + s


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    audit_p = STUDY / "research_data" / "reports" / "g6_final_audit.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else {}
    P, AB, MC, Q = N["primary_acc"], N["ablation"], N["mcnemar"], N["quality"]
    runs = pd.read_csv(HERE / "runs_all.csv")
    pp = pd.read_csv(DATA / "analysis" / "paired_primary_D_minus_A.csv")
    seedagg = runs[runs.m.isin(["A", "D"])].groupby(["feature_method", "K", "m"])
    L = []
    A = L.append
    A("# Hasil — Studi Digit Arab (MADBase)\n")
    A("**Status:** `FINAL — G6 CLOSED`  ")
    A("**Versi protokol/SAP:** 1.0 (amendemen administratif 1.0.1, lihat `DEVIATION_LOG.md`)  ")
    A("**Tanggal freeze desain:** 2026-10-09 Asia/Jakarta  ")
    A(f"**Commit kode run:** `{Q['git_commit'][0]}` (engine `core/` identik dengan commit; lihat DEV-02)  ")
    A(f"**Audit G6:** `research_data/reports/g6_final_audit.json` ({audit.get('generated_at', 'lihat file')})\n")
    A("Semua angka di bawah berasal dari artefak digit Arab yang lolos admission (`final_admission.json`, "
      "`accepted = true`). Akurasi dan macro-F1 dilaporkan dalam persen; selisih dalam poin persentase (pp). "
      "Angka dihitung ulang oleh `paper/build/analyze_madbase.py` dan cocok dengan `research_data/analysis/*.csv` "
      f"(selisih maksimum {N['crosscheck_primary_max_abs_diff']:.1e} untuk primer dan "
      f"{N['crosscheck_ablation_max_abs_diff']:.1e} untuk ablasi).\n")

    A("## 1. Ringkasan eksekutif\n")
    A(f"Dataset MADBase (`MADBASE-AUC-V1`) dievaluasi dengan protokol v1.0. Dari 450 run konfirmatori yang "
      f"direncanakan, 450 diterima, 0 gagal, dan 0 tidak tersedia; 90 run Jalur B dan 240 fit baseline klasik juga "
      f"lengkap. Estimand primer MORE-HD-C minus MORE-HD bernilai rata-rata {sg(P['mean'])} pp akurasi atas 24 sel "
      f"(positif pada {P['n_pos']}/24; SK 95% di atas nol pada {P['ci_pos']} sel dan tidak ada yang di bawah nol; "
      f"rentang {sg(P['min'])} sampai {sg(P['max'])} pp). Dengan lima seed, uji sign-flip tidak dapat mencapai "
      f"α = 0,05 (p mentah terkecil 0,0625; p Holm terkecil {f(N['primary_acc_holm_min'], 2)}). Ablasi mengulang pola MNIST: "
      f"B − A {sg(AB['B−A']['mean'])} pp, C − A {sg(AB['C−A']['mean'])} pp, D − B {sg(AB['D−B']['mean'])} pp (negatif di "
      f"9/9 sel), D − C {sg(AB['D−C']['mean'])} pp. Klaim manfaat spesifik fase kompleks **tidak** didukung; kenaikan "
      "D atas A konsisten dengan tambahan kapasitas parameter terlatih.\n")
    A("Interpretasi dibatasi pada snapshot, split, model, fitur, dan seed yang dilaporkan di bawah.\n")

    A("## 2. Identitas dataset dan protokol\n")
    A(table(["Item", "Nilai"], [
        ["Dataset/versi", "Modified Arabic Digits Database (MADBase), `MADBASE-AUC-V1`, snapshot 2026-10-08"],
        ["Glif/label", "Arabic-Indic `٠١٢٣٤٥٦٧٨٩` → internal 0..9"],
        ["Lisensi/citation", "Tanpa lisensi redistribusi eksplisit → `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`; Abdleazeem & El-Sherif (2008), IJDAR 11:127–141, doi:10.1007/s10032-008-0073-5"],
        ["Raw-manifest SHA-256", "snapshot `8262dca2d3db6205e0112586dd9a02c35b0b371da292aab0a15dc92976793f78`; `raw_files.sha256`"],
        ["Split-manifest set/hash", "`splits/split_registry.json` SHA-256 `b6ad3ef7cc95e801afc1aeb3fc07ed7bdadf8443bc806a5da432c532abf27e90`"],
        ["Asal final test", "`OFFICIAL_SOURCE` — subset tetap 20 penulis official test, identik lintas seed"],
        ["Train/validation/test per kelas", "1000 / 100 / 200 (train-validation writer-disjoint dari official training pool)"],
        ["Class regimes", "kumulatif/nested K = 3..10"],
        ["Feature methods", "PCA, Hu, Zernike (8 kanal, scaler train-only ke [0, π])"],
        ["Seed konfirmatori", "101, 202, 303, 404, 505 (pilot 42 dikeluarkan)"],
        ["Budget clustering/supervised", f"{Q['cluster_nfev'][0]} / {Q['supervised_nfev'][0]} NFEV (14 pilot konvergensi MADBase, aturan plateau beku)"],
        ["SAP/protocol version", "1.0 (amendemen administratif 1.0.1)"],
        ["Code commit/environment", f"`{Q['git_commit'][0]}`; Python 3.13.15, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, PennyLane 0.45.1; Windows 11; i3-12100F"],
    ]))
    A("")

    A("## 3. Audit dataset\n")
    A(table(["Pemeriksaan", "Hasil", "Keputusan/dampak"], [
        ["File terbaca", "70.000/70.000 citra terdekode", "Tidak ada eksklusi"],
        ["Jumlah per kelas", "6.000 train + 1.000 test per kelas", "Seimbang"],
        ["Exact duplicate", "39 grup di dalam satu split; 12 grup melintasi train/test", "Dipertahankan dan dilaporkan sebagai keterbatasan"],
        ["Near duplicate lintas split", "3.879 pasangan kandidat (dHash ≤ 1, MAE ≤ 2)", "Bukan aturan eksklusi; dilaporkan"],
        ["Writer leakage", "0 penulis tumpang tindih (train 1–600, test 601–700)", "Split writer-disjoint valid"],
        ["Label conflict", "0 grup", "—"],
        ["File rusak/blank/extreme crop", "0 blank; 1.068 train dan 155 test menyentuh border", "Snapshot dipertahankan tanpa crop/re-centring"],
    ]))
    A("")

    A("## 4. Accounting run\n")
    A(table(["Track", "Direncanakan", "Completed valid", "Failed", "Missing", "Excluded"], [
        ["Primer A/D", 240, 240, 0, 0, 0], ["Ablation B/C", 90, 90, 0, 0, 0], ["Referensi quantum (M)", 120, 120, 0, 0, 0],
        ["Jalur B (sekunder)", 90, N["jalurB_status"].get("COMPLETED", 0), 0, 0, 0],
        ["Baseline (NC/LR fit)", 240, 240, 0, 0, 0]]))
    A("\nSetiap run konfirmatori selesai pada attempt 1; tidak ada run ganda, label kuantum degenerate, atau output uji "
      f"bernorma nol. Selisih backend maksimum {Q['backend_max_abs_diff']:.2e}. Pemeriksaan input berpasangan: "
      "ablation 90/90 MATCH, referensi 120/120 MATCH.\n")
    A("### Run tidak lengkap atau ditolak\n")
    A(table(["Run ID", "Track/model", "Status", "Alasan", "Attempt", "Dampak pada pairing"],
            [["NONE (konfirmatori)", "—", "—", "—", "—", "—"],
             ["AD-R002-S42 (pilot)", "Pilot konvergensi / MORE-HD-C", "FAILED → retry COMPLETED", "Izin file Windows (teknis)", "1 → 2", "Pilot tidak masuk estimasi"]]))
    A("")

    A("## 5. Hasil primer A versus D\n")
    A("Satuan accuracy/macro-F1: persen; selisih dalam poin persentase. p raw = sign-flip eksak dua sisi; p adjusted = "
      "Holm atas 24 sel dalam outcome yang sama.\n")
    rows = []
    for metric, scale, lab in (("accuracy", 100, "Accuracy"), ("f1_macro", 100, "Macro-F1"), ("min_separation_ratio", 1, "Min separation ratio")):
        sub = pp[pp.metric == metric].copy()
        from scipy.stats import false_discovery_control  # noqa: F401  (hanya memastikan SciPy tersedia)
        import numpy as np
        ps = sub.sign_flip_p.values
        order = np.argsort(ps)
        adj = np.empty_like(ps)
        run = 0.0
        for i, idx in enumerate(order):
            run = max(run, (len(ps) - i) * ps[idx])
            adj[idx] = min(1.0, run)
        sub["p_adj"] = adj
        for _, r in sub.iterrows():
            g = runs[(runs.feature_method == r.feature_method) & (runs.K == r.K)]
            a = g[g.m == "A"][metric] * scale
            d = g[g.m == "D"][metric] * scale
            dd = 2 if scale == 100 else 3
            rows.append([FEAT[r.feature_method], int(r.K), 5, lab, f"{f(a.mean(), dd - 1 if scale == 100 else 3)} ± {f(a.std(), dd - 1 if scale == 100 else 3)}",
                         f"{f(d.mean(), dd - 1 if scale == 100 else 3)} ± {f(d.std(), dd - 1 if scale == 100 else 3)}",
                         f"{sg(r.mean_delta * scale, 1 if scale == 100 else 3)} [{f(r.ci95_low * scale, 1 if scale == 100 else 3)}; {f(r.ci95_high * scale, 1 if scale == 100 else 3)}]",
                         f(r.hedges_g, 2), f(r.sign_flip_p, 4), f(r.p_adj, 4)])
    A(table(["Fitur", "K", "n paired seeds", "Metric", "A mean ± SD", "D mean ± SD", "D−A mean [95% CI]", "Hedges g", "p raw", "p adjusted"], rows))
    A(f"\nKesimpulan primer: D − A akurasi rata-rata {sg(P['mean'])} pp (PCA {sg(N['primary_acc_by_feature']['PCA'])}, "
      f"Hu {sg(N['primary_acc_by_feature']['HU'])}, Zernike {sg(N['primary_acc_by_feature']['ZERNIKE'])}); positif "
      f"pada {P['n_pos']}/24 sel; SK di atas nol pada {', '.join(c.replace('HU', 'Hu').replace('ZERNIKE', 'Zernike') for c in N['primary_acc_ci_pos_cells'])}; tidak ada SK di bawah "
      f"nol; macro-F1 {sg(N['primary_f1_macro']['mean'])} pp ({N['primary_f1_macro']['n_pos']}/24 positif); separation ratio "
      f"{sg(N['primary_min_separation_ratio']['mean'], 3)} ({N['primary_min_separation_ratio']['n_pos']}/24 positif, tanpa SK "
      f"yang mengecualikan nol). Sesuai SAP §6, kesimpulan bersifat estimation-first; tidak ada p terkoreksi < 0,05. "
      f"McNemar sekunder: {MC['sig']}/120 perbandingan seed-sel signifikan setelah Holm dalam seed ({MC['sig_d_better']} "
      f"menguntungkan D, {MC['sig_a_better']} menguntungkan A).\n")

    A("## 6. Diagnostik struktur\n")
    act = runs[runs.m.isin(list("ABCD"))].groupby(["feature_method", "m"]).agg(
        act=("active_dimensions", "mean"), y=("yodd_norm_fraction", "mean"), sep=("min_separation_ratio", "mean"))
    rows = [[FEAT[fm], "K = 3..10 (A/D) atau 3, 6, 10 (B/C)", f"{m} ({NAME[m]})", f"{r.act:.0f}", f(r.y, 3), f(r.sep, 3), 0]
            for (fm, m), r in act.iterrows()]
    A(table(["Fitur", "K/regime", "Model", "Active dimensions", "Y-odd norm fraction", "Min separation ratio (mean)", "Degenerate count"], rows))
    A(f"\nInterpretasi: A dan B selalu 9 dimensi aktif dan f_Y = 0 (Proposisi 1); C dan D selalu 15 dimensi aktif "
      f"(f_Y rata-rata {f(N['yodd_mean']['C'], 3)} dan {f(N['yodd_mean']['D'], 3)}). Di dalam sel, korelasi f_Y–akurasi D lemah "
      f"(ρ Spearman {f(N['spearman_yodd_acc_D_within']['rho'], 2)}, p = {f(N['spearman_yodd_acc_D_within']['p'], 3)}; eksploratif), "
      "sehingga aktivasi Y-odd tidak disamakan dengan manfaat akurasi.\n")

    A("## 7. Ablation\n")
    rows = []
    for r in T["T3"]:
        fm, k, v = r[0], r[1], r[2:]
        for i, lab in enumerate(("B−A", "C−A", "D−B", "D−C")):
            m, lo, hi, ph = v[4 * i: 4 * i + 4]
            note = "kapasitas riil +30 parameter" if lab == "B−A" else "akses fase beku" if lab == "C−A" else \
                "kompleks vs riil, 60 parameter" if lab == "D−B" else "RZ terlatih vs beku"
            rows.append([FEAT[fm], k, lab, 5, f"{sg(m)} [{f(lo)}; {f(hi)}]", "—", f(ph, 3), note])
    ab_full = pd.read_csv(DATA / "analysis" / "ablation_contrasts.csv")
    ab_full = ab_full[ab_full.metric == "accuracy"]
    gmap = {(c, fm, k): g for c, fm, k, g in zip(ab_full.contrast, ab_full.feature_method, ab_full.K, ab_full.hedges_g)}
    cmap = {"B−A": "A-B (B-A)", "C−A": "A-C (C-A)", "D−B": "B-D (D-B)", "D−C": "C-D (D-C)"}
    for r in rows:
        key = (cmap[r[2]], {"PCA": "PCA", "Hu": "HU", "Zernike": "ZERNIKE"}[r[0]], r[1])
        r[5] = f(gmap.get(key, float("nan")), 2)
    A(table(["Fitur", "K", "Kontras", "n paired seeds", "Mean difference [95% CI] (pp)", "Hedges g", "p adjusted (Holm, 4 kontras)", "Interpretasi terbatas"], rows))
    A(f"\nRingkasan: B terbaik pada {N['best_model_count_ablation_cells'].get('B', 0)}/9 sel; rata-rata akurasi sel ablasi "
      f"A {f(N['model_mean_acc_ablation_cells']['A'])}%, C {f(N['model_mean_acc_ablation_cells']['C'])}%, "
      f"D {f(N['model_mean_acc_ablation_cells']['D'])}%, B {f(N['model_mean_acc_ablation_cells']['B'])}%. p Holm terkecil "
      f"{f(N['ablation_holm_min'], 2)}.\n")

    A("## 8. Baseline\n")
    A("Aturan chance predictor: analitik, akurasi 1/K dan expected macro-F1 1/K pada kelas seimbang (tanpa RNG).\n")
    cb = pd.read_csv(DATA / "primary__15_Classical_Baselines.csv")
    rows = []
    for fm in ("PCA", "HU", "ZERNIKE"):
        for k in (3, 10):
            for bl, lab in (("CHANCE", "Chance"), ("NC", "Nearest Centroid"), ("LR", "Logistic Regression")):
                q = cb[(cb.feature_method == fm) & (cb.K == k) & (cb.baseline == bl)]
                if bl == "CHANCE" or q.empty:
                    rows.append([FEAT[fm], k, lab, f(100 / k), f(100 / k), "Deskriptif"])
                else:
                    rows.append([FEAT[fm], k, lab, f"{f(q.accuracy.mean() * 100)} ± {f(q.accuracy.std() * 100)}",
                                 f"{f(q.macro_f1.mean() * 100)} ± {f(q.macro_f1.std() * 100)}", "Deskriptif"])
            for m in ("A", "D", "M"):
                q = runs[(runs.feature_method == fm) & (runs.K == k) & (runs.m == m)]
                rows.append([FEAT[fm], k, f"{m} ({NAME[m]})", f"{f(q.accuracy.mean() * 100)} ± {f(q.accuracy.std() * 100)}",
                             f"{f(q.f1_macro.mean() * 100)} ± {f(q.f1_macro.std() * 100)}", "Deskriptif"])
    A(table(["Fitur", "K", "Baseline/model", "Accuracy mean ± SD", "Macro-F1 mean ± SD", "Status perbandingan"], rows))
    bm = N["baseline_mean"]
    A(f"\nRata-rata atas 24 sel: LR {f(bm['LR'])}%, NC {f(bm['NC'])}%, D {f(bm['D'])}%, A {f(bm['A'])}%, M {f(bm['M'])}%, "
      f"chance {f(bm['chance'])}%. D tidak pernah menyamai NC atau LR pada sel mana pun. Tabel ini tidak boleh dipakai "
      "untuk klaim quantum advantage.\n")

    A("## 9. Analisis per kelas\n")
    pc = pd.concat([pd.read_csv(DATA / f"{t}__09_Per_Class_Metrics.csv") for t in ("primary", "ablation")])
    pc = pc[pc.K == 10]
    pc["m"] = pc.architecture.map({"MORE-HD": "A", "MORE-HD-C": "D", "MORE-HD-60P": "B", "MORE-HD-C-FixedRZ": "C"})
    g = pc.groupby(["m", "feature_method", "class"]).agg(precision=("precision", "mean"), recall=("recall", "mean"),
                                                         f1=("f1", "mean"), support=("support", "mean")).reset_index()
    g.to_csv(HERE / "per_class_k10.csv", index=False)
    rows = [[f"{r.m} ({NAME[r.m]})", FEAT[r.feature_method], 10, f"{int(r['class'])} ({'٠١٢٣٤٥٦٧٨٩'[int(r['class'])]})",
             f(r.precision, 3), f(r.recall, 3), f(r.f1, 3), int(r.support)] for _, r in g[g.m.isin(["A", "D"])].iterrows()]
    A(table(["Model", "Fitur", "K", "Kelas", "Precision", "Recall", "F1", "Support"], rows))
    A("\n(Rata-rata lima seed; tabel lengkap untuk A–D di `paper/build/per_class_k10.csv`; confusion lengkap di workbook "
      "sheet `10_Confusion_Data`.)\n")
    six = N["six_nine"]
    A(f"Pasangan confusion pra-spesifik 6↔9 (K = 10): porsi galat A–D {f(six['share_of_errors_mean_ABCD'] * 100)}% "
      f"(PCA {f(six['share_by_feature']['PCA'] * 100)}%, Hu {f(six['share_by_feature']['HU'] * 100)}%, Zernike "
      f"{f(six['share_by_feature']['ZERNIKE'] * 100)}%); akurasi merged-6/9 naik rata-rata {f(six['merged_gain_ABCD_pp'])} pp.\n")
    cp = N["confusion_top3"]
    A("Analisis tambahan yang dipilih setelah melihat hasil (**EXPLORATORY**): pasangan galat terbesar pada K = 10 untuk "
      "A–D — " + "; ".join(f"{FEAT[k]}: " + ", ".join(f"{p} {f(s * 100)}%" for p, s in v) for k, v in cp.items()) +
      ". Pada fitur invarian-rotasi (Hu, Zernike) pasangan ٧↔٨ dan ٠↔٥ menonjol; pada PCA, 7↔8 hanya "
      f"{f(N['confusion_pair_78']['PCA'] * 100)}%. File: `paper/build/confusion_pairs_k10.csv`.\n")

    A("## 10. Runtime dan optimasi\n")
    rr = pd.concat([pd.read_csv(DATA / f"{t}__01_Run_Summary.csv") for t in ("primary", "ablation", "more")])
    rr["m"] = rr.architecture.map({**{v: k for k, v in NAME.items()}})
    rows = []
    for m in "ADBCM":
        q = rr[rr.m == m]
        rows.append([f"{m} ({NAME[m]})", "semua sel track", len(q), int(q.cluster_nfev.median()), int(q.supervised_nfev.median()),
                     f"{f(q.total_cpu_sec.median(), 0)} s (median)", f"{f(q.total_runtime_sec.median(), 0)} s (median)",
                     "budget habis = valid (SAP §5); plateau pilot terpenuhi"])
    A(table(["Model", "Kondisi", "n run", "nfev clustering", "nfev supervised", "CPU time", "Wall time", "Catatan convergence"], rows))
    if "mechanism" in N:
        mm = N["mechanism"]
        A(f"\nEksploratif (sel ablasi): run yang masih membaik pada 100 evaluasi supervised terakhir — B "
          f"{f(mm['fisher_sup_B_vs_D']['B'], 0)}%, D {f(mm['fisher_sup_B_vs_D']['D'], 0)}% (Fisher p = "
          f"{f(mm['fisher_sup_B_vs_D']['p'], 2)}); clustering B {f(mm['fisher_clu_B_vs_D']['B'], 0)}%, D "
          f"{f(mm['fisher_clu_B_vs_D']['D'], 0)}% (p = {f(mm['fisher_clu_B_vs_D']['p'], 2)}). Paralelisme 8 proses untuk 16 "
          "run pertama lalu 6 proses (DEV-01); waktu bersifat deskriptif.\n")

    A("## 11. Perbandingan deskriptif dengan MNIST\n")
    X = N["cross"]
    mn, ad = X["mnist"], X["madbase"]
    A(table(["Aspek", "Studi digit Arab", "Studi MNIST", "Perbedaan protokol relevan"], [
        ["Dataset/populasi", "MADBase, 700 penulis, glif Arab-Indik", "MNIST (LeCun dkk., 1998)", "Domain, penulis, akuisisi tidak berpasangan"],
        ["Split/sampel", "1000/100/200 per kelas; train/val writer-disjoint; test 20 penulis tetap lintas seed",
         "1000/100/200 per kelas; manifest per seed", "Struktur test berbeda; budget 900/940 vs 840/1000"],
        ["D−A accuracy", f"{sg(ad['DA']['mean'])} pp ({ad['DA']['n_pos']}/24 positif; SK>0: {ad['DA']['ci_pos']}, SK<0: {ad['DA']['ci_neg']})",
         f"{sg(mn['DA']['mean'])} pp ({mn['DA']['n_pos']}/24 positif; SK>0: {mn['DA']['ci_pos']}, SK<0: {mn['DA']['ci_neg']})",
         f"Korelasi pola sel ρ = {f(X['cell_DA_spearman']['rho'], 2)}; tanda sama {X['cell_DA_same_sign']}/24"],
        ["D−A macro-F1", f"{sg(ad['DA_f1']['mean'])} pp ({ad['DA_f1']['n_pos']}/24)", f"{sg(mn['DA_f1']['mean'])} pp ({mn['DA_f1']['n_pos']}/24)", "—"],
        ["Ablasi B−A / C−A / D−B / D−C", " / ".join(sg(ad['ablation'][k]['mean']) for k in ("B−A", "C−A", "D−B", "D−C")),
         " / ".join(sg(mn['ablation'][k]['mean']) for k in ("B−A", "C−A", "D−B", "D−C")), "Pola arah identik"],
        ["Pola fitur/confusion", "PCA terbaik; galat invarian-rotasi terkonsentrasi pada ٧↔٨ dan ٠↔٥; 6↔9 "
         f"{f(ad['six_nine_share_ABCD'] * 100)}%", f"PCA terbaik; 6↔9 {f(mn['six_nine_share_ABCD'] * 100)}% galat", "Geometri glif berbeda"],
    ]))
    A("\nJangan menafsirkan selisih antardataset sebagai efek kausal dataset.\n")

    A("## 12. Deviations dan amendments\n")
    A("Lihat `DEVIATION_LOG.md` (DEV-01 s.d. DEV-05). Ringkasan: tidak ada deviasi yang mengubah data, model, metrik, "
      "budget, atau aturan analisis konfirmatori.\n")

    A("## 13. Keterbatasan\n")
    for t in [
        "Representativitas: 700 penulis MADBase; set uji konfirmatori hanya 20 penulis yang sama untuk semua seed; 12 grup duplikat identik melintasi train/test resmi.",
        "Lima seed: SK lebar; resolusi sign-flip minimum 0,0625 sehingga tidak ada uji tingkat seed yang dapat signifikan pada α = 0,05.",
        "Confound urutan kelas: K kumulatif mencampurkan jumlah kelas dan identitas glif yang ditambahkan.",
        "Fitur/preprocessing: delapan kanal; Hu dan Zernike invarian-rotasi sehingga ٧/٨ sulit dibedakan; tanpa centring atau augmentasi.",
        "Simulasi statevector tanpa noise; bukan hardware.",
        "Generalisasi lintas dataset hanya deskriptif; budget dan struktur split berbeda dari MNIST.",
        "Analisis mekanistik dan pasangan kelas bersifat eksploratif (dirancang setelah hasil diketahui).",
    ]:
        A(f"- {t}")
    A("")

    A("## 14. Kesimpulan\n")
    A(f"Pada MADBase, MORE-HD-C lebih akurat daripada MORE-HD sebesar rata-rata {sg(P['mean'])} pp (positif pada "
      f"{P['n_pos']}/24 sel, SK 95% per sel lebar dan hanya {P['ci_pos']} sel mengecualikan nol). Ablasi yang dibekukan "
      f"sebelumnya menunjukkan bahwa sirkuit riil 60 parameter lebih akurat daripada MORE-HD-C di 9/9 sel "
      f"(D − B {sg(AB['D−B']['mean'])} pp) dan melatih RZ hampir tidak berbeda dari membekukannya (D − C "
      f"{sg(AB['D−C']['mean'])} pp), sehingga kenaikan tidak dapat diatribusikan pada fase kompleks; temuan MNIST "
      "tereplikasi secara arah dan besaran. Semua model kuantum tetap jauh di bawah baseline klasik.\n")

    A("## 15. Artefak reproduksi\n")
    man = json.loads((STUDY / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8")) if (STUDY / "REPRODUCIBILITY_MANIFEST.json").exists() else {}
    arts = man.get("artifacts", {})

    def first(group, name=None):
        for x in arts.get(group, []):
            if name is None or x["path"].endswith(name):
                return f"`{x['path']}`", f"`{(x['sha256'] or '')[:16]}…`"
        return "—", "—"

    rows = [["Dataset manifest", *first("dataset_manifest", "raw_files.sha256")],
            ["Split manifests", *first("split_manifests", "split_registry.json")],
            ["Run manifest", *first("run_plan", "confirmatory_run_plan.json")],
            ["Admission report", *first("admission_report", "final_admission.json")],
            ["Analysis tables", *first("analysis_tables", "MORE_HD_analysis_tables.xlsx")],
            ["Environment lock", *first("environment", "requirements-lock-py313.txt")],
            ["Code commit", f"`{Q['git_commit'][0]}`", "engine `core/` tanpa diff terhadap commit"]]
    A(table(["Artefak", "Path/link", "SHA-256/status"], rows))
    A("\nDaftar lengkap (termasuk workbook konsolidasi dan seluruh CSV analisis) beserta SHA-256 penuh: "
      "`REPRODUCIBILITY_MANIFEST.json` / `.md`.\n")
    (STUDY / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", STUDY / "RESULTS.md")


if __name__ == "__main__":
    main()
