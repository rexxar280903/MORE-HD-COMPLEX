# Hasil — Studi Digit Arab (MADBase)

**Status:** `FINAL — G6 CLOSED`  
**Versi protokol/SAP:** 1.0 (amendemen administratif 1.0.1, lihat `DEVIATION_LOG.md`)  
**Tanggal freeze desain:** 2026-10-09 Asia/Jakarta  
**Commit kode run:** `188bab6a068afcb02d76b18bd23cf5449943ed2d` (engine `core/` identik dengan commit; lihat DEV-02)  
**Audit G6:** `research_data/reports/g6_final_audit.json` (2026-10-11T06:30:05+07:00)

Semua angka di bawah berasal dari artefak digit Arab yang lolos admission (`final_admission.json`, `accepted = true`). Akurasi dan macro-F1 dilaporkan dalam persen; selisih dalam poin persentase (pp). Angka dihitung ulang oleh `paper/build/analyze_madbase.py` dan cocok dengan `research_data/analysis/*.csv` (selisih maksimum 1.1e-16 untuk primer dan 1.5e-16 untuk ablasi).

## 1. Ringkasan eksekutif

Dataset MADBase (`MADBASE-AUC-V1`) dievaluasi dengan protokol v1.0. Dari 450 run konfirmatori yang direncanakan, 450 diterima, 0 gagal, dan 0 tidak tersedia; 90 run Jalur B dan 240 fit baseline klasik juga lengkap. Estimand primer MORE-HD-C minus MORE-HD bernilai rata-rata +5,9 pp akurasi atas 24 sel (positif pada 22/24; SK 95% di atas nol pada 3 sel dan tidak ada yang di bawah nol; rentang −5,4 sampai +25,4 pp). Dengan lima seed, uji sign-flip tidak dapat mencapai α = 0,05 (p mentah terkecil 0,0625; p Holm terkecil 1,00). Ablasi mengulang pola MNIST: B − A +12,3 pp, C − A +2,5 pp, D − B −9,0 pp (negatif di 9/9 sel), D − C +0,7 pp. Klaim manfaat spesifik fase kompleks **tidak** didukung; kenaikan D atas A konsisten dengan tambahan kapasitas parameter terlatih.

Interpretasi dibatasi pada snapshot, split, model, fitur, dan seed yang dilaporkan di bawah.

## 2. Identitas dataset dan protokol

| Item | Nilai |
|---|---|
| Dataset/versi | Modified Arabic Digits Database (MADBase), `MADBASE-AUC-V1`, snapshot 2026-10-08 |
| Glif/label | Arabic-Indic `٠١٢٣٤٥٦٧٨٩` → internal 0..9 |
| Lisensi/citation | Tanpa lisensi redistribusi eksplisit → `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`; Abdleazeem & El-Sherif (2008), IJDAR 11:127–141, doi:10.1007/s10032-008-0073-5 |
| Raw-manifest SHA-256 | snapshot `8262dca2d3db6205e0112586dd9a02c35b0b371da292aab0a15dc92976793f78`; `raw_files.sha256` |
| Split-manifest set/hash | `splits/split_registry.json` SHA-256 `b6ad3ef7cc95e801afc1aeb3fc07ed7bdadf8443bc806a5da432c532abf27e90` |
| Asal final test | `OFFICIAL_SOURCE` — subset tetap 20 penulis official test, identik lintas seed |
| Train/validation/test per kelas | 1000 / 100 / 200 (train-validation writer-disjoint dari official training pool) |
| Class regimes | kumulatif/nested K = 3..10 |
| Feature methods | PCA, Hu, Zernike (8 kanal, scaler train-only ke [0, π]) |
| Seed konfirmatori | 101, 202, 303, 404, 505 (pilot 42 dikeluarkan) |
| Budget clustering/supervised | 900 / 940 NFEV (14 pilot konvergensi MADBase, aturan plateau beku) |
| SAP/protocol version | 1.0 (amendemen administratif 1.0.1) |
| Code commit/environment | `188bab6a068afcb02d76b18bd23cf5449943ed2d`; Python 3.13.15, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, PennyLane 0.45.1; Windows 11; i3-12100F |

## 3. Audit dataset

| Pemeriksaan | Hasil | Keputusan/dampak |
|---|---|---|
| File terbaca | 70.000/70.000 citra terdekode | Tidak ada eksklusi |
| Jumlah per kelas | 6.000 train + 1.000 test per kelas | Seimbang |
| Exact duplicate | 39 grup di dalam satu split; 12 grup melintasi train/test | Dipertahankan dan dilaporkan sebagai keterbatasan |
| Near duplicate lintas split | 3.879 pasangan kandidat (dHash ≤ 1, MAE ≤ 2) | Bukan aturan eksklusi; dilaporkan |
| Writer leakage | 0 penulis tumpang tindih (train 1–600, test 601–700) | Split writer-disjoint valid |
| Label conflict | 0 grup | — |
| File rusak/blank/extreme crop | 0 blank; 1.068 train dan 155 test menyentuh border | Snapshot dipertahankan tanpa crop/re-centring |

## 4. Accounting run

| Track | Direncanakan | Completed valid | Failed | Missing | Excluded |
|---|---|---|---|---|---|
| Primer A/D | 240 | 240 | 0 | 0 | 0 |
| Ablation B/C | 90 | 90 | 0 | 0 | 0 |
| Referensi quantum (M) | 120 | 120 | 0 | 0 | 0 |
| Jalur B (sekunder) | 90 | 90 | 0 | 0 | 0 |
| Baseline (NC/LR fit) | 240 | 240 | 0 | 0 | 0 |

Setiap run konfirmatori selesai pada attempt 1; tidak ada run ganda, label kuantum degenerate, atau output uji bernorma nol. Selisih backend maksimum 1.78e-15. Pemeriksaan input berpasangan: ablation 90/90 MATCH, referensi 120/120 MATCH.

### Run tidak lengkap atau ditolak

| Run ID | Track/model | Status | Alasan | Attempt | Dampak pada pairing |
|---|---|---|---|---|---|
| NONE (konfirmatori) | — | — | — | — | — |
| AD-R002-S42 (pilot) | Pilot konvergensi / MORE-HD-C | FAILED → retry COMPLETED | Izin file Windows (teknis) | 1 → 2 | Pilot tidak masuk estimasi |

## 5. Hasil primer A versus D

Satuan accuracy/macro-F1: persen; selisih dalam poin persentase. p raw = sign-flip eksak dua sisi; p adjusted = Holm atas 24 sel dalam outcome yang sama.

| Fitur | K | n paired seeds | Metric | A mean ± SD | D mean ± SD | D−A mean [95% CI] | Hedges g | p raw | p adjusted |
|---|---|---|---|---|---|---|---|---|---|
| Hu | 3 | 5 | Accuracy | 70,1 ± 9,7 | 73,7 ± 3,6 | +3,5 [−8,8; 15,9] | 0,28 | 0,5625 | 1,0000 |
| Hu | 4 | 5 | Accuracy | 59,5 ± 4,4 | 54,1 ± 8,6 | −5,4 [−19,1; 8,2] | −0,40 | 0,3750 | 1,0000 |
| Hu | 5 | 5 | Accuracy | 45,3 ± 6,1 | 52,8 ± 8,5 | +7,5 [−8,6; 23,6] | 0,46 | 0,2500 | 1,0000 |
| Hu | 6 | 5 | Accuracy | 38,5 ± 6,1 | 42,9 ± 5,1 | +4,5 [−3,1; 12,0] | 0,58 | 0,1875 | 1,0000 |
| Hu | 7 | 5 | Accuracy | 30,6 ± 5,5 | 41,7 ± 7,6 | +11,2 [8,0; 14,4] | 3,46 | 0,0625 | 1,0000 |
| Hu | 8 | 5 | Accuracy | 29,1 ± 6,2 | 35,6 ± 2,9 | +6,4 [−2,8; 15,6] | 0,69 | 0,1250 | 1,0000 |
| Hu | 9 | 5 | Accuracy | 30,2 ± 3,4 | 33,9 ± 3,2 | +3,8 [−1,7; 9,2] | 0,69 | 0,1250 | 1,0000 |
| Hu | 10 | 5 | Accuracy | 28,6 ± 5,4 | 29,1 ± 6,1 | +0,5 [−10,7; 11,7] | 0,05 | 0,9375 | 1,0000 |
| PCA | 3 | 5 | Accuracy | 83,6 ± 11,6 | 82,4 ± 15,4 | −1,2 [−27,7; 25,4] | −0,04 | 0,8750 | 1,0000 |
| PCA | 4 | 5 | Accuracy | 64,3 ± 11,8 | 80,4 ± 3,2 | +16,2 [−1,9; 34,2] | 0,89 | 0,1250 | 1,0000 |
| PCA | 5 | 5 | Accuracy | 58,5 ± 13,2 | 67,7 ± 11,7 | +9,2 [−4,3; 22,7] | 0,68 | 0,1875 | 1,0000 |
| PCA | 6 | 5 | Accuracy | 60,8 ± 7,4 | 72,1 ± 6,6 | +11,3 [1,6; 21,0] | 1,16 | 0,0625 | 1,0000 |
| PCA | 7 | 5 | Accuracy | 45,8 ± 9,5 | 71,2 ± 4,0 | +25,4 [8,9; 41,9] | 1,53 | 0,0625 | 1,0000 |
| PCA | 8 | 5 | Accuracy | 51,5 ± 10,2 | 58,3 ± 6,4 | +6,8 [−9,7; 23,2] | 0,41 | 0,4375 | 1,0000 |
| PCA | 9 | 5 | Accuracy | 50,5 ± 10,2 | 54,1 ± 11,2 | +3,6 [−8,3; 15,6] | 0,30 | 0,4375 | 1,0000 |
| PCA | 10 | 5 | Accuracy | 44,1 ± 11,2 | 46,0 ± 9,3 | +1,9 [−3,2; 6,9] | 0,37 | 0,3750 | 1,0000 |
| Zernike | 3 | 5 | Accuracy | 75,4 ± 11,9 | 75,9 ± 8,5 | +0,5 [−16,4; 17,4] | 0,03 | 1,0000 | 1,0000 |
| Zernike | 4 | 5 | Accuracy | 61,6 ± 9,7 | 70,7 ± 9,7 | +9,0 [−8,7; 26,8] | 0,51 | 0,3750 | 1,0000 |
| Zernike | 5 | 5 | Accuracy | 50,7 ± 11,3 | 58,2 ± 6,3 | +7,5 [−7,2; 22,2] | 0,51 | 0,2500 | 1,0000 |
| Zernike | 6 | 5 | Accuracy | 53,7 ± 8,5 | 61,9 ± 4,4 | +8,2 [−4,1; 20,6] | 0,66 | 0,1250 | 1,0000 |
| Zernike | 7 | 5 | Accuracy | 48,8 ± 9,8 | 51,8 ± 5,2 | +3,0 [−9,6; 15,6] | 0,24 | 0,5625 | 1,0000 |
| Zernike | 8 | 5 | Accuracy | 43,5 ± 8,1 | 47,7 ± 2,6 | +4,2 [−5,7; 14,2] | 0,42 | 0,3125 | 1,0000 |
| Zernike | 9 | 5 | Accuracy | 35,1 ± 8,7 | 38,1 ± 5,1 | +3,0 [−14,1; 20,0] | 0,17 | 0,6875 | 1,0000 |
| Zernike | 10 | 5 | Accuracy | 35,6 ± 7,8 | 35,8 ± 3,3 | +0,2 [−12,6; 13,0] | 0,01 | 1,0000 | 1,0000 |
| Hu | 3 | 5 | Macro-F1 | 66,5 ± 14,0 | 72,9 ± 4,4 | +6,4 [−11,5; 24,4] | 0,36 | 0,5000 | 1,0000 |
| Hu | 4 | 5 | Macro-F1 | 58,2 ± 5,7 | 50,1 ± 10,0 | −8,1 [−24,9; 8,8] | −0,48 | 0,3125 | 1,0000 |
| Hu | 5 | 5 | Macro-F1 | 41,3 ± 9,7 | 48,7 ± 10,7 | +7,4 [−14,1; 28,9] | 0,34 | 0,3750 | 1,0000 |
| Hu | 6 | 5 | Macro-F1 | 36,0 ± 6,6 | 40,2 ± 8,0 | +4,2 [−5,1; 13,5] | 0,45 | 0,3125 | 1,0000 |
| Hu | 7 | 5 | Macro-F1 | 27,3 ± 6,9 | 39,8 ± 7,2 | +12,5 [7,4; 17,5] | 2,44 | 0,0625 | 1,0000 |
| Hu | 8 | 5 | Macro-F1 | 26,1 ± 7,5 | 33,7 ± 3,6 | +7,6 [−3,8; 18,9] | 0,66 | 0,1875 | 1,0000 |
| Hu | 9 | 5 | Macro-F1 | 28,0 ± 2,0 | 32,3 ± 4,9 | +4,3 [−1,6; 10,2] | 0,73 | 0,1875 | 1,0000 |
| Hu | 10 | 5 | Macro-F1 | 26,6 ± 5,9 | 27,4 ± 6,2 | +0,8 [−11,8; 13,3] | 0,06 | 0,8125 | 1,0000 |
| PCA | 3 | 5 | Macro-F1 | 83,1 ± 12,1 | 80,1 ± 18,3 | −3,0 [−33,0; 27,0] | −0,10 | 0,6875 | 1,0000 |
| PCA | 4 | 5 | Macro-F1 | 60,8 ± 13,9 | 80,1 ± 3,4 | +19,4 [−1,7; 40,4] | 0,91 | 0,1250 | 1,0000 |
| PCA | 5 | 5 | Macro-F1 | 55,3 ± 14,0 | 66,7 ± 12,5 | +11,4 [−5,5; 28,3] | 0,67 | 0,1875 | 1,0000 |
| PCA | 6 | 5 | Macro-F1 | 58,5 ± 8,5 | 71,8 ± 6,7 | +13,3 [2,7; 23,9] | 1,24 | 0,0625 | 1,0000 |
| PCA | 7 | 5 | Macro-F1 | 44,1 ± 10,9 | 70,8 ± 4,3 | +26,7 [8,1; 45,4] | 1,42 | 0,0625 | 1,0000 |
| PCA | 8 | 5 | Macro-F1 | 50,5 ± 10,7 | 57,1 ± 6,6 | +6,6 [−10,7; 24,0] | 0,38 | 0,4375 | 1,0000 |
| PCA | 9 | 5 | Macro-F1 | 49,5 ± 10,6 | 54,1 ± 10,7 | +4,6 [−7,6; 16,7] | 0,37 | 0,3125 | 1,0000 |
| PCA | 10 | 5 | Macro-F1 | 43,3 ± 12,5 | 44,9 ± 9,7 | +1,6 [−4,6; 7,7] | 0,26 | 0,5000 | 1,0000 |
| Zernike | 3 | 5 | Macro-F1 | 73,4 ± 14,3 | 75,4 ± 8,5 | +2,0 [−18,4; 22,4] | 0,10 | 0,7500 | 1,0000 |
| Zernike | 4 | 5 | Macro-F1 | 59,1 ± 11,7 | 70,0 ± 9,8 | +10,8 [−7,8; 29,5] | 0,58 | 0,2500 | 1,0000 |
| Zernike | 5 | 5 | Macro-F1 | 48,3 ± 11,9 | 55,8 ± 9,0 | +7,5 [−9,9; 24,8] | 0,43 | 0,3750 | 1,0000 |
| Zernike | 6 | 5 | Macro-F1 | 52,3 ± 9,7 | 61,1 ± 5,2 | +8,8 [−6,1; 23,7] | 0,59 | 0,1875 | 1,0000 |
| Zernike | 7 | 5 | Macro-F1 | 47,7 ± 9,6 | 50,3 ± 5,5 | +2,5 [−11,4; 16,5] | 0,18 | 0,6875 | 1,0000 |
| Zernike | 8 | 5 | Macro-F1 | 41,8 ± 8,4 | 46,6 ± 3,2 | +4,8 [−6,8; 16,4] | 0,41 | 0,3125 | 1,0000 |
| Zernike | 9 | 5 | Macro-F1 | 33,7 ± 8,5 | 38,1 ± 6,0 | +4,4 [−13,4; 22,2] | 0,25 | 0,5625 | 1,0000 |
| Zernike | 10 | 5 | Macro-F1 | 34,5 ± 7,9 | 34,3 ± 3,5 | −0,2 [−12,7; 12,2] | −0,02 | 0,9375 | 1,0000 |
| Hu | 3 | 5 | Min separation ratio | 0,002 ± 0,003 | 0,003 ± 0,003 | +0,001 [−0,003; 0,005] | 0,25 | 0,7500 | 1,0000 |
| Hu | 4 | 5 | Min separation ratio | 0,007 ± 0,010 | 0,006 ± 0,007 | −0,001 [−0,008; 0,005] | −0,21 | 0,5625 | 1,0000 |
| Hu | 5 | 5 | Min separation ratio | 0,000 ± 0,000 | 0,002 ± 0,004 | +0,002 [−0,003; 0,007] | 0,38 | 0,5000 | 1,0000 |
| Hu | 6 | 5 | Min separation ratio | 0,005 ± 0,005 | 0,001 ± 0,002 | −0,004 [−0,012; 0,005] | −0,41 | 0,5000 | 1,0000 |
| Hu | 7 | 5 | Min separation ratio | 0,000 ± 0,000 | 0,013 ± 0,015 | +0,012 [−0,006; 0,030] | 0,68 | 0,1250 | 1,0000 |
| Hu | 8 | 5 | Min separation ratio | 0,002 ± 0,003 | 0,005 ± 0,007 | +0,003 [−0,008; 0,014] | 0,24 | 0,5625 | 1,0000 |
| Hu | 9 | 5 | Min separation ratio | 0,002 ± 0,003 | 0,004 ± 0,003 | +0,002 [−0,004; 0,009] | 0,34 | 0,3750 | 1,0000 |
| Hu | 10 | 5 | Min separation ratio | 0,002 ± 0,002 | 0,003 ± 0,004 | +0,001 [−0,005; 0,007] | 0,20 | 0,6875 | 1,0000 |
| PCA | 3 | 5 | Min separation ratio | 0,195 ± 0,332 | 0,325 ± 0,436 | +0,130 [−0,299; 0,558] | 0,30 | 0,5625 | 1,0000 |
| PCA | 4 | 5 | Min separation ratio | 0,027 ± 0,026 | 0,047 ± 0,050 | +0,020 [−0,042; 0,081] | 0,31 | 0,6250 | 1,0000 |
| PCA | 5 | 5 | Min separation ratio | 0,039 ± 0,032 | 0,022 ± 0,035 | −0,017 [−0,058; 0,024] | −0,41 | 0,3125 | 1,0000 |
| PCA | 6 | 5 | Min separation ratio | 0,048 ± 0,050 | 0,113 ± 0,073 | +0,065 [−0,027; 0,158] | 0,71 | 0,0625 | 1,0000 |
| PCA | 7 | 5 | Min separation ratio | 0,027 ± 0,048 | 0,221 ± 0,134 | +0,194 [−0,017; 0,405] | 0,91 | 0,1250 | 1,0000 |
| PCA | 8 | 5 | Min separation ratio | 0,080 ± 0,092 | 0,088 ± 0,078 | +0,008 [−0,150; 0,165] | 0,05 | 0,8750 | 1,0000 |
| PCA | 9 | 5 | Min separation ratio | 0,060 ± 0,089 | 0,024 ± 0,022 | −0,037 [−0,125; 0,052] | −0,41 | 0,5000 | 1,0000 |
| PCA | 10 | 5 | Min separation ratio | 0,068 ± 0,089 | 0,031 ± 0,054 | −0,037 [−0,092; 0,018] | −0,66 | 0,1875 | 1,0000 |
| Zernike | 3 | 5 | Min separation ratio | 0,143 ± 0,303 | 0,138 ± 0,191 | −0,006 [−0,248; 0,236] | −0,02 | 0,6250 | 1,0000 |
| Zernike | 4 | 5 | Min separation ratio | 0,041 ± 0,033 | 0,108 ± 0,102 | +0,067 [−0,079; 0,212] | 0,46 | 0,1250 | 1,0000 |
| Zernike | 5 | 5 | Min separation ratio | 0,026 ± 0,025 | 0,072 ± 0,069 | +0,046 [−0,039; 0,132] | 0,54 | 0,2500 | 1,0000 |
| Zernike | 6 | 5 | Min separation ratio | 0,084 ± 0,144 | 0,068 ± 0,081 | −0,016 [−0,243; 0,211] | −0,07 | 0,8750 | 1,0000 |
| Zernike | 7 | 5 | Min separation ratio | 0,022 ± 0,033 | 0,032 ± 0,023 | +0,010 [−0,049; 0,069] | 0,17 | 0,6875 | 1,0000 |
| Zernike | 8 | 5 | Min separation ratio | 0,036 ± 0,033 | 0,040 ± 0,051 | +0,004 [−0,090; 0,099] | 0,04 | 0,9375 | 1,0000 |
| Zernike | 9 | 5 | Min separation ratio | 0,009 ± 0,012 | 0,026 ± 0,028 | +0,017 [−0,028; 0,062] | 0,37 | 0,3750 | 1,0000 |
| Zernike | 10 | 5 | Min separation ratio | 0,019 ± 0,023 | 0,023 ± 0,018 | +0,004 [−0,023; 0,030] | 0,15 | 0,8125 | 1,0000 |

Kesimpulan primer: D − A akurasi rata-rata +5,9 pp (PCA +9,1, Hu +4,0, Zernike +4,5); positif pada 22/24 sel; SK di atas nol pada PCA K=6, PCA K=7, Hu K=7; tidak ada SK di bawah nol; macro-F1 +6,5 pp (21/24 positif); separation ratio +0,020 (17/24 positif, tanpa SK yang mengecualikan nol). Sesuai SAP §6, kesimpulan bersifat estimation-first; tidak ada p terkoreksi < 0,05. McNemar sekunder: 86/120 perbandingan seed-sel signifikan setelah Holm dalam seed (65 menguntungkan D, 21 menguntungkan A).

## 6. Diagnostik struktur

| Fitur | K/regime | Model | Active dimensions | Y-odd norm fraction | Min separation ratio (mean) | Degenerate count |
|---|---|---|---|---|---|---|
| Hu | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | A (MORE-HD) | 9 | 0,000 | 0,003 | 0 |
| Hu | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | B (MORE-HD-60P) | 9 | 0,000 | 0,015 | 0 |
| Hu | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | C (MORE-HD-C-FixedRZ) | 15 | 0,445 | 0,003 | 0 |
| Hu | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | D (MORE-HD-C) | 15 | 0,428 | 0,005 | 0 |
| PCA | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | A (MORE-HD) | 9 | 0,000 | 0,068 | 0 |
| PCA | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | B (MORE-HD-60P) | 9 | 0,000 | 0,269 | 0 |
| PCA | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | C (MORE-HD-C-FixedRZ) | 15 | 0,353 | 0,234 | 0 |
| PCA | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | D (MORE-HD-C) | 15 | 0,449 | 0,109 | 0 |
| Zernike | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | A (MORE-HD) | 9 | 0,000 | 0,048 | 0 |
| Zernike | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | B (MORE-HD-60P) | 9 | 0,000 | 0,064 | 0 |
| Zernike | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | C (MORE-HD-C-FixedRZ) | 15 | 0,506 | 0,039 | 0 |
| Zernike | K = 3..10 (A/D) atau 3, 6, 10 (B/C) | D (MORE-HD-C) | 15 | 0,446 | 0,063 | 0 |

Interpretasi: A dan B selalu 9 dimensi aktif dan f_Y = 0 (Proposisi 1); C dan D selalu 15 dimensi aktif (f_Y rata-rata 0,435 dan 0,441). Di dalam sel, korelasi f_Y–akurasi D lemah (ρ Spearman 0,16, p = 0,076; eksploratif), sehingga aktivasi Y-odd tidak disamakan dengan manfaat akurasi.

## 7. Ablation

| Fitur | K | Kontras | n paired seeds | Mean difference [95% CI] (pp) | Hedges g | p adjusted (Holm, 4 kontras) | Interpretasi terbatas |
|---|---|---|---|---|---|---|---|
| PCA | 3 | B−A | 5 | +13,5 [−1,6; 28,6] | 0,89 | 0,250 | kapasitas riil +30 parameter |
| PCA | 3 | C−A | 5 | +5,2 [−15,7; 26,1] | 0,25 | 1,000 | akses fase beku |
| PCA | 3 | D−B | 5 | −14,7 [−32,6; 3,3] | −0,81 | 0,375 | kompleks vs riil, 60 parameter |
| PCA | 3 | D−C | 5 | −6,4 [−22,3; 9,6] | −0,40 | 1,000 | RZ terlatih vs beku |
| PCA | 6 | B−A | 5 | +14,6 [1,5; 27,7] | 1,11 | 0,500 | kapasitas riil +30 parameter |
| PCA | 6 | C−A | 5 | +7,2 [−14,3; 28,7] | 0,33 | 0,875 | akses fase beku |
| PCA | 6 | D−B | 5 | −3,3 [−10,1; 3,5] | −0,48 | 0,750 | kompleks vs riil, 60 parameter |
| PCA | 6 | D−C | 5 | +4,1 [−13,5; 21,7] | 0,23 | 0,875 | RZ terlatih vs beku |
| PCA | 10 | B−A | 5 | +21,9 [7,8; 36,0] | 1,54 | 0,250 | kapasitas riil +30 parameter |
| PCA | 10 | C−A | 5 | +3,0 [−11,6; 17,7] | 0,21 | 1,000 | akses fase beku |
| PCA | 10 | D−B | 5 | −20,1 [−31,8; −8,3] | −1,70 | 0,250 | kompleks vs riil, 60 parameter |
| PCA | 10 | D−C | 5 | −1,2 [−12,6; 10,3] | −0,10 | 1,000 | RZ terlatih vs beku |
| Hu | 3 | B−A | 5 | +7,3 [−10,8; 25,4] | 0,40 | 0,750 | kapasitas riil +30 parameter |
| Hu | 3 | C−A | 5 | +10,8 [3,4; 18,1] | 1,45 | 0,250 | akses fase beku |
| Hu | 3 | D−B | 5 | −3,8 [−18,2; 10,6] | −0,26 | 0,750 | kompleks vs riil, 60 parameter |
| Hu | 3 | D−C | 5 | −7,2 [−15,4; 0,9] | −0,88 | 0,375 | RZ terlatih vs beku |
| Hu | 6 | B−A | 5 | +12,7 [6,9; 18,4] | 2,19 | 0,250 | kapasitas riil +30 parameter |
| Hu | 6 | C−A | 5 | +6,6 [−7,1; 20,4] | 0,48 | 0,750 | akses fase beku |
| Hu | 6 | D−B | 5 | −8,2 [−12,4; −4,0] | −1,92 | 0,250 | kompleks vs riil, 60 parameter |
| Hu | 6 | D−C | 5 | −2,2 [−10,1; 5,7] | −0,27 | 0,750 | RZ terlatih vs beku |
| Hu | 10 | B−A | 5 | +7,1 [−1,6; 15,7] | 0,81 | 0,500 | kapasitas riil +30 parameter |
| Hu | 10 | C−A | 5 | +0,1 [−8,8; 9,0] | 0,01 | 1,000 | akses fase beku |
| Hu | 10 | D−B | 5 | −6,5 [−16,3; 3,3] | −0,66 | 0,562 | kompleks vs riil, 60 parameter |
| Hu | 10 | D−C | 5 | +0,4 [−10,1; 10,9] | 0,04 | 1,000 | RZ terlatih vs beku |
| Zernike | 3 | B−A | 5 | +7,5 [−6,9; 22,0] | 0,52 | 0,750 | kapasitas riil +30 parameter |
| Zernike | 3 | C−A | 5 | −0,8 [−21,8; 20,3] | −0,04 | 1,000 | akses fase beku |
| Zernike | 3 | D−B | 5 | −7,0 [−26,6; 12,6] | −0,36 | 1,000 | kompleks vs riil, 60 parameter |
| Zernike | 3 | D−C | 5 | +1,3 [−8,3; 10,8] | 0,13 | 1,000 | RZ terlatih vs beku |
| Zernike | 6 | B−A | 5 | +14,9 [0,6; 29,3] | 1,03 | 0,250 | kapasitas riil +30 parameter |
| Zernike | 6 | C−A | 5 | −4,0 [−13,3; 5,3] | −0,43 | 0,500 | akses fase beku |
| Zernike | 6 | D−B | 5 | −6,7 [−18,5; 5,1] | −0,56 | 0,500 | kompleks vs riil, 60 parameter |
| Zernike | 6 | D−C | 5 | +12,2 [−0,8; 25,3] | 0,93 | 0,250 | RZ terlatih vs beku |
| Zernike | 10 | B−A | 5 | +11,3 [1,3; 21,3] | 1,12 | 0,250 | kapasitas riil +30 parameter |
| Zernike | 10 | C−A | 5 | −5,5 [−14,7; 3,8] | −0,58 | 0,375 | akses fase beku |
| Zernike | 10 | D−B | 5 | −11,1 [−17,3; −5,0] | −1,79 | 0,250 | kompleks vs riil, 60 parameter |
| Zernike | 10 | D−C | 5 | +5,6 [−7,6; 18,9] | 0,42 | 0,375 | RZ terlatih vs beku |

Ringkasan: B terbaik pada 8/9 sel; rata-rata akurasi sel ablasi A 54,5%, C 57,0%, D 57,8%, B 66,8%. p Holm terkecil 0,25.

## 8. Baseline

Aturan chance predictor: analitik, akurasi 1/K dan expected macro-F1 1/K pada kelas seimbang (tanpa RNG).

| Fitur | K | Baseline/model | Accuracy mean ± SD | Macro-F1 mean ± SD | Status perbandingan |
|---|---|---|---|---|---|
| PCA | 3 | Chance | 33,3 | 33,3 | Deskriptif |
| PCA | 3 | Nearest Centroid | 96,5 ± 0,2 | 96,5 ± 0,2 | Deskriptif |
| PCA | 3 | Logistic Regression | 98,3 ± 0,4 | 98,3 ± 0,4 | Deskriptif |
| PCA | 3 | A (MORE-HD) | 83,6 ± 11,6 | 83,1 ± 12,1 | Deskriptif |
| PCA | 3 | D (MORE-HD-C) | 82,4 ± 15,4 | 80,1 ± 18,3 | Deskriptif |
| PCA | 3 | M (MORE-REPRO) | 73,3 ± 3,4 | 72,9 ± 3,3 | Deskriptif |
| PCA | 10 | Chance | 10,0 | 10,0 | Deskriptif |
| PCA | 10 | Nearest Centroid | 78,7 ± 0,6 | 79,0 ± 0,6 | Deskriptif |
| PCA | 10 | Logistic Regression | 83,9 ± 0,1 | 83,9 ± 0,1 | Deskriptif |
| PCA | 10 | A (MORE-HD) | 44,2 ± 11,2 | 43,3 ± 12,5 | Deskriptif |
| PCA | 10 | D (MORE-HD-C) | 46,0 ± 9,3 | 44,9 ± 9,7 | Deskriptif |
| PCA | 10 | M (MORE-REPRO) | 26,0 ± 3,9 | 24,5 ± 3,6 | Deskriptif |
| Hu | 3 | Chance | 33,3 | 33,3 | Deskriptif |
| Hu | 3 | Nearest Centroid | 90,2 ± 0,6 | 90,2 ± 0,6 | Deskriptif |
| Hu | 3 | Logistic Regression | 95,9 ± 0,2 | 95,9 ± 0,2 | Deskriptif |
| Hu | 3 | A (MORE-HD) | 70,1 ± 9,7 | 66,5 ± 14,0 | Deskriptif |
| Hu | 3 | D (MORE-HD-C) | 73,7 ± 3,6 | 72,9 ± 4,4 | Deskriptif |
| Hu | 3 | M (MORE-REPRO) | 75,8 ± 6,7 | 75,8 ± 6,6 | Deskriptif |
| Hu | 10 | Chance | 10,0 | 10,0 | Deskriptif |
| Hu | 10 | Nearest Centroid | 49,4 ± 1,0 | 48,3 ± 0,9 | Deskriptif |
| Hu | 10 | Logistic Regression | 69,5 ± 0,3 | 68,2 ± 0,3 | Deskriptif |
| Hu | 10 | A (MORE-HD) | 28,6 ± 5,4 | 26,6 ± 5,9 | Deskriptif |
| Hu | 10 | D (MORE-HD-C) | 29,1 ± 6,1 | 27,4 ± 6,2 | Deskriptif |
| Hu | 10 | M (MORE-REPRO) | 23,0 ± 4,6 | 22,1 ± 4,4 | Deskriptif |
| Zernike | 3 | Chance | 33,3 | 33,3 | Deskriptif |
| Zernike | 3 | Nearest Centroid | 85,4 ± 0,5 | 85,0 ± 0,5 | Deskriptif |
| Zernike | 3 | Logistic Regression | 94,6 ± 0,1 | 94,6 ± 0,1 | Deskriptif |
| Zernike | 3 | A (MORE-HD) | 75,4 ± 11,9 | 73,4 ± 14,3 | Deskriptif |
| Zernike | 3 | D (MORE-HD-C) | 75,9 ± 8,5 | 75,4 ± 8,5 | Deskriptif |
| Zernike | 3 | M (MORE-REPRO) | 64,4 ± 5,6 | 64,1 ± 5,4 | Deskriptif |
| Zernike | 10 | Chance | 10,0 | 10,0 | Deskriptif |
| Zernike | 10 | Nearest Centroid | 56,9 ± 0,5 | 56,4 ± 0,6 | Deskriptif |
| Zernike | 10 | Logistic Regression | 69,4 ± 0,3 | 69,1 ± 0,3 | Deskriptif |
| Zernike | 10 | A (MORE-HD) | 35,6 ± 7,8 | 34,5 ± 7,9 | Deskriptif |
| Zernike | 10 | D (MORE-HD-C) | 35,8 ± 3,3 | 34,3 ± 3,5 | Deskriptif |
| Zernike | 10 | M (MORE-REPRO) | 20,4 ± 0,8 | 16,7 ± 1,2 | Deskriptif |

Rata-rata atas 24 sel: LR 85,2%, NC 73,3%, D 55,7%, A 49,8%, M 37,9%, chance 17,9%. D tidak pernah menyamai NC atau LR pada sel mana pun. Tabel ini tidak boleh dipakai untuk klaim quantum advantage.

## 9. Analisis per kelas

| Model | Fitur | K | Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|---|---|
| A (MORE-HD) | Hu | 10 | 0 (٠) | 0,646 | 0,557 | 0,583 | 200 |
| A (MORE-HD) | Hu | 10 | 1 (١) | 0,463 | 0,284 | 0,346 | 200 |
| A (MORE-HD) | Hu | 10 | 2 (٢) | 0,217 | 0,363 | 0,254 | 200 |
| A (MORE-HD) | Hu | 10 | 3 (٣) | 0,199 | 0,283 | 0,214 | 200 |
| A (MORE-HD) | Hu | 10 | 4 (٤) | 0,148 | 0,125 | 0,122 | 200 |
| A (MORE-HD) | Hu | 10 | 5 (٥) | 0,444 | 0,434 | 0,435 | 200 |
| A (MORE-HD) | Hu | 10 | 6 (٦) | 0,292 | 0,435 | 0,303 | 200 |
| A (MORE-HD) | Hu | 10 | 7 (٧) | 0,178 | 0,141 | 0,134 | 200 |
| A (MORE-HD) | Hu | 10 | 8 (٨) | 0,148 | 0,124 | 0,132 | 200 |
| A (MORE-HD) | Hu | 10 | 9 (٩) | 0,232 | 0,111 | 0,137 | 200 |
| A (MORE-HD) | PCA | 10 | 0 (٠) | 0,612 | 0,452 | 0,514 | 200 |
| A (MORE-HD) | PCA | 10 | 1 (١) | 0,481 | 0,734 | 0,566 | 200 |
| A (MORE-HD) | PCA | 10 | 2 (٢) | 0,384 | 0,358 | 0,359 | 200 |
| A (MORE-HD) | PCA | 10 | 3 (٣) | 0,402 | 0,255 | 0,295 | 200 |
| A (MORE-HD) | PCA | 10 | 4 (٤) | 0,438 | 0,399 | 0,409 | 200 |
| A (MORE-HD) | PCA | 10 | 5 (٥) | 0,574 | 0,649 | 0,604 | 200 |
| A (MORE-HD) | PCA | 10 | 6 (٦) | 0,523 | 0,404 | 0,417 | 200 |
| A (MORE-HD) | PCA | 10 | 7 (٧) | 0,502 | 0,481 | 0,484 | 200 |
| A (MORE-HD) | PCA | 10 | 8 (٨) | 0,417 | 0,384 | 0,385 | 200 |
| A (MORE-HD) | PCA | 10 | 9 (٩) | 0,318 | 0,299 | 0,295 | 200 |
| A (MORE-HD) | Zernike | 10 | 0 (٠) | 0,411 | 0,421 | 0,403 | 200 |
| A (MORE-HD) | Zernike | 10 | 1 (١) | 0,532 | 0,644 | 0,579 | 200 |
| A (MORE-HD) | Zernike | 10 | 2 (٢) | 0,292 | 0,355 | 0,308 | 200 |
| A (MORE-HD) | Zernike | 10 | 3 (٣) | 0,291 | 0,305 | 0,277 | 200 |
| A (MORE-HD) | Zernike | 10 | 4 (٤) | 0,228 | 0,244 | 0,226 | 200 |
| A (MORE-HD) | Zernike | 10 | 5 (٥) | 0,596 | 0,444 | 0,500 | 200 |
| A (MORE-HD) | Zernike | 10 | 6 (٦) | 0,444 | 0,303 | 0,350 | 200 |
| A (MORE-HD) | Zernike | 10 | 7 (٧) | 0,367 | 0,338 | 0,333 | 200 |
| A (MORE-HD) | Zernike | 10 | 8 (٨) | 0,256 | 0,340 | 0,282 | 200 |
| A (MORE-HD) | Zernike | 10 | 9 (٩) | 0,245 | 0,166 | 0,197 | 200 |
| D (MORE-HD-C) | Hu | 10 | 0 (٠) | 0,619 | 0,637 | 0,620 | 200 |
| D (MORE-HD-C) | Hu | 10 | 1 (١) | 0,458 | 0,275 | 0,329 | 200 |
| D (MORE-HD-C) | Hu | 10 | 2 (٢) | 0,153 | 0,142 | 0,128 | 200 |
| D (MORE-HD-C) | Hu | 10 | 3 (٣) | 0,155 | 0,202 | 0,157 | 200 |
| D (MORE-HD-C) | Hu | 10 | 4 (٤) | 0,316 | 0,147 | 0,175 | 200 |
| D (MORE-HD-C) | Hu | 10 | 5 (٥) | 0,469 | 0,308 | 0,354 | 200 |
| D (MORE-HD-C) | Hu | 10 | 6 (٦) | 0,292 | 0,381 | 0,301 | 200 |
| D (MORE-HD-C) | Hu | 10 | 7 (٧) | 0,202 | 0,295 | 0,231 | 200 |
| D (MORE-HD-C) | Hu | 10 | 8 (٨) | 0,176 | 0,219 | 0,187 | 200 |
| D (MORE-HD-C) | Hu | 10 | 9 (٩) | 0,240 | 0,304 | 0,254 | 200 |
| D (MORE-HD-C) | PCA | 10 | 0 (٠) | 0,483 | 0,422 | 0,436 | 200 |
| D (MORE-HD-C) | PCA | 10 | 1 (١) | 0,644 | 0,883 | 0,736 | 200 |
| D (MORE-HD-C) | PCA | 10 | 2 (٢) | 0,451 | 0,393 | 0,383 | 200 |
| D (MORE-HD-C) | PCA | 10 | 3 (٣) | 0,378 | 0,320 | 0,344 | 200 |
| D (MORE-HD-C) | PCA | 10 | 4 (٤) | 0,620 | 0,509 | 0,544 | 200 |
| D (MORE-HD-C) | PCA | 10 | 5 (٥) | 0,735 | 0,465 | 0,533 | 200 |
| D (MORE-HD-C) | PCA | 10 | 6 (٦) | 0,354 | 0,370 | 0,353 | 200 |
| D (MORE-HD-C) | PCA | 10 | 7 (٧) | 0,510 | 0,462 | 0,467 | 200 |
| D (MORE-HD-C) | PCA | 10 | 8 (٨) | 0,379 | 0,469 | 0,408 | 200 |
| D (MORE-HD-C) | PCA | 10 | 9 (٩) | 0,284 | 0,308 | 0,285 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 0 (٠) | 0,417 | 0,263 | 0,317 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 1 (١) | 0,677 | 0,814 | 0,723 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 2 (٢) | 0,250 | 0,332 | 0,242 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 3 (٣) | 0,273 | 0,268 | 0,263 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 4 (٤) | 0,350 | 0,289 | 0,281 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 5 (٥) | 0,603 | 0,475 | 0,493 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 6 (٦) | 0,466 | 0,481 | 0,461 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 7 (٧) | 0,312 | 0,303 | 0,280 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 8 (٨) | 0,180 | 0,157 | 0,163 | 200 |
| D (MORE-HD-C) | Zernike | 10 | 9 (٩) | 0,264 | 0,197 | 0,209 | 200 |

(Rata-rata lima seed; tabel lengkap untuk A–D di `paper/build/per_class_k10.csv`; confusion lengkap di workbook sheet `10_Confusion_Data`.)

Pasangan confusion pra-spesifik 6↔9 (K = 10): porsi galat A–D 4,6% (PCA 9,1%, Hu 2,8%, Zernike 2,0%); akurasi merged-6/9 naik rata-rata 2,4 pp.

Analisis tambahan yang dipilih setelah melihat hasil (**EXPLORATORY**): pasangan galat terbesar pada K = 10 untuk A–D — Hu: 0↔5 6,3%, 7↔8 5,4%, 2↔3 5,2%; PCA: 6↔9 8,5%, 8↔9 8,0%, 1↔4 6,8%; Zernike: 7↔8 6,3%, 4↔6 6,1%, 3↔8 5,7%. Pada fitur invarian-rotasi (Hu, Zernike) pasangan ٧↔٨ dan ٠↔٥ menonjol; pada PCA, 7↔8 hanya 0,6%. File: `paper/build/confusion_pairs_k10.csv`.

## 10. Runtime dan optimasi

| Model | Kondisi | n run | nfev clustering | nfev supervised | CPU time | Wall time | Catatan convergence |
|---|---|---|---|---|---|---|---|
| A (MORE-HD) | semua sel track | 120 | 900 | 940 | 724 s (median) | 810 s (median) | budget habis = valid (SAP §5); plateau pilot terpenuhi |
| D (MORE-HD-C) | semua sel track | 120 | 900 | 940 | 726 s (median) | 811 s (median) | budget habis = valid (SAP §5); plateau pilot terpenuhi |
| B (MORE-HD-60P) | semua sel track | 45 | 900 | 940 | 869 s (median) | 959 s (median) | budget habis = valid (SAP §5); plateau pilot terpenuhi |
| C (MORE-HD-C-FixedRZ) | semua sel track | 45 | 900 | 940 | 714 s (median) | 802 s (median) | budget habis = valid (SAP §5); plateau pilot terpenuhi |
| M (MORE-REPRO) | semua sel track | 120 | 900 | 940 | 438 s (median) | 554 s (median) | budget habis = valid (SAP §5); plateau pilot terpenuhi |

Eksploratif (sel ablasi): run yang masih membaik pada 100 evaluasi supervised terakhir — B 67%, D 58% (Fisher p = 0,51); clustering B 18%, D 13% (p = 0,77). Paralelisme 8 proses untuk 16 run pertama lalu 6 proses (DEV-01); waktu bersifat deskriptif.

## 11. Perbandingan deskriptif dengan MNIST

| Aspek | Studi digit Arab | Studi MNIST | Perbedaan protokol relevan |
|---|---|---|---|
| Dataset/populasi | MADBase, 700 penulis, glif Arab-Indik | MNIST (LeCun dkk., 1998) | Domain, penulis, akuisisi tidak berpasangan |
| Split/sampel | 1000/100/200 per kelas; train/val writer-disjoint; test 20 penulis tetap lintas seed | 1000/100/200 per kelas; manifest per seed | Struktur test berbeda; budget 900/940 vs 840/1000 |
| D−A accuracy | +5,9 pp (22/24 positif; SK>0: 3, SK<0: 0) | +5,9 pp (20/24 positif; SK>0: 4, SK<0: 1) | Korelasi pola sel ρ = 0,04; tanda sama 18/24 |
| D−A macro-F1 | +6,5 pp (21/24) | +6,8 pp (20/24) | — |
| Ablasi B−A / C−A / D−B / D−C | +12,3 / +2,5 / −9,0 / +0,7 | +11,5 / +3,0 / −8,4 / +0,1 | Pola arah identik |
| Pola fitur/confusion | PCA terbaik; galat invarian-rotasi terkonsentrasi pada ٧↔٨ dan ٠↔٥; 6↔9 4,6% | PCA terbaik; 6↔9 3,1% galat | Geometri glif berbeda |

Jangan menafsirkan selisih antardataset sebagai efek kausal dataset.

## 12. Deviations dan amendments

Lihat `DEVIATION_LOG.md` (DEV-01 s.d. DEV-05). Ringkasan: tidak ada deviasi yang mengubah data, model, metrik, budget, atau aturan analisis konfirmatori.

## 13. Keterbatasan

- Representativitas: 700 penulis MADBase; set uji konfirmatori hanya 20 penulis yang sama untuk semua seed; 12 grup duplikat identik melintasi train/test resmi.
- Lima seed: SK lebar; resolusi sign-flip minimum 0,0625 sehingga tidak ada uji tingkat seed yang dapat signifikan pada α = 0,05.
- Confound urutan kelas: K kumulatif mencampurkan jumlah kelas dan identitas glif yang ditambahkan.
- Fitur/preprocessing: delapan kanal; Hu dan Zernike invarian-rotasi sehingga ٧/٨ sulit dibedakan; tanpa centring atau augmentasi.
- Simulasi statevector tanpa noise; bukan hardware.
- Generalisasi lintas dataset hanya deskriptif; budget dan struktur split berbeda dari MNIST.
- Analisis mekanistik dan pasangan kelas bersifat eksploratif (dirancang setelah hasil diketahui).

## 14. Kesimpulan

Pada MADBase, MORE-HD-C lebih akurat daripada MORE-HD sebesar rata-rata +5,9 pp (positif pada 22/24 sel, SK 95% per sel lebar dan hanya 3 sel mengecualikan nol). Ablasi yang dibekukan sebelumnya menunjukkan bahwa sirkuit riil 60 parameter lebih akurat daripada MORE-HD-C di 9/9 sel (D − B −9,0 pp) dan melatih RZ hampir tidak berbeda dari membekukannya (D − C +0,7 pp), sehingga kenaikan tidak dapat diatribusikan pada fase kompleks; temuan MNIST tereplikasi secara arah dan besaran. Semua model kuantum tetap jauh di bawah baseline klasik.

## 15. Artefak reproduksi

| Artefak | Path/link | SHA-256/status |
|---|---|---|
| Dataset manifest | `research_arabic_digits/research_data/manifests/raw_files.sha256` | `99f474e8af096e0e…` |
| Split manifests | `research_arabic_digits/splits/split_registry.json` | `b6ad3ef7cc95e801…` |
| Run manifest | `research_arabic_digits/research_data/reports/confirmatory_run_plan.json` | `d6808a70c167b71d…` |
| Admission report | `research_arabic_digits/research_data/reports/final_admission.json` | `b1db529ae6d9843a…` |
| Analysis tables | `research_arabic_digits/research_data/analysis/MORE_HD_analysis_tables.xlsx` | `232ed828232a37f0…` |
| Environment lock | `research_arabic_digits/requirements-lock-py313.txt` | `587047f603050dfb…` |
| Code commit | `188bab6a068afcb02d76b18bd23cf5449943ed2d` | engine `core/` tanpa diff terhadap commit |

Daftar lengkap (termasuk workbook konsolidasi dan seluruh CSV analisis) beserta SHA-256 penuh: `REPRODUCIBILITY_MANIFEST.json` / `.md`.
