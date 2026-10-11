# Statistical Analysis Plan — MADBase

**Status:** `FROZEN EXCEPT DATASET-SPECIFIC OPTIMIZER BUDGET`  
**Versi:** 1.0  
**Tanggal:** 2026-10-09

Tidak ada pilihan analisis di bawah yang boleh diubah berdasarkan metrik model
official test. Nilai budget optimizer bukan keputusan statistik dan diisi hanya
oleh aturan pilot konvergensi pra-spesifikasi.

## 1. Snapshot analisis

| Komponen | Nilai |
|---|---|
| Dataset | `MADBASE-AUC-V1` |
| Snapshot | `8262dca2d3db6205e0112586dd9a02c35b0b371da292aab0a15dc92976793f78` |
| Mapping label SHA-256 | `9d1e5c93e75322977a9e10f61612767475ca8524dbdf801fb7b94f786dfe21f9` |
| Split registry SHA-256 | `b6ad3ef7cc95e801afc1aeb3fc07ed7bdadf8443bc806a5da432c532abf27e90` |
| Split | writer-disjoint train/validation; fixed official test subset |
| Preprocessing | 28×28, invert polaritas, tanpa resize/augmentasi; fitur fit train-only |

Hash per-seed ada di `splits/split_registry.json` dan wajib cocok dengan run.

## 2. Unit analisis dan pairing

- Unit prediksi: satu official-test image.
- Unit replikasi primer: satu master seed dalam satu sel feature × K.
- Seed konfirmatori: 101, 202, 303, 404, 505.
- Unit pairing: A dan D dengan dataset, sample IDs, split, input arrays,
  clustering pairs, dan random namespaces yang diwajibkan identik.
- Official-test sample IDs sama lintas seed; karena itu citra bukan replikasi
  independen lintas seed.

Lima seed dan 200 test samples per kelas adalah batas feasibility komputasi,
bukan hasil power calculation yang menjamin deteksi efek kecil. Ketidakpastian
utama dilaporkan sebagai paired seed variation. Uji instance-level McNemar
hanya sekunder.

## 3. Matriks

- Primer: A/D × PCA/Hu/Zernike × K=3..10 × 5 seed = 240 run.
- Ablation: B/C × 3 fitur × K={3,6,10} × 5 seed = 90 run.
- MORE-REPRO: M × 3 fitur × K=3..10 × 5 seed = 120 run.
- Jalur B: A/D × 3 fitur × K={3,6,10} × 5 seed = 90 run.
- Baseline: Nearest Centroid dan Logistic Regression untuk 120 cells = 240
  fits; chance `1/K` dihitung analitik.

## 4. Outcome dan metrik

Tiga outcome primer:

1. accuracy official test;
2. macro-F1 official test;
3. minimum quantum-label separation ratio.

Untuk metric `m`, feature `f`, K `k`, seed `s`:

```text
d(m,f,k,s) = metric_D(m,f,k,s) - metric_A(m,f,k,s)
```

Estimand adalah mean `d` atas lima seed. Accuracy/F1 dilaporkan sebagai
proporsi dan delta boleh ditampilkan sebagai poin persentase dengan label unit.

Kontrak mesin:

- accuracy denominator = seluruh prediksi valid; missing/invalid membuat run
  gagal admission;
- macro-F1 memakai label tetap `0..K-1` dan `zero_division=0`;
- prediksi memilih jarak cosine minimum; tie memakai indeks kelas terendah dari
  `numpy.argmin`;
- zero/near-zero norm memakai epsilon `1e-10` dan guard numerik beku;
- separation ratio = minimum pairwise cosine distance dibagi regular-simplex
  bound `1 + 1/(K-1)`.

Outcome sekunder mencakup metrik per kelas, confusion matrix, mean/minimum
separation, Y-odd activity, active dimensions, loss, NFEV, runtime, dan baseline.

## 5. Admission dan missingness

Populasi intent-to-run adalah seluruh run ID yang dicetak oleh run plan sebelum
eksekusi. Valid completed set membutuhkan tepat satu attempt `COMPLETED` yang
lulus dataset, snapshot, split, sample, input, feature, pair, dan final-stage
checks. Failed attempts tetap disimpan.

Kegagalan teknis diulang dengan run UID, seed, config, dan random sources yang
sama. Run yang berhenti pada budget tetap valid bila tidak memiliki error
numerik/protokol; status non-convergence dilaporkan. Tidak ada eksklusi atau
retry berdasarkan accuracy/F1 final.

## 6. Ringkasan dan inferensi

Per sel/outcome laporkan n planned/valid, mean dan SD tiap model, paired mean
difference, SD, median, IQR, 95% t-interval untuk mean paired difference,
paired Hedges g, dan exact two-sided sign-flip p-value.

Holm diterapkan dalam keluarga outcome primer yang sama lintas 24 sel
feature × K. Semua p-value mentah dan adjusted dilaporkan.

Dengan lima seed, resolusi minimum exact two-sided sign-flip adalah `2/32 =
0.0625`; uji itu tidak dapat mencapai alpha 0.05 bahkan saat semua tanda sama.
Karena itu kesimpulan utama bersifat estimation-first: besar, arah, interval,
dan konsistensi efek. P-value tidak boleh dipresentasikan sebagai bukti
konfirmatori kuat. McNemar exact per seed adalah sekunder dan dikoreksi Holm
dalam seed.

## 7. Planned contrasts

- D − A: estimand primer.
- A − B: pengaruh tambahan layer/parameter pada state real-valued.
- A − C: akses fase beku dibanding model A.
- B − D: kontrol 60 parameter real versus model kompleks 60 parameter.
- C − D: RZ beku versus RZ terlatih.
- A − M dan D − M: perbandingan dengan MORE-REPRO pada protokol sama.

Empat kontras ablation dikoreksi Holm di setiap feature × K. Baseline dan
MORE literature values bersifat sekunder/deskriptif kecuali perbandingan paired
MORE-REPRO yang dilaporkan dengan prosedur sama.

## 8. Analisis kelas dan lintas dataset

Semua confusion matrix dan per-class precision/recall/F1 dilaporkan. Confusion
6↔9 dan merged-6/9 accuracy pada K=10 adalah sekunder yang dipra-spesifikasi.
Pemilihan contoh error setelah hasil dibuka diberi label eksploratif.

MADBase versus MNIST hanya dibandingkan secara deskriptif. Perbedaan tidak
ditafsirkan sebagai efek kausal dataset.

## 9. Keluaran wajib

- admission report dan seed-level metrics;
- paired primary differences, CI, effect size, raw/adjusted p-values;
- ablation, MORE-REPRO, McNemar, 6↔9, dan baseline tables;
- confusion/per-class outputs;
- manifest, environment, commit/ref, protocol version, dan failure/deviation
  records.

Implementasi yang berlaku adalah `analysis/run_analysis.py`. Pilot analysis
hanya memverifikasi mekanisme dan tidak boleh disalin ke tabel hasil final.

## 10. Boundary dan amendemen

Freeze desain dilakukan 2026-10-09 Asia/Jakarta. Confirmatory boundary dibuka
2026-10-09 ±23:45 WIB setelah budget MADBase 900/940 dibekukan; timestamp,
commit/ref (`188bab6`), environment, serta hash keputusan budget tercatat pada run
manifests. (Amendemen administratif 1.0.1, DEV-04.) Perubahan setelahnya masuk amendment log dan menyebut outcome apa
yang sudah tersedia.

| Tanggal | Versi | Perubahan | Outcome official test tersedia? |
|---|---|---|---|
| 2026-10-08 | 0.1 | Kerangka awal | Tidak |
| 2026-10-09 | 1.0 | Dataset, split, matriks, metrik, inferensi, admission, dan batas klaim dibekukan; budget tetap digated oleh pilot | Tidak ada metrik model konfirmatori |
| 2026-10-11 | 1.0.1 | Administratif: status boundary, penutupan G6, label eksploratif untuk analisis pasca-hasil (DEV-04, DEV-05) | Ya — tidak ada perubahan analisis konfirmatori |
