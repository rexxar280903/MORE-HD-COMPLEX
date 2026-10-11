# Template Hasil — Studi Digit Arab

**Status:** `TEMPLATE — JANGAN DIISI DENGAN HASIL MNIST`
**Versi protokol/SAP:** 1.0
**Tanggal freeze desain:** 2026-10-09 Asia/Jakarta
**Commit analisis:** `[RESULT: commit/ref yang direkam saat analisis final]`

Semua token `[RESULT]` hanya boleh diisi dari artefak digit Arab yang lulus
admission checks. Jangan menghapus tabel kegagalan atau deviasi meskipun nilainya
nol.

## 1. Ringkasan eksekutif

Dataset `[RESULT: nama dan versi]` dievaluasi menggunakan protokol `[RESULT:
versi]`. Dari `[RESULT]` run konfirmatori yang direncanakan, `[RESULT]` diterima,
`[RESULT]` gagal, dan `[RESULT]` tidak tersedia. Estimand primer MORE-HD-C minus
MORE-HD menunjukkan `[RESULT: ringkasan netral dengan ketidakpastian]`.

Interpretasi dibatasi pada snapshot, split, model, fitur, dan seed yang
dilaporkan di bawah.

## 2. Identitas dataset dan protokol

| Item | Nilai |
|---|---|
| Dataset/versi | `[RESULT]` |
| Glif/label | `[RESULT]` |
| Lisensi/citation | `[RESULT]` |
| Raw-manifest SHA-256 | `[RESULT]` |
| Split-manifest set/hash | `[RESULT]` |
| Asal final test | `[RESULT: OFFICIAL_SOURCE/CONSTRUCTED_FROZEN]` |
| Train/validation/test per kelas | `[RESULT]` |
| Class regimes | `[RESULT]` |
| Feature methods | `[RESULT]` |
| Seed konfirmatori | `[RESULT]` |
| Budget clustering/supervised | `[RESULT]` |
| SAP/protocol version | `[RESULT]` |
| Code commit/environment | `[RESULT]` |

## 3. Audit dataset

| Pemeriksaan | Hasil | Keputusan/dampak |
|---|---|---|
| File terbaca | `[RESULT]` | `[RESULT]` |
| Jumlah per kelas | `[RESULT]` | `[RESULT]` |
| Exact duplicate | `[RESULT]` | `[RESULT]` |
| Near duplicate lintas split | `[RESULT]` | `[RESULT]` |
| Writer leakage | `[RESULT/N.A.]` | `[RESULT]` |
| Label conflict | `[RESULT]` | `[RESULT]` |
| File rusak/blank/extreme crop | `[RESULT]` | `[RESULT]` |

## 4. Accounting run

| Track | Direncanakan | Completed valid | Failed | Missing | Excluded |
|---|---:|---:|---:|---:|---:|
| Primer A/D | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| Ablation B/C | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| Referensi quantum | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| Baseline | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

### Run tidak lengkap atau ditolak

| Run ID | Track/model | Status | Alasan | Attempt | Dampak pada pairing |
|---|---|---|---|---:|---|
| `[RESULT atau NONE]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

## 5. Hasil primer A versus D

Satuan accuracy/macro-F1: `[RESULT: proporsi atau poin persentase]`.

| Fitur | K/regime | n paired seeds/runs | Metric | A mean ± SD | D mean ± SD | D−A mean [95% CI] | Effect size | p raw | p adjusted |
|---|---|---:|---|---|---|---|---|---:|---:|
| `[RESULT]` | `[RESULT]` | `[RESULT]` | Accuracy | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| `[RESULT]` | `[RESULT]` | `[RESULT]` | Macro-F1 | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| `[RESULT]` | `[RESULT]` | `[RESULT]` | Min separation ratio | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

Kesimpulan primer: `[RESULT: ikuti hierarchy/multiplicity SAP; sertakan besaran
efek dan ketidakpastian, bukan hanya p-value]`.

## 6. Diagnostik struktur

| Fitur | K/regime | Model | Active dimensions | Y-odd norm fraction | Min separation | Degenerate count |
|---|---|---|---:|---:|---:|---:|
| `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

Interpretasi: `[RESULT]`. Aktivasi Y-odd tidak boleh disamakan dengan manfaat
akurasi tanpa hubungan empiris yang sesuai desain.

## 7. Ablation

| Fitur | K/regime | Kontras | n paired seeds/runs | Mean difference [95% CI] | Effect size | p adjusted | Interpretasi terbatas |
|---|---|---|---:|---|---|---:|---|
| `[RESULT]` | `[RESULT]` | A−B | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| `[RESULT]` | `[RESULT]` | A−C | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| `[RESULT]` | `[RESULT]` | B−D | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| `[RESULT]` | `[RESULT]` | C−D | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

## 8. Baseline

Aturan chance predictor: `[RESULT: rule, analytic/simulated, RNG dan jumlah
replikasi bila relevan]`.

| Fitur | K/regime | Baseline/model | Accuracy mean ± SD | Macro-F1 mean ± SD | Status perbandingan |
|---|---|---|---|---|---|
| `[RESULT]` | `[RESULT]` | Chance | `[RESULT]` | `[RESULT]` | Deskriptif |
| `[RESULT]` | `[RESULT]` | Nearest Centroid | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| `[RESULT]` | `[RESULT]` | Logistic Regression | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| `[RESULT]` | `[RESULT]` | A/D/M | `[RESULT]` | `[RESULT]` | `[RESULT]` |

Tidak boleh memakai tabel ini untuk klaim quantum advantage tanpa analisis sumber
daya, tuning, dan desain pembanding yang memadai.

## 9. Analisis per kelas

| Model | Fitur | K/regime | Kelas | Precision | Recall | F1 | Support |
|---|---|---|---:|---:|---:|---:|---:|
| `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

Pasangan confusion pra-spesifik: `[RESULT/NONE]`.
Analisis tambahan yang dipilih setelah melihat hasil: `[RESULT/NONE; tandai
EXPLORATORY]`.

## 10. Runtime dan optimasi

| Model | Kondisi | n run | nfev clustering | nfev supervised | CPU time | Wall time | Catatan convergence |
|---|---|---:|---:|---:|---|---|---|
| `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

## 11. Perbandingan deskriptif dengan MNIST

| Aspek | Studi digit Arab | Studi MNIST | Perbedaan protokol relevan |
|---|---|---|---|
| Dataset/populasi | `[RESULT]` | `[RESULT dari sumber MNIST yang sah]` | `[RESULT]` |
| Split/sampel | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| D−A accuracy | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| D−A macro-F1 | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| Pola fitur/confusion | `[RESULT]` | `[RESULT]` | `[RESULT]` |

Jangan menafsirkan selisih antardataset sebagai efek kausal dataset.

## 12. Deviations dan amendments

| ID | Waktu | Tahap | Deviasi dari protokol beku | Alasan | Outcome sudah tersedia? | Penanganan |
|---|---|---|---|---|---|---|
| `[RESULT atau NONE]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

## 13. Keterbatasan

- `[RESULT: representativitas dataset]`
- `[RESULT: jumlah seed/presisi interval]`
- `[RESULT: confound urutan kelas/K bila relevan]`
- `[RESULT: keterbatasan fitur/preprocessing]`
- `[RESULT: simulasi versus hardware, bila relevan]`
- `[RESULT: generalisasi lintas dataset]`

## 14. Kesimpulan

`[RESULT: satu paragraf yang menjawab RQ, menyebut besaran efek dan
ketidakpastian, serta tidak melampaui batas klaim]`

## 15. Artefak reproduksi

| Artefak | Path/link | SHA-256/status |
|---|---|---|
| Dataset manifest | `[RESULT]` | `[RESULT]` |
| Split manifests | `[RESULT]` | `[RESULT]` |
| Run manifest | `[RESULT]` | `[RESULT]` |
| Admission report | `[RESULT]` | `[RESULT]` |
| Analysis tables | `[RESULT]` | `[RESULT]` |
| Environment lock | `[RESULT]` | `[RESULT]` |
| Code commit | `[RESULT]` | `[RESULT]` |
