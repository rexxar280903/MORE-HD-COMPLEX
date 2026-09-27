# MORE-HD-COMPLEX

<!-- RESEARCH_PROGRESS_START -->
## Research Progress

![Research Progress](https://img.shields.io/badge/Research%20Progress-24.3%25-blue)

**Current research readiness: 24.3%**

`█████░░░░░░░░░░░░░░░ 24.3%`

Progress dihitung otomatis dari **38 item Research Readiness Gate** dengan bobot:
`CLOSED = 100%`, `READY FOR VERIFICATION = 75%`, `IN PROGRESS = 50%`, `PILOT ONLY = 25%`, dan `BLOCKED = 0%`.

### Progress History

| Tanggal | Commit acuan | Progress | Perubahan | Ringkasan |
|---|---|---:|---:|---|
| 2026-09-22 | [`5a26fdd`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/5a26fdd13d09123de634997b62e6bf75dfee840f) | 13.9% | — | Baseline readiness setelah protokol ukuran dataset dikunci. |
| 2026-09-23 | [`34e51ec`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/34e51ecdbc049f296113fbfa282d688b93ceebbe) | 10.1% | -3.8 pp | Scope diperketat dengan penambahan 10 item readiness baru. |
| 2026-09-26 | [`6f8c7c1`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/6f8c7c1f04b9d29e00f36bb6aa603a8681fd601b) | 18.2% | +8.1 pp | Protokol multi-seed, analysis plan, workbook 240-run, dan logging objective mulai terkunci/terverifikasi. |
| 2026-09-27 | [`5aeb1ca`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/5aeb1cabe3e55673b51b0cf09e4c0865ba79f2f4) | 20.9% | +2.7 pp | Protokol deterministic clustering pairs dan dokumentasi 240-run diselaraskan. |
| 2026-09-27 | [`f753c85`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/f753c85a08d8e2d45cbe6429d8ea79c2a6e7eeaa) | 24.3% | +3.4 pp | docs: lock targeted ablation for G1-05 and G1-07 |
| 2026-09-27 | [`6670319`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/6670319c34497679b0781ddf0a3654319a8fe4d7) | **24.3%** | **0.0 pp** | docs: consolidate ablation into main pseudocode |

**Trend terbaru:** `13.9% → 10.1% → 18.2% → 20.9% → 24.3% → 24.3%`

Riwayat lengkap tersimpan di [`research_progress_history.json`](research_progress_history.json).

> Bagian ini dikelola otomatis oleh GitHub Actions. Nilai dapat turun bila scope/kriteria penelitian bertambah; tracker tidak memaksa progress selalu meningkat.
<!-- RESEARCH_PROGRESS_END -->

Proyek riset quantum machine learning yang memperluas skripsi sarjana (FRD-09, arsitektur **MORE-HD**) menjadi eksperimen faktorial yang lebih ketat secara metodologis: **MORE-HD-C**.

## Latar Belakang

**MORE-HD** (*Multi-Readout Qubit Expansion for High-Dimensional Label Space*) adalah Variational Quantum Circuit yang memperluas kerangka MORE (Wu dkk., 2023) menggunakan 2 qubit readout untuk memperbesar ruang label kuantum dari 3D menjadi 15D, diterapkan pada klasifikasi multi-kelas MNIST.

- 10 qubit total (8 data qubit + 2 readout qubit)
- 30 parameter terlatih (gerbang RY + CNOT), 3.03× lebih sedikit dibanding MORE (91 parameter)
- Unggul ~9.74 poin rata-rata pada tugas 3–7 kelas, tetapi menurun performanya pada 8 dan 10 kelas

**Temuan akar masalah:** 6 dari 15 dimensi observable MORE-HD selalu nol secara struktural, karena kombinasi gerbang RY + CNOT hanya menghasilkan state kuantum bernilai riil (real subspace dari Hilbert space). Ini mengurangi dimensionalitas efektif dan berkontribusi pada kegagalan klasifikasi di jumlah kelas tinggi — fenomena yang disebut *curse of density* (adaptasi dari istilah "crowded quantum labels" milik Wu dkk.).

## Kontribusi MORE-HD-C

MORE-HD-C mengatasi keterbatasan struktural tersebut dengan **modifikasi V1**: menambahkan gerbang RZ setelah RY pada layer variational (RY+RZ per qubit, bukan RY saja). RZ menghilangkan *real-state restriction* sehingga keenam observable Y-odd yang tadinya nol secara struktural **dapat** aktif; aktivasi tersebut tetap harus dibuktikan secara empiris dan tidak dijamin hanya oleh keberadaan RZ.

- 60 parameter terlatih (2× MORE-HD), masih < 91 parameter MORE.
- Encoding, entangling block (CNOT), jumlah qubit, readout, dan urutan pengukuran 15 observable identik dengan MORE-HD.
- Perbandingan utama MORE-HD (30 parameter) versus MORE-HD-C (60 parameter) **tidak cukup untuk mengatribusikan perubahan performa hanya kepada fase kompleks**, karena jumlah parameter terlatih berubah bersamaan dengan ansatz. Karena itu, eksperimen utama dilengkapi targeted ablation G1-05/G1-07 untuk memisahkan efek parameter budget dan akses ke state kompleks.

## Desain Eksperimen: Faktorial Utama 2 × 3 × 8

Eksperimen utama membandingkan:

| Faktor | Level |
|---|---|
| **A — Arsitektur** | MORE-HD (RY saja) vs MORE-HD-C (RY+RZ) |
| **B — Metode fitur** | PCA, Hu Moment Invariants, Zernike Moments |
| **C — Jumlah kelas** | K = 3 sampai K = 10 (kumulatif: `[0..K-1]`) |

```
8 skenario kelas × 2 arsitektur × 3 metode fitur = 48 kondisi per seed
48 kondisi × 5 confirmatory seeds (101, 202, 303, 404, 505) = 240 confirmatory runs
Seed 42 = PILOT ONLY dan tidak masuk agregasi hasil final
```

Semua metode fitur dikunci menghasilkan tepat 8 channel (sama dengan 8 data qubit) agar arsitektur sirkuit, topologi entanglement, dan prosedur evaluasi tidak berubah — sehingga faktor `feature_method` murni menguji efek representasi fitur terhadap pipeline clustering + classification.

### Targeted Ablation G1-05/G1-07

Untuk memisahkan efek jumlah parameter dari efek akses ke state kompleks, ablation hanya dijalankan pada `K={3,6,10}`, seluruh feature method, dan lima confirmatory seed yang sama. Model A dan D direuse dari hasil eksperimen utama sehingga tidak dieksekusi dua kali; hanya model B dan C yang menambah run baru.

| Kode | Model | State | Parameter terlatih | Peran |
|---|---|---|---:|---|
| A | MORE-HD | real | 30 | baseline utama; reuse dari 240 run |
| B | MORE-HD-60P | real | 60 | parameter-budget / real-capacity control |
| C | MORE-HD-C-FixedRZ | kompleks dapat diakses | 30 RY; 30 RZ fixed non-zero | complex-state control pada trainable budget 30 |
| D | MORE-HD-C | kompleks dapat diakses | 60 (30 RY + 30 RZ) | model utama; reuse dari 240 run |

```
3 K × 3 feature × 2 model tambahan (B,C) = 18 kondisi ablation per seed
18 × 5 confirmatory seeds = 90 additional ablation runs
240 primary + 90 ablation = 330 unique confirmatory executions
```

A/D dan B/C memakai split serta pair manifest yang sama pada cell yang berpasangan. Primary A/D juga memakai structured paired initialization sehingga reuse A/D tetap valid untuk ablation. **Source of truth teknis ablation** sekarang berada langsung di `Pseudocode_2x3_manual_runs.md` Bagian 11.

### Pipeline Dua Fase

1. **Clustering (unsupervised)** — mengikuti konsep pairing MORE: tepat 5 sampel TRAIN per kelas dipilih secara deterministik tanpa replacement, lalu seluruh unordered unique pairs (`i < j`) digunakan sekali tanpa balancing/reweighting. Pair set dibekukan sebelum COBYLA dan disimpan melalui `pair_manifest.json` + `pair_stats.json`. COBYLA meminimalkan `train_loss` berbasis cosine distance berbobot matriks korelasi antar-kelas; monitoring memakai validation secara pasif dan official test tidak diakses selama optimasi.
2. **Supervised** — fine-tuning terhadap label kuantum (centroid ternormalisasi) hasil fase clustering. Optimasi menggunakan train; monitoring selama pengembangan menggunakan validation. Official test hanya dipanggil sekali melalui `FINAL_EVALUATION` setelah seluruh keputusan run dibekukan.

Tersedia dua jalur eksekusi:
- **Jalur A (otomatis)** — memakai `result.x` sebagai **final point resmi COBYLA**; `best_observed_point` dicatat terpisah dan tidak diasumsikan sama dengan `result.x`.
- **Jalur B (deterministic secondary path)** — dijalankan setelah run Jalur A selesai dan menggunakan artefak clustering run sumber. Checkpoint dipilih otomatis dengan aturan lexicographic train/validation yang dibekukan (`pseudo_accuracy_val` → `min_separation_val` → `mean_true_distance_val` → `train_loss` → `eval_id`), sementara `active_dimensions` hanya diagnostik. Budget supervised diwarisi dari Jalur A dan official test baru digunakan pada `FINAL_EVALUATION`. Jalur B tidak mengganti hasil primer `result.x` dan tidak termasuk hitungan 330 primary+ablation executions.

## Tools & Stack

- **PennyLane** — konstruksi dan simulasi sirkuit kuantum
- **SciPy (COBYLA)** — optimasi clustering dan supervised
- **MNIST** — dataset utama
- **PCA / Hu Moment Invariants / Zernike Moments** — tiga metode reduksi fitur yang dibandingkan
- Log crash-safe: JSONL untuk metrik, binary append-only fixed-record untuk parameter (menghindari risiko korupsi `.npz`)

## Status Proyek

Proyek berada pada tahap **penguncian desain metodologis** sebelum implementasi kode dan eksperimen final dijalankan. Seluruh keputusan desain dikontrol lewat dokumen *Research Readiness Gates*. Matriks utama memiliki 48 kondisi per seed dan 240 confirmatory runs pada lima seed yang sudah ditetapkan. Targeted ablation G1-05/G1-07 menambah 90 run baru pada K={3,6,10}, sehingga total rencana menjadi **330 unique confirmatory executions**. Seed 42 tetap hanya untuk pilot. Eksperimen konfirmatori baru boleh dimulai setelah Gate G0 dan G1 berstatus `CLOSED` dan item teknis yang memengaruhi hasil telah diverifikasi.

<!-- READINESS_SUMMARY_START -->
### Ringkasan Kesiapan (Readiness Gate) — per 2026-09-27

| Gate | Cakupan | Item | Closed | Ready for Verification | In Progress | Pilot Only | Blocked | Kesiapan* |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | Validitas data & evaluasi | 4 | 2 | 0 | 1 | 0 | 1 | **62.5%** |
| G1 | Optimasi & desain eksperimen | 11 | 0 | 1 | 5 | 1 | 4 | **31.8%** |
| G2 | Preprocessing & definisi numerik | 6 | 0 | 0 | 3 | 0 | 3 | **25.0%** |
| G3 | Artefak & spreadsheet | 7 | 0 | 1 | 1 | 0 | 5 | **17.9%** |
| G4 | Reproducibility & verifikasi sirkuit | 4 | 0 | 0 | 0 | 0 | 4 | **0.0%** |
| D | Ketidakkonsistenan dokumen (BAB 1, FRD-09, dll.) | 6 | 0 | 0 | 1 | 0 | 5 | **8.3%** |
| **Total** |  | **38** | **2** | **2** | **11** | **1** | **22** | **24.3%** |

\* Kesiapan dihitung sebagai rata-rata bobot per item: `CLOSED=100%`, `READY FOR VERIFICATION=75%`, `IN PROGRESS=50%`, `PILOT ONLY=25%`, dan `BLOCKED=0%`.
<!-- READINESS_SUMMARY_END -->

**Yang sudah dikunci (2026-09-22):** desain ekstraksi Hu Moments (input grayscale, formula signed-log, padding channel ke-8 = 0.0) dan Zernike Moments (8 pasangan `(n,m)` revisi, pemetaan unit disk, magnitude invarian rotasi), serta kebijakan clipping seragam (tidak ada clipping untuk PCA/Hu/Zernike). Ketiganya masih `IN PROGRESS` karena implementasi kode nyata + unit test belum dikerjakan.

**Prasyarat sebelum eksperimen konfirmatori final boleh dijalankan (Gate C):**
- Seluruh item G0 dan G1 harus memenuhi kriteria Gate C; status aktual mengikuti blok readiness otomatis di atas.
- Item teknis G2–G4 yang memengaruhi hasil berstatus `CLOSED`.
- Protokol, daftar seed, split data, budget optimizer, primary analysis plan, dan targeted ablation plan telah dibekukan.
- Workbook master utama, workbook ablation, dan konsolidator hasil telah diuji sebelum dipakai untuk hasil final.

Smoke test dengan 10 evaluasi COBYLA dan seed 42 diperbolehkan sebagai `PILOT ONLY`, tetapi hasilnya **tidak boleh** diperlakukan sebagai hasil konfirmatori publikasi.

## Referensi

Wu et al. (2023), McClean et al. (2018), Cerezo et al. (2021), Holmes et al. (2022), Schuld et al. (2021).