# MORE-HD-COMPLEX

<!-- RESEARCH_PROGRESS_START -->
## Research Progress

![Research Progress](https://img.shields.io/badge/Research%20Progress-20.9%25-blue)

**Current research readiness: 20.9%**

`████░░░░░░░░░░░░░░░░ 20.9%`

Progress dihitung langsung dari **37 item Research Readiness Gate** dengan bobot:
`CLOSED = 100%`, `READY FOR VERIFICATION = 75%`, `IN PROGRESS = 50%`, `PILOT ONLY = 25%`, dan `BLOCKED = 0%`.

### Progress History

| Tanggal | Commit acuan | Progress | Perubahan | Ringkasan |
|---|---|---:|---:|---|
| 2026-09-22 | [`5a26fdd`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/5a26fdd13d09123de634997b62e6bf75dfee840f) | 13.9% | — | Baseline readiness setelah protokol ukuran dataset dikunci. |
| 2026-09-23 | [`34e51ec`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/34e51ecdbc049f296113fbfa282d688b93ceebbe) | 10.1% | -3.8 pp | Scope diperketat dengan penambahan 10 item readiness baru; penurunan mencerminkan denominator/kriteria yang bertambah, bukan hilangnya pekerjaan selesai. |
| 2026-09-26 | [`6f8c7c1`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/6f8c7c1f04b9d29e00f36bb6aa603a8681fd601b) | 18.2% | +8.1 pp | Protokol multi-seed, analysis plan, workbook 240-run, dan logging objective mulai terkunci/terverifikasi. |
| 2026-09-27 | [`5aeb1ca`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/5aeb1cabe3e55673b51b0cf09e4c0865ba79f2f4) | **20.9%** | **+2.7 pp** | Protokol deterministic clustering pairs dan dokumentasi 240-run diselaraskan. |

**Trend:** `13.9% → 10.1% → 18.2% → 20.9%`

> Mulai titik ini, setiap commit penelitian yang mengubah status readiness gate dapat menambahkan satu baris baru. Persentase tidak dipaksa selalu naik: bila scope penelitian diperketat atau item wajib baru ditambahkan, nilai dapat turun agar indikator tetap metodologis dan tidak sekadar kosmetik.
<!-- RESEARCH_PROGRESS_END -->

Proyek riset quantum machine learning yang memperluas skripsi sarjana (FRD-09, arsitektur **MORE-HD**) menjadi eksperimen faktorial yang lebih ketat secara metodologis: **MORE-HD-C**.

## Latar Belakang

**MORE-HD** (*Multi-Readout Qubit Expansion for High-Dimensional Label Space*) adalah Variational Quantum Circuit yang memperluas kerangka MORE (Wu dkk., 2023) menggunakan 2 qubit readout untuk memperbesar ruang label kuantum dari 3D menjadi 15D, diterapkan pada klasifikasi multi-kelas MNIST.

- 10 qubit total (8 data qubit + 2 readout qubit)
- 30 parameter terlatih (gerbang RY + CNOT), 3.03× lebih sedikit dibanding MORE (91 parameter)
- Unggul ~9.74 poin rata-rata pada tugas 3–7 kelas, tetapi menurun performanya pada 8 dan 10 kelas

**Temuan akar masalah:** 6 dari 15 dimensi observable MORE-HD selalu nol secara struktural, karena kombinasi gerbang RY + CNOT hanya menghasilkan state kuantum bernilai riil (real subspace dari Hilbert space). Ini mengurangi dimensionalitas efektif dan berkontribusi pada kegagalan klasifikasi di jumlah kelas tinggi — fenomena yang disebut *curse of density* (adaptasi dari istilah "crowded quantum labels" milik Wu dkk.).

## Kontribusi MORE-HD-C

MORE-HD-C mengatasi kelemahan struktural tersebut dengan **modifikasi V1**: menambahkan gerbang RZ setelah RY pada layer variational (RY+RZ per qubit, bukan RY saja), yang mengaktifkan dimensi imajiner Hilbert space sehingga keenam observable Y-odd yang tadinya nol berpotensi aktif.

- 60 parameter terlatih (2× MORE-HD), masih < 91 parameter MORE — narasi efisiensi tetap terjaga
- Encoding, entangling block (CNOT), dan urutan pengukuran 15 observable **identik** dengan MORE-HD — satu-satunya perbedaan ada di layer variational, agar efek RZ bisa diatribusikan secara bersih

## Desain Eksperimen: Faktorial 2 × 3 × 8

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

### Pipeline Dua Fase

1. **Clustering (unsupervised)** — mengikuti konsep pairing MORE: tepat 5 sampel TRAIN per kelas dipilih secara deterministik tanpa replacement, lalu seluruh unordered unique pairs (`i < j`) digunakan sekali tanpa balancing/reweighting. Pair set dibekukan sebelum COBYLA dan disimpan melalui `pair_manifest.json` + `pair_stats.json`. COBYLA meminimalkan `train_loss` berbasis cosine distance berbobot matriks korelasi antar-kelas; monitoring memakai validation secara pasif dan official test tidak diakses selama optimasi.
2. **Supervised** — fine-tuning terhadap label kuantum (centroid ternormalisasi) hasil fase clustering, dengan train/test loss dipantau langsung.

Tersedia dua jalur eksekusi:
- **Jalur A (otomatis)** — memakai `result.x` COBYLA (iterasi dengan train_loss terbaik) sebagai output clustering.
- **Jalur B (manual)** — memungkinkan pemilihan iterasi clustering tertentu (berdasarkan `clustering_log.jsonl`) sebagai alternatif, untuk kasus di mana train_loss terbaik ≠ struktur label kuantum terbaik.

## Tools & Stack

- **PennyLane** — konstruksi dan simulasi sirkuit kuantum
- **SciPy (COBYLA)** — optimasi clustering dan supervised
- **MNIST** — dataset utama
- **PCA / Hu Moment Invariants / Zernike Moments** — tiga metode reduksi fitur yang dibandingkan
- Log crash-safe: JSONL untuk metrik, binary append-only fixed-record untuk parameter (menghindari risiko korupsi `.npz`)

## Status Proyek

Proyek berada pada tahap **penguncian desain metodologis** sebelum implementasi kode dan eksperimen final dijalankan. Seluruh keputusan desain dikontrol lewat dokumen *Research Readiness Gates*. Matriks utama memiliki 48 kondisi per seed dan 240 confirmatory runs pada lima seed yang sudah ditetapkan; seed 42 hanya untuk pilot. Eksperimen konfirmatori baru boleh dimulai setelah Gate G0 dan G1 berstatus `CLOSED`.

### Ringkasan Kesiapan (Readiness Gate) — per 2026-09-27

| Gate | Cakupan | Item | Closed | Ready for Verification | In Progress | Pilot Only | Blocked | Kesiapan* |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | Validitas data & evaluasi | 4 | 2 | 0 | 1 | 0 | 1 | **62.5%** |
| G1 | Optimasi & desain eksperimen | 11 | 0 | 1 | 3 | 1 | 6 | **22.7%** |
| G2 | Preprocessing & definisi numerik | 6 | 0 | 0 | 3 | 0 | 3 | **25.0%** |
| G3 | Artefak & spreadsheet | 6 | 0 | 1 | 0 | 0 | 5 | **12.5%** |
| G4 | Reproducibility & verifikasi sirkuit | 4 | 0 | 0 | 0 | 0 | 4 | **0%** |
| D | Ketidakkonsistenan dokumen (BAB 1, FRD-09, dll.) | 6 | 0 | 0 | 1 | 0 | 5 | **8.3%** |
| **Total** |  | **37** | **2** | **2** | **8** | **1** | **24** | **20.9%** |

\* Kesiapan dihitung sebagai rata-rata bobot per item: `CLOSED=100%`, `READY FOR VERIFICATION=75%`, `IN PROGRESS=50%`, `PILOT ONLY=25%`, dan `BLOCKED=0%`.

**Yang sudah dikunci (2026-09-22):** desain ekstraksi Hu Moments (input grayscale, formula signed-log, padding channel ke-8 = 0.0) dan Zernike Moments (8 pasangan `(n,m)` revisi, pemetaan unit disk, magnitude invarian rotasi), serta kebijakan clipping seragam (tidak ada clipping untuk PCA/Hu/Zernike). Ketiganya masih `IN PROGRESS` karena implementasi kode nyata + unit test belum dikerjakan.

**Prasyarat sebelum 48 run final boleh dijalankan (Gate C):**
- Seluruh item G0 dan G1 berstatus `CLOSED` (saat ini 0 dari 9 item G0+G1 closed).
- Item teknis G2–G4 yang memengaruhi hasil berstatus `CLOSED`.
- Protokol, daftar seed, split data, budget optimizer, dan rencana analisis statistik telah dibekukan.
- Workbook master dan konsolidator hasil telah diuji.

Smoke test dengan 10 evaluasi COBYLA dan seed 42 diperbolehkan sebagai `PILOT ONLY`, tetapi hasilnya **tidak boleh** diperlakukan sebagai hasil konfirmatori publikasi.

## Referensi

Wu et al. (2023), McClean et al. (2018), Cerezo et al. (2021), Holmes et al. (2022), Schuld et al. (2021).