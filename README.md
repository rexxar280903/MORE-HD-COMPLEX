# MORE-HD-COMPLEX

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
8 skenario kelas × 2 arsitektur × 3 metode fitur = 48 run per seed
```

Semua metode fitur dikunci menghasilkan tepat 8 channel (sama dengan 8 data qubit) agar arsitektur sirkuit, topologi entanglement, dan prosedur evaluasi tidak berubah — sehingga faktor `feature_method` murni menguji efek representasi fitur terhadap pipeline clustering + classification.

### Pipeline Dua Fase

1. **Clustering (unsupervised)** — COBYLA meminimalkan `train_loss` berbasis cosine distance berbobot matriks korelasi antar-kelas; 6 metrik tambahan (`pseudo_accuracy_test`, `avg_margin_test`, `min_separation`, `correlation_consistency`, `active_dimensions`) dicatat pasif per iterasi untuk analisis pasca-training, bukan sinyal optimasi.
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

Proyek berada pada tahap **penguncian desain metodologis** sebelum implementasi kode dan eksperimen final dijalankan. Seluruh keputusan desain dikontrol lewat dokumen *Research Readiness Gates* dan hanya boleh maju ke eksperimen final (48 run) setelah Gate G0 dan G1 berstatus `CLOSED`.

### Ringkasan Kesiapan (Readiness Gate) — per 2026-09-22

| Gate | Cakupan | Item | Closed | In Progress | Pilot Only | Blocked | Kesiapan* |
|---|---|---|---|---|---|---|---|
| G0 | Validitas data & evaluasi | 4 | 0 | 0 | 0 | 4 | **0%** |
| G1 | Optimasi & desain eksperimen | 5 | 0 | 0 | 1 | 4 | **5%** |
| G2 | Preprocessing & definisi numerik | 5 | 0 | 3 | 0 | 2 | **30%** |
| G3 | Artefak & spreadsheet | 4 | 0 | 0 | 0 | 4 | **0%** |
| G4 | Reproducibility & verifikasi sirkuit | 4 | 0 | 0 | 0 | 4 | **0%** |
| D | Ketidakkonsistenan dokumen (BAB 1, FRD-09, dll.) | 5 | 0 | 0 | 0 | 5 | **0%** |
| **Total** | | **27** | **0** | **3** | **1** | **23** | **≈6.5%** |

\* Kesiapan dihitung sebagai rata-rata bobot per item: `CLOSED=100%`, `IN PROGRESS=50%`, `PILOT ONLY=25%` (boleh smoke test, tidak boleh jadi hasil final), `BLOCKED=0%`.

**Yang sudah dikunci (2026-09-22):** desain ekstraksi Hu Moments (input grayscale, formula signed-log, padding channel ke-8 = 0.0) dan Zernike Moments (8 pasangan `(n,m)` revisi, pemetaan unit disk, magnitude invarian rotasi), serta kebijakan clipping seragam (tidak ada clipping untuk PCA/Hu/Zernike). Ketiganya masih `IN PROGRESS` karena implementasi kode nyata + unit test belum dikerjakan.

**Prasyarat sebelum 48 run final boleh dijalankan (Gate C):**
- Seluruh item G0 dan G1 berstatus `CLOSED` (saat ini 0 dari 9 item G0+G1 closed).
- Item teknis G2–G4 yang memengaruhi hasil berstatus `CLOSED`.
- Protokol, daftar seed, split data, budget optimizer, dan rencana analisis statistik telah dibekukan.
- Workbook master dan konsolidator hasil telah diuji.

Smoke test dengan 10 evaluasi COBYLA dan seed 42 diperbolehkan sebagai `PILOT ONLY`, tetapi hasilnya **tidak boleh** diperlakukan sebagai hasil konfirmatori publikasi.

## Referensi

Wu et al. (2023), McClean et al. (2018), Cerezo et al. (2021), Holmes et al. (2022), Schuld et al. (2021).