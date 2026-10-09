# Catatan Perbaikan dan Verifikasi MADBase

**Tanggal:** 2026-10-09 (Asia/Jakarta)  
**Cakupan:** kesiapan training MORE-HD/MORE-HD-C menggunakan dataset angka
Arabic-Indic MADBase  
**Kesimpulan:** `SIAP UNTUK PILOT`; run `CONFIRMATORY` sengaja dikunci sampai
budget optimizer dari pilot konvergensi MADBase dibekukan.

Dokumen ini mencatat perubahan yang dilakukan, alasan, bukti verifikasi, dan
pekerjaan yang masih wajib sebelum eksperimen final. Metrik smoke di bawah
hanya bukti teknis dan bukan hasil penelitian.

## 1. Kondisi awal

Data MADBase sudah lengkap dan valid, tetapi jalur training masih terikat ke
MNIST. Masalah utama yang ditemukan:

- tidak ada pilihan dataset eksplisit pada CLI/config;
- split, ID sampel, artefak, workbook, baseline, dan analysis masih membawa
  asumsi/nama MNIST;
- official test dapat tersedia terlalu awal dalam objek data;
- belum ada split writer-aware dan fixed official-test subset untuk MadBase;
- belum ada guard agar budget optimizer MNIST tidak dipakai pada MadBase;
- dokumen riset masih berstatus draft dengan keputusan penting belum terisi;
- belum ada bukti end-to-end untuk lima arsitektur dan jalur hilir.

## 2. Validasi dataset

Snapshot yang digunakan adalah `MADBASE-AUC-V1`:

| Artefak | Ukuran | SHA-256 |
|---|---:|---|
| `MAHDBase_TrainingSet.rar` | 31.184.713 byte | `ae2d5bcd658320eaf054de1ba46aac187a0470830d685258a0a2b0659c8d2eb8` |
| `MAHDBase_TestingSet.rar` | 5.487.352 byte | `a8294a266f1843b4c659291dfda6e71620abbfcaec47e4313024c691eb494b1d` |

Hasil audit:

- 70.000/70.000 BMP dapat dibaca;
- official train/test = 60.000/10.000;
- semua citra 28×28, grayscale, label 0..9;
- setiap kelas seimbang: 6.000 train dan 1.000 test;
- writer train 1–600 dan test 601–700, overlap 0;
- mapping glif diverifikasi sebagai Arabic-Indic `٠١٢٣٤٥٦٧٨٩`;
- exact/near duplicates diaudit. Kebijakan beku adalah mempertahankan snapshot
  resmi dan melaporkan flag, tanpa eksklusi diam-diam.

Snapshot dataset internal:
`8262dca2d3db6205e0112586dd9a02c35b0b371da292aab0a15dc92976793f78`.

## 3. Kebijakan lisensi dan distribusi

Halaman resmi AUC menyediakan data, tetapi tidak menampilkan teks lisensi
redistribusi eksplisit. Karena itu kebijakan yang diterapkan adalah:

```text
LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION
```

Arsip, BMP, cache array, dan contoh mentah tetap lokal serta diabaikan Git.
Kode, checksum, manifest ID, konfigurasi, dan hasil turunan non-rekonstruktif
dapat disimpan. Kebijakan ini bukan pernyataan bahwa lisensi formal telah
diberikan; izin tertulis tetap diperlukan sebelum redistribusi raw.

## 4. Perubahan implementasi

### 4.1 Pemilihan dataset eksplisit

- Menambahkan registry `MNIST`/`MADBASE` di `core/datasets.py`.
- Menambahkan `--dataset {MNIST,MADBASE}` ke entry point.
- Menyimpan dataset ID dalam config, run UID, manifest, dan report.
- Run MadBase dipisahkan di `runs/arabic_digits/`.
- Mismatch dataset/config/split ditolak, sehingga data MNIST dan MadBase tidak
  dapat tercampur tanpa terdeteksi.

File utama: `core/constants.py`, `core/config.py`, `core/cli.py`,
`core/datasets.py`, `core/pipeline.py`.

### 4.2 Official-test isolation

- Menambahkan `MADBaseTrainingView` yang tidak mempunyai atribut
  `test_images`/`test_labels`.
- Loader training hanya membuka official training pool.
- Pilot memakai writer-disjoint holdout dari training pool.
- Official test konfirmatori dimuat lazy hanya di `final_evaluation`.
- Vault memblokir akses dini dan mencatat status pemakaian test.

File utama: `core/madbase.py`, `core/datasets.py`, `core/data_pipeline.py`,
`core/evaluation.py`, `core/pipeline.py`.

### 4.3 Split writer-aware dan pairing

Manifest yang dibuat:

- `splits/pilot_seed42.json`;
- `splits/confirmatory_seed101.json`;
- `splits/confirmatory_seed202.json`;
- `splits/confirmatory_seed303.json`;
- `splits/confirmatory_seed404.json`;
- `splits/confirmatory_seed505.json`;
- `splits/split_registry.json`.

Kontrak split:

- 1000 train, 100 validation, 200 final-test samples per kelas;
- train = 100 writer dan validation = 10 writer berbeda;
- pilot test = 20 writer lain dari official training pool;
- confirmatory test = 20 writer official-test tetap;
- official-test sample IDs identik pada semua seed, dengan hash
  `fece765dd5f7e16f51d53c2fa9038ba3279d0ec0459060298e26c873bf1d9a2e`;
- class regime kumulatif/nested K=3..10.

File utama: `core/splits.py`, `scripts/build_madbase_splits.py`,
`core/pairing.py`.

### 4.4 Fitur dan artefak generik

- PCA, Hu, dan Zernike memakai geometri 28×28 dan menghasilkan delapan kanal.
- PCA dan scaler fit hanya pada train.
- Artefak memakai `sample_id` dan `source_index`; alias `mnist_index` hanya
  dipertahankan pada jalur legacy MNIST.
- Dataset, snapshot, test origin, dan input/transformer hashes disimpan.
- Jalur B dan baseline memakai identitas/hashes yang sama dengan source run.

File utama: `core/data_pipeline.py`, `core/workbook.py`, `core/jalur_b.py`,
`core/classical_baselines.py`, `main_classical_baseline.py`.

### 4.5 Workbook, konsolidasi, dan analisis

Tiga template MadBase dibuat dan divalidasi:

- `research_data/MORE_HD_master_confirmatory_240runs.xlsx`;
- `research_data/MORE_HD_master_ablation_90runs.xlsx`;
- `research_data/MORE_HD_master_more_reference_120runs.xlsx`.

Consolidator dan analysis menerima filter dataset/run mode/pilot tag. Pilot
smoke berhasil dikonsolidasikan ke `research_data/consolidated_pilot/` dan
dianalisis ke `research_data/analysis_pilot/`, termasuk primary pairing,
ablation, MORE reference, McNemar, dan baseline comparison.

File utama: `scripts/build_madbase_workbook_templates.py`,
`scripts/consolidate_runs.py`, `analysis/run_analysis.py`.

### 4.6 Guard budget konfirmatori

Budget optimizer MNIST tidak diwariskan. Untuk MadBase, CLI menolak semua run
`CONFIRMATORY` sampai file berikut ada, menyebut dataset `MADBASE`, dan
berstatus `FROZEN`:

```text
research_arabic_digits/research_data/pilot/convergence_budget_decision.json
```

Nilai dibuat oleh `scripts/analyze_pilot_budget.py --dataset MADBASE` setelah
matriks pilot konvergensi selesai. Guard ini adalah perlindungan protokol,
bukan bug yang harus dilewati.

## 5. Verifikasi yang dijalankan

### 5.1 Test suite

Environment verifikasi: Python 3.13.15, pytest 9.1.1, Qiskit 2.5.2, NumPy
2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, PennyLane 0.45.1, Pillow 12.3.0.

Hasil final setelah seluruh perubahan kode dan dokumentasi:

```text
69 passed, 11 skipped, 0 failed in 19.28s
```

Sebelas skip adalah integration test MNIST yang membutuhkan raw snapshot MNIST
lokal yang tidak terpasang. Test MadBase dan cross-check Qiskit tetap berjalan.

Test baru mencakup writer-disjoint split, fixed confirmatory identities,
train-only feature fitting, finite outputs, early vault access rejection,
dataset mismatch rejection, dan training view tanpa official test arrays.

### 5.2 Smoke lima arsitektur

Semua run memakai seed 42, PCA, K=3, 20/6/8 per kelas, dan 100 NFEV:

| Arsitektur | Status | Accuracy pilot |
|---|---|---:|
| MORE-HD | `COMPLETED` | 0,8750 |
| MORE-HD-C | `COMPLETED` | 0,9167 |
| MORE-HD-60P | `COMPLETED` | 0,9167 |
| MORE-HD-C-FixedRZ | `COMPLETED` | 0,6667 |
| MORE-REPRO | `COMPLETED` | 0,2917 |

Seluruh run memiliki paired identity yang sama, non-primary pairing `MATCH`,
dan semua objective-evaluation IDs mengacu ke train. Accuracy tersebut hanya
smoke output dan tidak boleh dipakai sebagai kesimpulan model.

Hu dan Zernike juga melewati smoke terpisah. Workbook smoke Zernike berisi
summary/config yang cocok dan 24 prediction rows.

### 5.3 Jalur hilir

- Dua baseline fit selesai: Nearest Centroid dan Logistic Regression.
- Jalur B selesai dengan selected eval ID deterministik.
- Consolidator menerima 2 primary, 2 ablation, 1 MORE reference, 1 Jalur B,
  dan baseline CSV.
- Analysis membaca 5 run dan menghasilkan tujuh tabel analisis pilot.
- Percobaan memulai `CONFIRMATORY` tanpa decision file berhenti sebelum data
  atau optimizer dijalankan, dengan pesan untuk menyelesaikan convergence pilot.

## 6. Dokumentasi dan provenance

- `PROTOCOL_FREEZE.json` berisi kontrak mesin-terbaca.
- `requirements-lock-py313.txt` mencatat versi environment.
- `research_data/manifests/readiness_report.json` mencatat status readiness.
- `research_data/pilot/test_summary.json` mencatat hasil test.
- `research_data/pilot/smoke_check.json` mencatat pemeriksaan smoke.
- `README.md`, `RUNBOOK.md`, `READINESS_GATES.md`, `DATASET_CARD.md`,
  `RESEARCH_PROTOCOL.md`, `STATISTICAL_ANALYSIS_PLAN.md`, dan
  `DECISION_LOG.md` telah diselaraskan.

Perubahan pengguna yang sudah ada di working tree tidak dihapus atau di-reset.

## 7. Status akhir dan pekerjaan tersisa

### Siap sekarang

- audit/rebuild manifests;
- unit/integration tests;
- smoke dan timing pilot;
- baseline, Jalur B, consolidation, dan analysis pilot;
- pilot konvergensi MadBase.

### Belum diizinkan

- 450 run Jalur A konfirmatori;
- 90 run Jalur B final;
- interpretasi/publikasi metrik final.

Satu-satunya blocker pra-konfirmatori adalah pilot konvergensi MadBase. Jalankan:

```powershell
python scripts/run_pilot.py convergence --dataset MADBASE --first-only
python scripts/run_pilot.py convergence --dataset MADBASE --skip-first --parallel 1
python scripts/analyze_pilot_budget.py --dataset MADBASE --tag convergence
python scripts/check_arabic_readiness.py --require-confirmatory
```

Tahap tersebut panjang dan menghasilkan bukti empiris yang tidak boleh
direkayasa atau diganti oleh budget studi MNIST.
