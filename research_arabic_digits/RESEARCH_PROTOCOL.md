# Protokol Penelitian — MORE-HD-C pada MADBase

**Status:** `FROZEN FOR PILOT / CONFIRMATORY PENDING OPTIMIZER BUDGET`  
**Versi:** 1.0  
**Tanggal:** 2026-10-09

Protokol ini berlaku untuk dataset `MADBASE-AUC-V1`. Seluruh keputusan selain
budget optimizer konfirmatori telah dibekukan sebelum ada metrik model dari
official test. Budget konfirmatori hanya boleh diisi oleh keluaran pilot
konvergensi MADBase yang mengikuti aturan plateau pra-spesifikasi.

## 1. Tujuan dan estimand

Tujuan primer adalah mengestimasi selisih berpasangan MORE-HD-C (D) minus
MORE-HD (A) pada accuracy, macro-F1, dan minimum quantum-label separation ratio
untuk feature method, K, seed, split, dan final-test instances yang sama.

Analisis ablation memisahkan efek jumlah parameter dan akses state kompleks.
MORE-REPRO serta baseline klasik memberi konteks. Perbandingan dengan MNIST
hanya deskriptif karena domain, penulis, dan akuisisinya tidak berpasangan.

## 2. Dataset dan sampel

| Komponen | Keputusan beku |
|---|---|
| Dataset | Modified Arabic Digits Database, `MADBASE-AUC-V1` |
| Snapshot | `8262dca2d3db6205e0112586dd9a02c35b0b371da292aab0a15dc92976793f78` |
| Unit observasi | satu citra digit tulisan tangan 28×28 |
| Label | internal 0..9; glif Arabic-Indic `٠١٢٣٤٥٦٧٨٩` |
| Official source split | 60.000 train / 10.000 test; writer 1–600 / 601–700 |
| Ukuran studi per kelas | train 1000, validation 100, final test 200 |
| Seed pilot | 42 |
| Seed konfirmatori | 101, 202, 303, 404, 505 |
| K | kumulatif/nested 3, 4, 5, 6, 7, 8, 9, 10 |

Training dan validation dibuat writer-disjoint dari official training pool.
Pilot memakai kelompok writer ketiga dari pool yang sama. Konfirmatori memakai
20 writer official-test yang dibekukan dan sama untuk seluruh seed. Dengan
demikian variasi seed mengukur training/split-train variation, bukan pergantian
test instances.

Kandidat exact/near duplicate tidak dihapus: snapshot resmi dipertahankan dan
flag audit dilaporkan. Tidak boleh ada eksklusi berdasarkan performa model.

## 3. Isolasi final test

Objek training MADBase tidak mengekspos `test_images` atau `test_labels`.
Official test dimuat secara lazy hanya di dalam tahap `final_evaluation` dan
dilindungi vault. Clustering, training, hyperparameter selection, convergence,
dan pilot tidak boleh mengakses outcome official test.

## 4. Model dan matriks eksperimen

| Track | Model | Kondisi | Run konfirmatori |
|---|---|---|---:|
| Primer | A = MORE-HD, D = MORE-HD-C | 3 fitur × 8 K × 5 seed | 240 |
| Ablation | B = MORE-HD-60P, C = MORE-HD-C-FixedRZ | 3 fitur × K {3,6,10} × 5 seed | 90 |
| Referensi | M = MORE-REPRO | 3 fitur × 8 K × 5 seed | 120 |
| Jalur B sekunder | A dan D dari checkpoint terpilih deterministik | 3 fitur × K {3,6,10} × 5 seed | 90 |

Total Jalur A adalah 450 eksekusi unik. Sirkuit, encoding, jumlah kanal, loss,
observable, dan backend tidak berubah karena dataset. Semua run memakai
simulasi statevector analitik complex128 tanpa shots.

## 5. Representasi fitur

- PCA: delapan komponen, full SVD, fit hanya pada train.
- Hu: tujuh Hu moments grayscale, signed-log, kanal kedelapan bernilai 0.
- Zernike: delapan term beku dari `core/constants.py`, magnitude pada geometri
  28×28.
- Scaling: state fit train-only menuju rentang `[0, π]`, tanpa clipping.
- Resize, crop, centring, dan augmentasi: tidak dilakukan.

Setiap array, transformer, scaler, sample list, dan manifest disimpan dengan
provenance dan SHA-256. Nilai non-finite dan zero norm ditangani oleh kontrak
numerik yang diuji; kejadian dicatat, tidak disembunyikan.

## 6. Pilot dan budget

Smoke memakai 20/6/8 sampel per kelas dan 100 NFEV untuk menguji jalur kode.
Pilot konvergensi memakai ukuran penuh studi, seed 42, kondisi K=3/10 dan
PCA/Zernike dengan cap 1000 NFEV. Aturan plateau di
`scripts/analyze_pilot_budget.py` memilih budget final secara mekanis.

Budget MNIST 840/1000 tidak otomatis sah untuk MADBase. Guard CLI menolak run
konfirmatori sampai keputusan MADBase berstatus `FROZEN`. Pilot tidak masuk
estimasi final.

## 7. Baseline dan analisis tambahan

- Chance dihitung analitik sebagai `1/K` untuk accuracy dan expected macro-F1
  pada kelas seimbang.
- Nearest Centroid memakai fitur input yang sama.
- Multinomial Logistic Regression memilih C hanya melalui validation dengan
  aturan implementasi beku.
- Confusion 6↔9 dipra-spesifikasi sebagai analisis sekunder karena kemiripan
  bentuk; confusion lengkap tetap dilaporkan.

Perbandingan quantum-versus-classical bukan bukti quantum advantage.

## 8. Artefak dan admission

Setiap run menyimpan config, dataset/snapshot ID, split hash, sample IDs,
preprocessing/input hashes, initialization, pair manifest, objective log,
checkpoint provenance, final metrics/predictions, runtime, environment, dan
status attempt. Nama generik `sample_id`/`source_index` dipakai; alias lama
MNIST hanya dipertahankan untuk kompatibilitas.

Run masuk analisis hanya bila dataset, snapshot, split, input, pairing, config,
dan final-evaluation status cocok serta tepat satu attempt `COMPLETED` diterima.

## 9. Batas klaim

- Hasil berlaku untuk snapshot, split, fitur, dan model yang dibekukan.
- Tren terhadap K juga mencerminkan identitas kelas yang ditambahkan.
- Aktivasi Y-odd bukan bukti otomatis peningkatan accuracy.
- Lima seed membatasi presisi dan resolusi inferensi.
- Tidak ada klaim kausal efek dataset, quantum advantage, representativitas
  seluruh penulis Arab, atau penggunaan untuk identifikasi penulis.
- Data mentah tidak boleh didistribusikan tanpa izin/lisensi eksplisit.

## 10. Confirmatory boundary

Boundary baru terbuka bila:

1. `convergence_budget_decision.json` untuk MADBase berstatus `FROZEN`;
2. `check_arabic_readiness.py --require-confirmatory` lulus;
3. commit/ref dan environment setiap run direkam;
4. tidak ada outcome model official-test yang digunakan untuk mengubah desain.

Setiap perubahan setelah boundary adalah amendemen bertanggal.

