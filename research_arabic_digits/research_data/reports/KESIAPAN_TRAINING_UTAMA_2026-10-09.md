# Audit kesiapan training utama MADBase

Tanggal: 9 Oktober 2026. Status: **SIAP UNTUK KONFIRMATORI**.
Audit ini tidak menjalankan training utama. Seluruh 450 konfigurasi masih
NOT_STARTED saat audit dilakukan.

## Hasil pemeriksaan

- Seluruh 14 kondisi pilot konvergensi selesai, masing-masing satu attempt
  COMPLETED diterima. Attempt pertama yang gagal karena izin Windows tetap
  tersimpan dan dikecualikan; retry memakai konfigurasi dan seed yang sama.
- Budget final clustering 900 dan supervised 940 evaluasi, ditetapkan mekanis
  melalui aturan plateau dari train loss; seluruh kurva memenuhi aturan itu.
- Snapshot arsip MADBase, hash split, hash array, pasangan clustering,
  data berpasangan antarmodel, bentuk delapan fitur, hasil finite, metrik pilot,
  serta pembacaan artefak untuk ekspor Excel tervalidasi.
- Enam manifest split valid. Training dan validation writer-disjoint;
  official test tetap terisolasi dari pemilihan budget dan desain.
- Tiga template Excel MADBase valid. Sheet riwayat terbesar pada track primer
  memerlukan paling banyak 216.000 baris clustering dan 225.600 baris supervised,
  masing-masing berada di bawah batas worksheet Excel.
- Environment cocok dengan dependency lock. Bukti uji regresi terbaru ada di
  `pretraining_tests.xml` dan `../pilot/test_summary.json`.
- Disk tersedia saat audit 11,26 GB. Estimasi artefak dan workbook 450 run utama
  2,69 GB; kebutuhan dengan cadangan operasional 6,37 GB. Ini proyeksi dari ukuran
  artefak pilot, bukan jaminan batas ukuran hasil.

## Matriks eksperimen yang dijalankan

| Tahap | Model | Kondisi | Jumlah |
|---|---|---|---:|
| Primer | MORE-HD (A), MORE-HD-C (D) | 3 fitur × 8 K × 5 seed × 2 model | 240 run |
| Ablasi | MORE-HD-60P (B), MORE-HD-C-FixedRZ (C) | 3 fitur × 3 K × 5 seed × 2 model | 90 run |
| Referensi | MORE-REPRO (M) | 3 fitur × 8 K × 5 seed | 120 run |
| Jalur B sekunder | A dan D dari checkpoint clustering terpilih deterministik | 3 fitur × 3 K × 5 seed × 2 model | 90 run |
| Baseline klasik | Nearest Centroid, Logistic Regression | 3 fitur × 8 K × 5 seed × 2 model | 240 fit |
| Chance | 1/K | Dihitung analitik | Tanpa training |

Fitur: PCA, Hu, Zernike. K primer/referensi: 3–10; K ablasi/Jalur B: 3, 6, 10.
Seed utama: 101, 202, 303, 404, 505. Pilot seed 42 tidak masuk hasil utama.
Jumlah training kuantum 540; baseline klasik 240 fit di 120 kondisi.

Setiap run Jalur A mencakup preprocessing fit train-only, clustering,
pembentukan label kuantum, supervised training, lalu evaluasi official test.
Ukuran per kelas: 1.000 train, 100 validation, 200 official test. Data dan hasil
MADBase tetap dipisah dari MNIST.

## Pelaksanaan dan keluaran

Maksimal enam run paralel adalah konfigurasi yang sudah dicoba di PC ini;
setiap proses membatasi thread perhitungan ke satu. Beban workbook dan fase
10 kelas dapat berbeda dari pengukuran pilot, sehingga perlu dipantau saat
training utama berjalan. Paralelisme dicatat per run.

Baseline klasik dan Jalur B dijalankan setelah run sumber yang diperlukan
selesai. Setelah seluruh hasil tersedia, lakukan konsolidasi Excel dan analisis
statistik (agregasi seed, perbandingan berpasangan, confidence interval, effect
size, koreksi Holm, confusion lengkap dan analisis sekunder 6↔9).

Artefak sumber tetap JSON/JSONL/NPY/BIN; Excel dihasilkan dari artefak itu.
Training utama mengaktifkan workbook per run; konsolidasi membangun tiga
workbook gabungan. Pemeriksaan admission final wajib sebelum pelaporan hasil.

## Bukti yang dapat diperiksa

- `full_readiness_audit.json`: audit hash, pairing, split, konfigurasi dan storage.
- `confirmatory_run_plan.json`: daftar 450 konfigurasi dan perintah training.
- `../pilot/convergence_budget_decision.json`: aturan dan keputusan budget.
- `../pilot/convergence_check.json`: verifikasi optimizer dan holdout pilot.
- `../manifests/readiness_report.json`: pemeriksaan kesiapan resmi.

Audit final dan hasil penelitian tetap harus diverifikasi setelah seluruh
training selesai; status siap menjalankan tidak berarti penelitian selesai.
