# MORE-HD-C Research Readiness Gates

Dokumen ini adalah catatan kontrol persisten untuk proyek MORE-HD COMPLEX. Dokumen harus diperiksa sebelum membuat kode, sebelum menjalankan pilot, sebelum membekukan protokol, dan sebelum menjalankan eksperimen final. Hasil eksperimen final tidak boleh dimulai apabila masih ada item berstatus `BLOCKED` pada Gate G0 atau G1.

## Status dan aturan penggunaan

| Status | Arti |
|---|---|
| `BLOCKED` | Keputusan atau implementasi belum selesai dan menghalangi eksperimen final. |
| `IN PROGRESS` | Perbaikan sedang dikerjakan tetapi belum diverifikasi. |
| `READY FOR VERIFICATION` | Implementasi selesai dan menunggu pengujian atau bukti. |
| `CLOSED` | Kriteria penerimaan terpenuhi dan bukti tersedia. |
| `PILOT ONLY` | Diizinkan untuk smoke test, tetapi tidak boleh dipakai sebagai hasil penelitian final. |

Setiap perubahan status harus mencantumkan tanggal, keputusan final, dan lokasi bukti seperti nama file konfigurasi, hasil unit test, log pilot, atau bagian pseudocode terkait. Status tidak boleh ditutup hanya berdasarkan keberhasilan program dijalankan.

## Gate G0 Validitas data dan evaluasi

| ID | Status awal | Masalah | Keputusan yang wajib dikunci | Kriteria penerimaan |
|---|---|---|---|---|
| G0-01 | `BLOCKED` | Test set digunakan selama objective clustering dan supervised. | Pisahkan optimization train, validation, dan official test. Semua monitoring dan checkpoint selection memakai validation. | Tidak ada akses ke official test di `CLUSTERING_LOOP`, `SUPERVISED_LOOP`, atau Jalur B. Test hanya dipanggil oleh final evaluation setelah protokol dibekukan. |
| G0-02 | `BLOCKED` | Jalur B memakai pemilihan iterasi manual dan berpotensi cherry-picking. | Tetapkan aturan pemilihan checkpoint yang deterministik sebelum melihat hasil test. | Fungsi pemilihan otomatis, kriterianya terdokumentasi, hanya menggunakan train/validation, dan menghasilkan keputusan yang sama untuk input yang sama. |
| G0-03 | `BLOCKED` | Ukuran dataset final belum konsisten antara pseudocode dan FRD lama. | Bedakan konfigurasi smoke test, pilot, dan eksperimen final. Tetapkan jumlah train, validation, dan test per kelas untuk hasil publikasi. | Satu protokol final disetujui dan nilainya konsisten pada config, pseudocode, spreadsheet, metode paper, dan nama run. |
| G0-04 | `BLOCKED` | Satu seed belum cukup untuk menilai variasi optimasi. | Seed 42 tetap digunakan untuk pilot. Eksperimen konfirmatori memakai beberapa seed yang ditetapkan sebelum run final. | Daftar seed final, aturan split berpasangan, jumlah replikasi, dan rencana agregasi statistik terdokumentasi. |

## Gate G1 Optimasi dan desain eksperimen

| ID | Status awal | Masalah | Keputusan yang wajib dikunci | Kriteria penerimaan |
|---|---|---|---|---|
| G1-01 | `PILOT ONLY` | Sepuluh evaluasi COBYLA terlalu kecil untuk 30 dan 60 parameter. | Gunakan 10 hanya untuk smoke test. Tentukan `max_nfev` final melalui pilot konvergensi dan gunakan anggaran yang adil untuk kedua arsitektur. | Kurva konvergensi pilot tersedia; `max_nfev`, `tol`, `rhobeg`, dan stopping rule final sudah dibekukan. |
| G1-02 | `BLOCKED` | Objective logging menyebut setiap function evaluation sebagai iterasi. | Pisahkan `eval_id`, callback/iteration, `nfev`, final point, dan best observed point. | Log serta spreadsheet memakai istilah yang benar dan menyimpan `success`, `status`, `message`, `fun`, serta `nfev`. |
| G1-03 | `BLOCKED` | Pembentukan pasangan clustering dan ketidakseimbangan pasangan belum dikunci. | Tetapkan jumlah sampel per kelas, aturan pasangan, sampling tanpa duplikasi, dan kebijakan penyeimbangan/bobot pasangan sekelas versus berbeda kelas. | Pair builder deterministik, statistik pasangan tersimpan, dan unit test memverifikasi jumlah serta komposisi pasangan untuk K=3 sampai K=10. |
| G1-04 | `BLOCKED` | Analisis faktorial dan statistik inferensial belum ditetapkan. | Bandingkan arsitektur secara berpasangan pada split dan seed yang sama; tentukan estimand, confidence interval, effect size, dan pengujian statistik. | Dokumen analysis plan tersedia sebelum hasil final dibuka. |
| G1-05 | `BLOCKED` | Peningkatan MORE-HD-C dapat dipengaruhi penambahan parameter 30 menjadi 60, bukan fase kompleks saja. | Tetapkan batas klaim dan, bila memungkinkan, ablation atau parameter-budget control. | Paper tidak mengklaim kausalitas yang melebihi desain; ablation atau penjelasan keterbatasan tersedia. |

## Gate G2 Definisi preprocessing dan numerik

| ID | Status saat ini | Masalah | Keputusan yang wajib dikunci | Kriteria penerimaan |
|---|---|---|---|---|
| G2-01 | `IN PROGRESS` (desain dikunci 2026-09-22; menunggu implementasi kode + unit test) | Implementasi Hu Moments belum sepenuhnya operasional. | Tetapkan grayscale/binary input, threshold, formula signed-log, epsilon, penanganan non-finite, dan padding kanal kedelapan. | Test menunjukkan tujuh Hu descriptors, nilai finite, dan kanal kedelapan tetap tepat 0.0 setelah scaling. |
| G2-02 | `IN PROGRESS` (desain dikunci 2026-09-22; menunggu implementasi kode + unit test) | Implementasi Zernike belum sepenuhnya operasional. | Tetapkan pemetaan unit disk, pusat, radius, interpolasi, perlakuan piksel di luar disk, normalisasi, magnitude, dan urutan delapan pasangan `(n,m)`. | Extraction routine deterministik, menghasilkan tepat delapan fitur, dan lulus test pada citra referensi. |
| G2-03 | `IN PROGRESS` (kebijakan dikunci 2026-09-22: tidak ada clipping untuk PCA, HU, maupun ZERNIKE; menunggu unit test) | Kebijakan scaling untuk nilai validation/test di luar rentang train belum ditentukan. | Scaler hanya fit pada train dan kebijakan clipping ditetapkan eksplisit. | Unit test membuktikan tidak ada fitting pada validation/test dan seluruh sudut input mengikuti rentang yang disepakati. |
| G2-04 | `BLOCKED` | Cosine distance dan normalisasi centroid belum memiliki perlindungan zero norm. | Tetapkan epsilon dan perilaku ketika norm mendekati nol. | Tidak ada NaN/Inf dan kasus zero/near-zero mempunyai hasil deterministik serta tercatat sebagai warning. |
| G2-05 | `BLOCKED` | Aktivasi dimensi hanya bergantung pada threshold tunggal. | Ukur nilai raw per observable, tambahkan ukuran kontinu untuk enam Y-odd observables, dan lakukan sensitivity check threshold. | Hasil menyertakan count, mean/max absolute value, ukuran kontinu, dan threshold yang digunakan. |

## Gate G3 Artefak dan spreadsheet

| ID | Status awal | Masalah | Keputusan yang wajib dikunci | Kriteria penerimaan |
|---|---|---|---|---|
| G3-01 | `BLOCKED` | Workbook hanya menyediakan 10 slot history per run. | Jadikan history dinamis atau bangun template berdasarkan `max_nfev` final. | Semua evaluasi dapat ditulis tanpa truncation atau overwrite. |
| G3-02 | `BLOCKED` | Artefak belum cukup untuk mengisi semua sheet. | Simpan prediction details, class distances, sample IDs, per-observable activity, pair statistics, intra-class norm statistics, runtime, raw MSE, per-class metrics, dan optimizer metadata. | Setiap kolom workbook memiliki sumber artefak yang jelas dan diuji pada satu run. |
| G3-03 | `BLOCKED` | Konsolidasi workbook lokal belum diotomatisasi. | Buat konsolidator satu-proses yang idempotent dengan duplicate detection berdasarkan `run_id` dan attempt. | Pengujian menggabungkan beberapa run, menolak duplikasi, dan tidak mengubah data run lain. |
| G3-04 | `BLOCKED` | Status run dan penanganan crash belum lengkap. | Gunakan status `RUNNING`, `COMPLETED`, dan `FAILED`; simpan error; tetapkan kebijakan resume atau attempt baru. | Simulasi kegagalan tidak merusak run lama dan rerun tidak menimpa artefak sebelumnya. |

## Gate G4 Reproducibility dan verifikasi sirkuit

| ID | Status awal | Masalah | Keputusan yang wajib dikunci | Kriteria penerimaan |
|---|---|---|---|---|
| G4-01 | `BLOCKED` | Environment komputasi belum dikunci. | Rekam versi Python, PennyLane, SciPy, NumPy, scikit-learn, library Hu/Zernike, device, shots, dtype, OS, dan Git commit. | Environment dapat dibuat ulang dan metadata tersimpan pada setiap manifest run. |
| G4-02 | `BLOCKED` | Identitas sampel MNIST belum disimpan. | Gunakan split generator deterministik dan simpan indeks asli serta hash split. | Arsitektur yang dibandingkan memakai sampel identik; split dapat direkonstruksi dari artefak. |
| G4-03 | `BLOCKED` | Belum ada test struktural untuk circuit output. | Uji jumlah parameter, output shape, urutan observable, structural zeros MORE-HD, dan aktivasi Y-odd yang mungkin pada MORE-HD-C. | Seluruh unit test lulus untuk kedua arsitektur sebelum pilot. |
| G4-04 | `BLOCKED` | Matriks korelasi belum menangani label nonkontigu dan kasus nilai off-diagonal konstan. | Gunakan `class_to_index`, simpan raw MSE, dan tetapkan fallback normalisasi. | S simetris, diagonal -1, off-diagonal dalam rentang yang ditetapkan, dan tidak ada pembagian dengan nol. |

## Ketidakkonsistenan dokumen yang harus diperbaiki

| ID | Status awal | Dokumen | Perbaikan |
|---|---|---|---|
| D-01 | `BLOCKED` | BAB 1 | Hanya Hu Moments yang menggunakan satu kanal padding. Zernike menghasilkan delapan koefisien yang dikunci. |
| D-02 | `BLOCKED` | BAB 1 dan klaim metode | Ganti klaim bahwa RZ memastikan 15 dimensi aktif menjadi RZ menghilangkan real-state restriction sehingga enam Y-odd observables dapat aktif. |
| D-03 | `BLOCKED` | FRD-09 | Selaraskan ukuran data, iterasi, versi Python, tiga metode fitur, dua arsitektur, dan artefak keluaran dengan protokol terbaru. |
| D-04 | `BLOCKED` | FRD-09 | Perbaiki deskripsi qubit menjadi 8 data qubit dan 2 readout qubit tanpa ancilla tambahan yang terpisah. |
| D-05 | `BLOCKED` | Pseudocode, workbook, dan paper | Gunakan istilah train, validation, official test, objective evaluation, dan final evaluation secara konsisten. |

## Tahapan persetujuan eksperimen

### Gate A Sebelum generate kode utama

- G0-01 sampai G0-03 harus memiliki keputusan desain yang eksplisit.
- Definisi Hu, Zernike, pairing, dan safe cosine harus selesai.
- Schema artefak harus dapat mengisi seluruh workbook.

### Gate B Sebelum smoke test

- Unit test preprocessing dan sirkuit lulus.
- Smoke config diberi label `PILOT ONLY`.
- Test set resmi tidak digunakan untuk monitoring.

### Gate C Sebelum eksperimen final

- Seluruh item G0 dan G1 berstatus `CLOSED`.
- Seluruh item teknis G2 sampai G4 yang memengaruhi hasil berstatus `CLOSED`.
- Protokol, daftar seed, split, optimizer budget, dan analysis plan telah dibekukan.
- Workbook final dan konsolidator telah diuji.

### Gate D Sebelum analisis publikasi

- Seluruh run memiliki status `COMPLETED` dan manifest valid.
- Tidak ada duplicate run, missing artifact, NaN, atau truncated history.
- Official test baru dievaluasi setelah pemilihan model dan parameter selesai.
- Analisis mengikuti rencana statistik yang dibekukan, bukan pola yang ditemukan setelah melihat hasil.

## Log keputusan

| Tanggal | ID | Status baru | Keputusan dan bukti |
|---|---|---|---|
| 2026-09-19 | Semua | `INITIAL REVIEW` | Checklist dibuat berdasarkan audit pseudocode, workbook, FRD-09, BAB 1, Introduction IEEE, dan baseline MORE. |
| 2026-09-22 | G2-01 | `BLOCKED` -> `IN PROGRESS` | Input citra Hu dikunci grayscale ternormalisasi (bukan biner/Otsu). Formula signed-log dikunci `h' = -sign(h) x log10(|h|+epsilon)`, `epsilon=1e-30`, non-finite diganti 0.0 dan dicatat sebagai `n_non_finite_hu`. Padding kanal ke-8 dikunci `0.0` (menang atas referensi pi/2 di `noa3.ipynb`, mengikuti keputusan formal di BAB 1). Pseudocode `HU_MOMENTS` dan `SIGNED_LOG_TRANSFORM` ditambahkan di `Pseudocode_2x3_manual_runs.md` bagian 2.8.4. Bukti: bagian 2.8.2 dan 2.8.4 dokumen pseudocode. Masih `IN PROGRESS`, bukan `CLOSED`, karena implementasi kode nyata dan unit test (7 descriptor, nilai finite, kanal ke-8 tepat 0.0 setelah scaling) belum dikerjakan. |
| 2026-09-22 | G2-02 | `BLOCKED` -> `IN PROGRESS` | 8 pasangan `(n,m)` Zernike direvisi menjadi `[(2,0),(2,2),(3,1),(3,3),(4,0),(4,2),(5,1),(5,5)]`, membuang `(0,0)` dan `(1,1)` yang kurang diskriminatif untuk citra tersentralisasi. `MAP_IMAGE_TO_UNIT_DISK` dikunci: pusat geometris, radius = setengah sisi terpendek, piksel di luar disk di-mask 0, tanpa interpolasi. `EXTRACT_ZERNIKE_TERMS` dikunci sebagai ekstraksi magnitude `|Z_nm|` per pasangan dari hasil komputasi hingga derajat maksimum. Pseudocode kedua fungsi ditambahkan di `Pseudocode_2x3_manual_runs.md` bagian 2.8.4. Bukti: bagian 2.8.3 dan 2.8.4 dokumen pseudocode. Masih `IN PROGRESS` karena implementasi kode nyata dan unit test pada citra referensi belum dikerjakan. |
| 2026-09-22 | G2-03 | `BLOCKED` -> `IN PROGRESS` | Kebijakan clipping diseragamkan lintas ketiga `feature_method`: tidak ada clipping eksplisit setelah scaling untuk PCA, HU, maupun ZERNIKE (menyimpang dari kode referensi `noa3.ipynb` yang meng-clip Hu/Zernike, demi menghindari confound perlakuan berbeda antar metode). Bukti: bagian 2.5.1 dokumen pseudocode. Masih `IN PROGRESS` karena belum ada unit test yang memverifikasi scaler hanya fit pada train dan tidak ada clipping diterapkan. |

## Ringkasan status saat ini

- Kode prototipe dan unit test: dapat mulai setelah keputusan Gate A dimasukkan ke spesifikasi. Keputusan Hu Moments dan Zernike Moments (G2-01, G2-02) sudah dikunci di level desain per 2026-09-22; implementasi kode + unit test masih menjadi pekerjaan berikutnya.
- Smoke test 10 evaluasi dengan seed 42: diperbolehkan sebagai `PILOT ONLY`.
- Eksperimen 48 run final: `BLOCKED` sampai Gate C terpenuhi.
- Hasil test dari pilot tidak boleh diperlakukan sebagai hasil konfirmatori publikasi.