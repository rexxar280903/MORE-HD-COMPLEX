# MORE-HD-COMPLEX

<!-- RESEARCH_PROGRESS_START -->
## Research Progress

![Research Progress](https://img.shields.io/badge/Research%20Progress-100.0%25-blue)

**Current research readiness: 100.0%**

`████████████████████ 100.0%`

Progress dihitung otomatis dari **39 item Research Readiness Gate** dengan bobot:
`CLOSED = 100%`, `READY FOR VERIFICATION = 75%`, `IN PROGRESS = 50%`, `PILOT ONLY = 25%`, dan `BLOCKED = 0%`.

### Progress History

| Tanggal | Commit acuan | Progress | Perubahan | Ringkasan |
|---|---|---:|---:|---|
| 2026-09-27 | [`5aeb1ca`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/5aeb1cabe3e55673b51b0cf09e4c0865ba79f2f4) | 20.9% | +2.7 pp | Protokol deterministic clustering pairs dan dokumentasi 240-run diselaraskan. |
| 2026-09-27 | [`f753c85`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/f753c85a08d8e2d45cbe6429d8ea79c2a6e7eeaa) | 24.3% | +3.4 pp | docs: lock targeted ablation for G1-05 and G1-07 |
| 2026-09-27 | [`6670319`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/6670319c34497679b0781ddf0a3654319a8fe4d7) | 24.3% | 0.0 pp | docs: consolidate ablation into main pseudocode |
| 2026-09-27 | [`7e6603c`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/7e6603cdef2adff678606b1bb721473b6cecae75) | 25.7% | +1.4 pp | docs: resolve G0-02 with deterministic Jalur B selector |
| 2026-09-27 | [`acae2ca`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/acae2ca6eee5d7354427967dec9486afd10ba074) | 27.0% | +1.3 pp | docs: lock pilot budget protocol (G1-01) and COBYLA simplex rules (G1-06) |
| 2026-09-27 | [`6413658`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/6413658f6ae02f9b35fb060bb93cea778300fdb3) | 28.3% | +1.3 pp | docs: resolve G1-09 cumulative class-sequence confound as explicit limitation |
| 2026-09-27 | [`d7c205e`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/d7c205e00bc2e65d8a4c1938f8b3368d8a59b6cc) | 29.6% | +1.3 pp | docs: resolve G1-10 with classical baselines and no-R MORE reference |
| 2026-09-27 | [`26b5fd2`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/26b5fd29fc15eb3298367a7507587bd35f16697c) | 29.6% | 0.0 pp | docs: consistency audit — smoke budget, pilot scope, workbook schema map |
| 2026-10-01 | [`cdd7edc`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/cdd7edcfe209ccd358afb70a7734d43fbe6cef63) | 29.6% | 0.0 pp | docs(sim): kunci backend simulasi statevector analitik (G4-01 sebagian, prasyarat G2-04) |
| 2026-10-01 | [`f71590b`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/f71590bf7ffb4df3a9c4705fbbdb5d6d74a2d934) | 31.4% | +1.8 pp | docs(gates): G2-04 zero-norm safety + G2-07 centroid mengikuti MORE |
| 2026-10-01 | [`bba4657`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/bba46576330385c7c85cb7cb9e5fb508c117bc4d) | 31.4% | 0.0 pp | docs(gates): G1-03 mengikuti paper MORE untuk jumlah pasangan clustering |
| 2026-10-01 | [`58b8380`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/58b838090c8190a75fc4ec28329cce8a2f327f58) | 32.7% | +1.3 pp | docs(gates): G3-05 caching per sampel unik + monitoring validation FULL_VAL |
| 2026-10-01 | [`04f097d`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/04f097de9275f845cf7c8a0f60f06df58ddb5c35) | 32.7% | 0.0 pp | docs(runtime): pencatatan waktu 4 tingkat + runtime deskriptif + microbenchmark |
| 2026-10-01 | [`634a784`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/634a784ea4b0d05e54c16cdb2b2bb46a8937b927) | 35.3% | +2.6 pp | docs(gates): G2-05 diagnostik Y-odd per sampel + G2-06 outcome pemisahan mengikuti MORE |
| 2026-10-05 | [`47089ce`](https://github.com/rexxar280903/MORE-HD-COMPLEX/commit/47089ce8824f7c8417388f28ec7f2db54c818fe3) | **100.0%** | **+64.7 pp** | Gate C: pipeline MORE-HD/MORE-HD-C, track MORE-REPRO, pilot seed 42, budget final 840/1000 (#1) |

**Trend terbaru:** `28.3% → 29.6% → 29.6% → 29.6% → 31.4% → 31.4% → 32.7% → 32.7% → 35.3% → 100.0%`

Riwayat lengkap tersimpan di [`research_progress_history.json`](research_progress_history.json).

> Bagian ini dikelola otomatis oleh GitHub Actions. Nilai dapat turun bila scope/kriteria penelitian bertambah; tracker tidak memaksa progress selalu meningkat.
<!-- RESEARCH_PROGRESS_END -->

Proyek riset quantum machine learning: **MORE-HD** (8 data qubit + 2 readout qubit, readout 15 observable Pauli) dan pengembangannya **MORE-HD-C** (RY+RZ), dibandingkan langsung dengan **MORE** (Wu dkk., 2023) pada klasifikasi MNIST multi-kelas K = 3..10. Seluruh training dijalankan ulang dari nol dengan protokol yang dikunci sebelum hasil apa pun dibuka; dokumen skripsi lama (BAB 1, FRD-09) tidak lagi menjadi acuan (lihat [Catatan skripsi](#catatan-skripsi)).

## Status (2026-10-05)

- **Implementasi selesai**: seluruh pseudocode kini berupa kode yang dapat dijalankan (`core/`, `main_*.py`) dengan **73 unit/integration test** yang memetakan kriteria penerimaan tiap gate (`tests/`).
- **Pilot seed 42 selesai** (PILOT ONLY, tidak masuk hasil): smoke test, timing pilot, microbenchmark, dan pilot konvergensi 14 run. Budget COBYLA final dibekukan dari aturan plateau yang dikunci sebelum pilot: **`max_nfev_clustering = 840`, `max_nfev_supervised = 1000`** (clustering: semua run pilot plateau; supervised: run R048-S42 belum plateau tipis pada cap 1.000, sehingga budget supervised = cap dan klaim fase supervised dinyatakan pada budget evaluasi yang sama).
- **Semua 39 item readiness gate berstatus `CLOSED` untuk Gate C** (lihat blok ringkasan di bawah). Kewajiban yang hanya bisa dipenuhi setelah run konfirmatori (tabel hasil, 240 fit baseline, kontras ablation, perbandingan MORE, sensitivitas 6↔9) dipindah ke daftar **Gate D `P-01…P-08`** di dokumen gate.
- **Langkah berikutnya (Anda):** jalankan 450 run konfirmatori + 90 Jalur B mengikuti [`docs/RUNBOOK.md`](docs/RUNBOOK.md). Proyeksi ±32 CPU-jam single-thread (≈8–10 jam dengan 4 proses paralel) pada mesin referensi.

## Latar Belakang

**MORE-HD** (*Multi-Readout Qubit Expansion for High-Dimensional Label Space*) memperluas kerangka MORE (Wu dkk., 2023) dari 1 qubit readout (3 observable) menjadi 2 qubit readout (15 observable Pauli non-identitas), memakai 10 qubit: **8 data qubit + 2 readout qubit, tanpa ancilla terpisah**, dan 30 parameter terlatih (RY + CNOT), lebih sedikit dari 91 parameter MORE.

**Structural zeros.** Dengan gerbang RY dan CNOT saja, seluruh amplitudo state bernilai riil, sehingga enam observable dengan jumlah faktor Y ganjil (`IY, XY, YI, YX, YZ, ZY`) selalu bernilai nol. Engine kami mereproduksi ini secara eksak (0.0) untuk MORE-HD.

**Motivasi dan hipotesis.** Pada skripsi (protokol lama), MORE-HD meningkatkan akurasi pada K = 3–7 (rata-rata ~9.74 poin) tetapi turun tajam pada K = 8 dan 10. Hipotesis penelitian ini: keterbatasan wilayah ruang readout yang dapat dicapai sirkuit bernilai riil ikut membuat label kuantum berdesakan pada K besar (*crowded quantum labels*, mekanisme yang diidentifikasi Wu dkk.; "curse of density" adalah adaptasi istilah kami). Angka skripsi tidak dipakai sebagai hasil: seluruh perbandingan diulang dari nol dengan protokol baru, dan pertanyaan penelitian lengkap ada di [`docs/METHODS_DRAFT.md`](docs/METHODS_DRAFT.md) §1.

## Kontribusi MORE-HD-C

MORE-HD-C menambahkan gerbang RZ setelah setiap RY pada layer variational (60 parameter terlatih, masih < 91 parameter MORE). RZ menghilangkan *real-state restriction* sehingga keenam observable Y-odd **dapat** aktif; aktivasi itu diukur secara empiris (`yodd_norm_fraction`) dan tidak dengan sendirinya membuktikan peningkatan akurasi. Encoding, blok entangling CNOT, jumlah qubit, readout, dan urutan 15 observable identik dengan MORE-HD.

## Desain eksperimen (dikunci)

| Track | Model | Kondisi | Seed konfirmatori | Run | Peran |
|---|---|---|---:|---:|---|
| Primer | A = MORE-HD, D = MORE-HD-C | K=3..10 × PCA/HU/ZERNIKE | 101, 202, 303, 404, 505 | 240 | estimand primer D − A |
| Ablation | B = MORE-HD-60P (6 layer RY, 60 param), C = MORE-HD-C-FixedRZ (30 RY + 30 RZ beku) | K∈{3,6,10} × 3 fitur | 5 seed | 90 | memisahkan efek jumlah parameter vs akses state kompleks |
| **Referensi MORE (baru)** | M = MORE-REPRO (QCNN MORE 91 param, readout X/Y/Z, tanpa loss adjuster R) | K=3..10 × 3 fitur | 5 seed | 120 | perbandingan langsung dengan MORE pada protokol identik |
| Jalur B (sekunder) | A, D dari checkpoint clustering terpilih deterministik | K∈{3,6,10} × 3 fitur | 5 seed | 90 | analisis sensitivitas |
| Baseline klasik | chance 1/K, Nearest Centroid, Logistic Regression | 120 cell | 5 seed | 240 fit | referensi konteks |

Total **450 eksekusi konfirmatori unik** (240 + 90 + 120) ditambah 90 Jalur B. Seed 42 hanya untuk pilot. Split train/validation/official test = 1000/100/200 per kelas, berpasangan lintas model dan fitur, nested lintas K (`splits/seed*.json`, di-commit). Semua fitur menghasilkan tepat 8 kanal (hanya Hu memakai satu kanal padding 0.0; Zernike memberi 8 koefisien asli).

## Perbandingan dengan MORE (Wu dkk., 2023)

1. **Dalam protokol yang sama (MORE-REPRO).** Sirkuit MORE direproduksi gerbang demi gerbang dari kode publik (`github.com/Jindi0/MORE@867d194`, `model.py::build_qcnn`) — kesetaraannya dengan sirkuit Qiskit MORE diuji (< 1e-12) — lalu dilatih dengan data, split, fitur, pasangan clustering, matriks korelasi, aturan label, budget COBYLA, dan official test yang identik dengan MORE-HD/MORE-HD-C. Selisih A−M dan D−M (berpasangan per seed) dilaporkan sesuai SAP §10.3.
2. **Angka yang dilaporkan paper MORE.** Kolom MORE\R Tabel I (acuan utama) dan MORE+R (konteks) hanya deskriptif, karena protokolnya berbeda. Perbedaan protokol terhadap kode publik MORE dicatat di [`docs/METHODS_DRAFT.md`](docs/METHODS_DRAFT.md) §8.
3. Agar setara dengan MORE, pipeline mengikuti MORE pada: 5 sampel clustering/kelas dan seluruh `C(5K,2)` pasangan, loss clustering `−S_ij·d_cos`, matriks korelasi `calc_class_rela` (`MSE/max`, diagonal −1), label kuantum normalisasi→median→normalisasi, loss supervised jarak cosine ke label, prediksi label terdekat, COBYLA `rhobeg=1.0`.

## Pipeline dua fase

1. **Clustering (unsupervised):** 5 sampel TRAIN per kelas dipilih acak ber-seed, seluruh pasangan unik `i<j` dipakai tanpa balancing; COBYLA meminimalkan `mean(−S_ij · d_cos(v_i, v_j))`; validation hanya dimonitor pasif.
2. **Label kuantum:** centroid MORE (normalisasi → median per komponen → normalisasi) dari 5 sampel clustering per kelas pada `result.x`.
3. **Supervised:** COBYLA meminimalkan rata-rata jarak cosine output setiap sampel train (1000/kelas) ke label kelasnya.
4. **Evaluasi akhir:** official test dibuka sekali, di tahap `final_evaluation` saja (dijaga `OfficialTestVault`).

Budget sama untuk semua model (total `nfev`), tahap simplex awal COBYLA ditandai `phase`, dan simulasi statevector analitik complex128 tanpa shot noise (engine produksi terverifikasi vs PennyLane `default.qubit`).

## Struktur repositori

```text
core/                      implementasi (data, fitur, sirkuit + engine, loop, evaluasi, workbook, statistik)
main_train.py              Jalur A primer (MORE-HD, MORE-HD-C)
main_ablation.py           ablation B/C
main_more_reference.py     track MORE-REPRO
main_selected_clustering.py  Jalur B
main_classical_baseline.py baseline klasik
analysis/run_analysis.py   analisis SAP (agregasi, kontras, MORE, McNemar, 6↔9, baseline)
scripts/                   pilot, proyeksi, microbenchmark, rencana run, template workbook, util
tests/                     unit + integration test per gate
splits/                    split manifest seed 42/101/202/303/404/505
research_data/             template workbook (primer, ablation, MORE reference), bukti pilot
docs/                      METHODS_DRAFT.md (draf metode paper), RUNBOOK.md
```

## Cara menjalankan (ringkas)

Stack terkunci (`requirements.txt`): Python 3.11, NumPy 2.4.6 (engine statevector produksi), SciPy 1.17.1 (COBYLA), scikit-learn 1.9.1 (PCA, baseline klasik), PennyLane 0.45.1 (acuan verifikasi engine), openpyxl (workbook). OpenCV, mahotas, dan Qiskit hanya dipakai test sebagai acuan (`requirements-test.txt`).

```bash
pip install -r requirements-test.txt
python scripts/download_mnist.py
python -m pytest tests -q
python scripts/print_run_plan.py --pending --n-parallel 4   # daftar perintah run konfirmatori
```

Detail lengkap (attempt ulang, Jalur B, baseline, konsolidasi, analisis): [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Dokumen kendali

- [`MORE_HD_RESEARCH_READINESS_GATES (1).md`](MORE_HD_RESEARCH_READINESS_GATES%20(1).md) — status gate, log keputusan, kewajiban Gate D.
- [`MORE_HD_STATISTICAL_ANALYSIS_PLAN.md`](MORE_HD_STATISTICAL_ANALYSIS_PLAN.md) — SAP v2.0 (dibekukan sebelum hasil).
- [`Pseudocode_2x3_manual_runs.md`](Pseudocode_2x3_manual_runs.md) — spesifikasi teknis + peta implementasi (Bagian 14).
- [`docs/METHODS_DRAFT.md`](docs/METHODS_DRAFT.md) — teks metode, perbandingan MORE, keterbatasan untuk paper.

## Ringkasan kesiapan

<!-- READINESS_SUMMARY_START -->
### Ringkasan Kesiapan (Readiness Gate) — per 2026-10-05

| Gate | Cakupan | Item | Closed | Ready for Verification | In Progress | Pilot Only | Blocked | Kesiapan* |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | Validitas data & evaluasi | 4 | 4 | 0 | 0 | 0 | 0 | **100.0%** |
| G1 | Optimasi & desain eksperimen | 11 | 11 | 0 | 0 | 0 | 0 | **100.0%** |
| G2 | Preprocessing & definisi numerik | 7 | 7 | 0 | 0 | 0 | 0 | **100.0%** |
| G3 | Artefak & spreadsheet | 7 | 7 | 0 | 0 | 0 | 0 | **100.0%** |
| G4 | Reproducibility & verifikasi sirkuit | 4 | 4 | 0 | 0 | 0 | 0 | **100.0%** |
| D | Ketidakkonsistenan dokumen (BAB 1, FRD-09, dll.) | 6 | 6 | 0 | 0 | 0 | 0 | **100.0%** |
| **Total** |  | **39** | **39** | **0** | **0** | **0** | **0** | **100.0%** |

\* Kesiapan dihitung sebagai rata-rata bobot per item: `CLOSED=100%`, `READY FOR VERIFICATION=75%`, `IN PROGRESS=50%`, `PILOT ONLY=25%`, dan `BLOCKED=0%`.
<!-- READINESS_SUMMARY_END -->

## Catatan skripsi

Atas arahan pemilik penelitian (2026-10-04), dokumen skripsi lama — `BAB 1.docx` dan FRD-09 — **tidak lagi menjadi sumber kebenaran** karena seluruh training diulang dari nol dengan protokol baru. Berkas `BAB 1.docx` dipertahankan hanya sebagai arsip. Klaim metode yang berlaku ada di [`docs/METHODS_DRAFT.md`](docs/METHODS_DRAFT.md), termasuk daftar penyimpangan terhadap protokol skripsi (§12). Hasil MORE-HD pada penelitian ini tidak identik protokolnya dengan skripsi.

## Referensi

Wu, J., Hu, T., & Li, Q. (2023). *MORE: Measurement and Correlation Based Variational Quantum Circuit for Multi-classification*. arXiv:2307.11875; kode: github.com/Jindi0/MORE. McClean et al. (2018), Cerezo et al. (2021), Holmes et al. (2022), Schuld et al. (2021), Hu (1962), Teague (1980).
