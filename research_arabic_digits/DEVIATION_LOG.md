# Deviation & Amendment Log — Studi Digit Arab (MADBase)

**Status:** `FINAL — G6`
**Dibuat:** 2026-10-11 (Asia/Jakarta)
**Cakupan:** seluruh eksekusi konfirmatori (2026-10-09 23:45 – 2026-10-10 19:02 WIB), Jalur B, baseline,
konsolidasi, analisis, dan penulisan hasil.

Log ini mencatat setiap penyimpangan dari protokol/SAP v1.0 dan setiap amendemen setelah confirmatory boundary
dibuka. Kolom "Outcome sudah tersedia?" menyatakan apakah metrik official test sudah terlihat ketika deviasi
terjadi atau dicatat. **Tidak ada deviasi yang mengubah data, split, model, metrik, budget optimizer, aturan
admission, atau aturan analisis konfirmatori.**

| ID | Waktu (WIB) | Tahap | Deviasi dari protokol beku | Alasan | Outcome sudah tersedia? | Penanganan / dampak |
|---|---|---|---|---|---|---|
| DEV-01 | 2026-10-09 23:45 → 2026-10-10 00:12; 2026-10-10 16:49 | Eksekusi | Antrean konfirmatori dimulai dengan 8 proses paralel (16 run pertama, `n_parallel_declared = 8`), lalu diserahkan ke 6 proses (434 run berikutnya; `handover_to6.json`). Setelah fase utama selesai (selama sebagian Jalur B/baseline), kontrol antrean diturunkan ke 2 proses prioritas BelowNormal (`queue/control.json`). Laporan kesiapan menyebut maksimum 6 proses sebagai konfigurasi yang sudah dicoba. | Kapasitas mesin; kebutuhan CPU pengguna untuk tugas lain | Tidak (run pertama belum selesai) | Setiap proses dibatasi satu thread BLAS/OMP; simulasi deterministik, sehingga metrik tidak terpengaruh. Hanya waktu eksekusi yang terdampak; runtime dilaporkan deskriptif dan disertai catatan ini. |
| DEV-02 | Sepanjang eksekusi | Provenance kode | Semua run merekam `git_commit = 188bab6`, tetapi working tree tidak bersih: skrip orkestrasi antrean (`scripts/run_confirmatory_queue.py`, `confirmatory_worker.py`, `verify_confirmatory_completion.py`, `keep_training_awake.py`) belum di-commit dan beberapa skrip pilot/benchmark serta dokumen termodifikasi. | Skrip orkestrasi ditulis setelah commit terakhir | Tidak | `git diff 188bab6 -- core` kosong (engine, fitur, optimizer, evaluasi identik dengan commit; diverifikasi `tools/finalize_g6.py`). `core/pilot_audit.py` hanya alat audit pilot. **Tindakan:** commit seluruh skrip orkestrasi dan dokumen G6 sebelum publikasi agar provenance lengkap. |
| DEV-03 | 2026-10-09 (pilot) | Pilot konvergensi | Attempt 1 `AD-R002-S42` gagal teknis (izin file Windows); retry dengan UID, seed, dan konfigurasi sama diterima. | Galat sistem berkas | Tidak (pilot memakai holdout training pool) | Sudah dicatat sebelum boundary (`READINESS_GATES.md`, `convergence_budget_decision.json`). Attempt gagal disimpan dan dikecualikan. Tidak berdampak pada estimasi konfirmatori. |
| DEV-04 | 2026-10-11 | Dokumentasi (amendemen administratif 1.0.1) | SAP §10 dan protokol masih menyatakan "confirmatory boundary belum dibuka"; RESEARCH_PROTOCOL/SAP diberi baris amendemen yang mencatat boundary dibuka 2026-10-09 ±23:45 WIB dengan budget 900/940 dan commit 188bab6. | Teks belum diperbarui setelah gate AD-G3-03 ditutup | Ya (dicatat setelah hasil) | Murni administratif; tidak ada perubahan desain/analisis. |
| DEV-05 | 2026-10-11 | Analisis | Analisis tambahan setelah hasil diketahui: (a) analisis mekanistik (linear probe, Jacobian parameter-shift, konvergensi) pada 180 run sel ablasi; (b) porsi galat per pasangan kelas pada K = 10; (c) uji binomial lintas sel dan korelasi Spearman. | Menjelaskan hasil ablasi dan pola galat khas aksara Arab-Indik; mengikuti analisis eksploratif studi MNIST | Ya | Seluruhnya diberi label **EKSPLORATIF** di RESULTS.md dan paper; tidak mengubah tabel konfirmatori. Kode: `paper/build/mechanism/mechanism_analysis_madbase.py`, `paper/build/analyze_madbase.py`. Evaluasi ulang mereproduksi akurasi tercatat dengan selisih 0,0. |

## Catatan interpretasi (bukan deviasi)

- Analisis sekunder 6↔9 dipra-spesifikasi karena diwarisi dari studi MNIST (di mana 6 dan 9 saling rotasi). Pada
  glif Arab-Indik ٦ dan ٩ bukan rotasi satu sama lain; analisis tetap dilaporkan sesuai SAP, dan pasangan yang
  relevan bagi aksara ini dilaporkan terpisah sebagai analisis eksploratif (DEV-05b).
- MORE (Wu dkk., 2023) tidak melaporkan akurasi pada MADBase; perbandingan dengan MORE memakai MORE-REPRO di dalam
  protokol yang sama (SAP §7), bukan nilai literatur.

## Riwayat amendemen protokol/SAP

| Tanggal | Versi | Perubahan | Outcome official test tersedia? |
|---|---|---|---|
| 2026-10-08 | 0.1 | Kerangka awal | Tidak |
| 2026-10-09 | 1.0 | Desain, split, matriks, metrik, inferensi, admission, batas klaim dibekukan; budget digated pilot | Tidak |
| 2026-10-09 | 1.0 (gate) | Budget MADBase 900/940 dibekukan dari 14 pilot konvergensi; boundary dibuka | Tidak |
| 2026-10-11 | 1.0.1 | Amendemen administratif: status boundary, penutupan G6, label eksploratif (DEV-04, DEV-05) | Ya — tidak ada perubahan analisis konfirmatori |
