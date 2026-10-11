# Penelitian MORE-HD-C pada Digit Arab (MADBase)

**Status:** `SELESAI — 450/450 RUN KONFIRMATORI DITERIMA, G6 CLOSED`  
**Versi dokumentasi:** 1.1  
**Terakhir diperbarui:** 2026-10-11

Folder ini adalah dossier terpisah untuk studi MORE-HD/MORE-HD-C pada Modified
Arabic Digits Database (MADBase). Studi MNIST tetap dapat direproduksi melalui
jalur lama; setiap perintah studi Arab harus menyebut `--dataset MADBASE`.

## Status penelitian (2026-10-11)

- **Eksekusi selesai.** 450 run konfirmatori (240 primer, 90 ablasi, 120 MORE-REPRO), 90 run Jalur B, dan
  240 fit baseline klasik diterima oleh `research_data/reports/final_admission.json` (2026-10-10 19:02 WIB,
  `accepted = true`, tanpa galat). Semua run selesai pada attempt pertama.
- **G6 ditutup.** Audit `research_data/reports/g6_final_audit.json`, hasil lengkap `RESULTS.md`, deviasi
  `DEVIATION_LOG.md`, dan paket reproduksi `REPRODUCIBILITY_MANIFEST.md`/`.json`.
- **Temuan utama.** MORE-HD-C − MORE-HD = +5,9 pp akurasi (positif pada 22/24 sel; SK 95% di atas nol pada
  3 sel, tidak ada di bawah nol). Ablasi: B − A +12,3 pp, C − A +2,5 pp, D − B −9,0 pp (9/9 sel negatif),
  D − C +0,7 pp. Kenaikan D tidak dapat diatribusikan pada fase kompleks; kesimpulan MNIST tereplikasi.
  Semua model kuantum tetap jauh di bawah regresi logistik (85,2%) dan nearest centroid (73,3%).
- **Paper.** Draf bahasa Indonesia (template VJCS): `paper/MORE-HD-C_MADBase_paper_ID.docx`; catatan
  diskusi dan cara build ulang: `paper/CATATAN_DISKUSI_PAPER_MADBASE.md`.
- **Dashboard.** `research_data/dashboard/` (jalankan `serve.ps1`, buka `http://localhost:4174`).
- **Data mentah hanya untuk riset lokal dan tidak boleh didistribusikan** (`LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`).
- Studi MNIST di root repositori tidak diubah oleh pekerjaan ini; angkanya hanya dibaca untuk perbandingan
  deskriptif (SAP §8).

## Kontrak studi yang sudah dibekukan

| Komponen | Nilai |
|---|---|
| Dataset | `MADBASE-AUC-V1` |
| Snapshot | `8262dca2d3db6205e0112586dd9a02c35b0b371da292aab0a15dc92976793f78` |
| Kelas | Arabic-Indic `٠١٢٣٤٥٦٧٨٩`, internal `0..9` |
| K | kumulatif/nested `3..10` |
| Fitur | PCA, Hu, Zernike; masing-masing delapan kanal |
| Seed pilot | 42 |
| Seed konfirmatori | 101, 202, 303, 404, 505 |
| Ukuran per kelas | train 1000, validation 100, final test 200 |
| Split train/validation | writer-disjoint dari official training pool |
| Final test | 20 writer official-test tetap, identik lintas seed |
| Model | A, D; ablation B/C; referensi M |
| Kebijakan duplikat | pertahankan snapshot resmi dan laporkan flag; tanpa eksklusi diam-diam |

## Bukti yang tersedia

| Area | Status | Bukti utama |
|---|---|---|
| Dataset dan loader | `CLOSED` | `DATASET_CARD.md`, `core/madbase.py`, manifest audit |
| Split dan anti-leakage | `CLOSED` | `splits/*.json`, `tests/test_madbase_pipeline.py` |
| Fitur dan pipeline | `CLOSED` | PCA/Hu/Zernike finite, train-only fit, smoke lima arsitektur |
| Artefak dan workbook | `CLOSED` | tiga template di `research_data/` |
| Baseline/Jalur B | `CLOSED` untuk pilot | output lokal di `runs/arabic_digits/` |
| Konsolidasi/analisis | `CLOSED` untuk pilot | `research_data/consolidated_pilot/`, `analysis_pilot/` |
| Budget optimizer MADBase | `CLOSED` | 14 pilot lengkap; keputusan beku 900/940 |
| Eksekusi konfirmatori + Jalur B + baseline | `CLOSED` | `research_data/reports/final_admission.json` |
| Konsolidasi dan analisis final | `CLOSED` | `research_data/consolidated/`, `research_data/analysis/`, `RESULTS.md` |
| G6 (pelaporan, deviasi, reproduksi) | `CLOSED` | `research_data/reports/g6_final_audit.json`, `DEVIATION_LOG.md`, `REPRODUCIBILITY_MANIFEST.md` |

Metrik smoke hanya bukti teknis dan **bukan hasil penelitian**. Tidak ada metrik
model dari official test konfirmatori yang digunakan untuk memilih desain. Hasil
penelitian yang sah hanya yang tercantum di `RESULTS.md`.

## Perintah operasional

Pemeriksaan kesiapan:

```powershell
python scripts/check_arabic_readiness.py
```

Smoke ulang lima arsitektur:

```powershell
python scripts/run_pilot.py smoke --dataset MADBASE --parallel 1
python scripts/check_pilot_runs.py --dataset MADBASE --tag smoke
```

Pilot konvergensi yang telah selesai (jangan menjalankan ulang kondisi yang sudah selesai):

```powershell
python scripts/run_pilot.py convergence --dataset MADBASE --first-only
python scripts/run_pilot.py convergence --dataset MADBASE --skip-first --parallel 1
python scripts/analyze_pilot_budget.py --dataset MADBASE --tag convergence
python scripts/check_arabic_readiness.py --require-confirmatory
```

Pilot konvergensi merupakan pekerjaan komputasi panjang. Jangan memulai run
konfirmatori hanya dengan menghapus guard atau menyalin budget dari MNIST.

Penutupan dan pelaporan (setelah konsolidasi dan analisis final; lihat `RUNBOOK.md` §10):

```powershell
python research_arabic_digits/tools/finalize_g6.py
python research_arabic_digits/paper/build/extract_workbooks.py
python research_arabic_digits/paper/build/collect_jalurb.py
python research_arabic_digits/paper/build/analyze_madbase.py
python research_arabic_digits/paper/build/write_results.py
python research_arabic_digits/paper/build/make_figures.py
python research_arabic_digits/paper/build/build_paper.py
```

## Peta dokumen

- `RESULTS.md`: hasil final (struktur `RESULTS_TEMPLATE.md`), angka dari artefak yang lolos admission.
- `DEVIATION_LOG.md`: deviasi dan amendemen setelah boundary konfirmatori.
- `REPRODUCIBILITY_MANIFEST.md` / `.json`: SHA-256 artefak, commit, environment.
- `paper/`: naskah, gambar, dan skrip build paper (`paper/build/`).
- `research_data/dashboard/`: dashboard HTML lokal hasil training MADBase.
- `tools/finalize_g6.py`: audit penutupan G6 dan pembuat manifest reproduksi.

- `PERBAIKAN_DAN_VERIFIKASI_2026-10-09.md`: catatan lengkap pekerjaan dan bukti.
- `DATASET_CARD.md`: identitas, snapshot, audit, lisensi, dan preprocessing.
- `RESEARCH_PROTOCOL.md`: desain eksperimen yang berlaku.
- `STATISTICAL_ANALYSIS_PLAN.md`: estimand, metrik, inferensi, dan admission.
- `READINESS_GATES.md`: status gate pilot/konfirmatori.
- `RUNBOOK.md`: urutan perintah aman.
- `PROTOCOL_FREEZE.json`: kontrak mesin-terbaca.
- `splits/split_registry.json`: registry split yang dibekukan.
