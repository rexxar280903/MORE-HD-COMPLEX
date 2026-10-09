# Penelitian MORE-HD-C pada Digit Arab (MADBase)

**Status:** `SIAP UNTUK PILOT — KONFIRMATORI DIKUNCI SAMPAI BUDGET KONVERGENSI MADBASE BEKU`  
**Versi dokumentasi:** 1.0  
**Terakhir diperbarui:** 2026-10-09

Folder ini adalah dossier terpisah untuk studi MORE-HD/MORE-HD-C pada Modified
Arabic Digits Database (MADBase). Studi MNIST tetap dapat direproduksi melalui
jalur lama; setiap perintah studi Arab harus menyebut `--dataset MADBASE`.

## Putusan kesiapan

- **Pilot teknis siap dijalankan.** Loader, split writer-disjoint, fitur,
  final-test vault, lima arsitektur, baseline, Jalur B, workbook, konsolidasi,
  dan analisis sudah diuji end-to-end.
- **Konfirmatori belum boleh dijalankan.** Guard CLI menolak run
  `CONFIRMATORY` sampai
  `research_data/pilot/convergence_budget_decision.json` berstatus `FROZEN`
  untuk MADBase.
- **Official test tidak dipakai pada pilot.** Pilot memakai holdout penulis dari
  official training pool. Official test baru dimuat secara lazy di tahap
  `final_evaluation` pada run konfirmatori.
- **Data mentah hanya untuk riset lokal dan tidak boleh didistribusikan.**
  Halaman resmi menyediakan unduhan tetapi tidak menyatakan lisensi
  redistribusi secara eksplisit. Kebijakan proyek adalah
  `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`.

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
| Budget optimizer MADBase | `BLOCKED` | pilot konvergensi belum dijalankan |

Metrik smoke hanya bukti teknis dan **bukan hasil penelitian**. Tidak ada metrik
model dari official test konfirmatori yang digunakan untuk memilih desain.

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

Tahap berikutnya yang wajib sebelum konfirmatori:

```powershell
python scripts/run_pilot.py convergence --dataset MADBASE --first-only
python scripts/run_pilot.py convergence --dataset MADBASE --skip-first --parallel 1
python scripts/analyze_pilot_budget.py --dataset MADBASE --tag convergence
python scripts/check_arabic_readiness.py --require-confirmatory
```

Pilot konvergensi merupakan pekerjaan komputasi panjang. Jangan memulai run
konfirmatori hanya dengan menghapus guard atau menyalin budget dari MNIST.

## Peta dokumen

- `PERBAIKAN_DAN_VERIFIKASI_2026-10-09.md`: catatan lengkap pekerjaan dan bukti.
- `DATASET_CARD.md`: identitas, snapshot, audit, lisensi, dan preprocessing.
- `RESEARCH_PROTOCOL.md`: desain eksperimen yang berlaku.
- `STATISTICAL_ANALYSIS_PLAN.md`: estimand, metrik, inferensi, dan admission.
- `READINESS_GATES.md`: status gate pilot/konfirmatori.
- `RUNBOOK.md`: urutan perintah aman.
- `PROTOCOL_FREEZE.json`: kontrak mesin-terbaca.
- `splits/split_registry.json`: registry split yang dibekukan.

