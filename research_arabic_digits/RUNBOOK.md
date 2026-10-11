# Runbook MADBase

**Status:** `SELESAI — G6 CLOSED`  
**Versi:** 1.1  
**Tanggal:** 2026-10-11

Semua perintah dijalankan dari root repositori. Gunakan environment yang cocok
dengan `research_arabic_digits/requirements-lock-py313.txt`.

## 1. Pemeriksaan awal

```powershell
python scripts/audit_madbase.py
python scripts/check_arabic_readiness.py
python -m pytest -q
```

Keluaran kesiapan harus menyatakan `pilot_ready: true`. Nilai
`confirmatory_ready: false` masih benar sebelum pilot konvergensi selesai.

## 2. Membuat ulang split dan template

Hanya lakukan ini bila kontrak studi sengaja diubah; perubahan hash merupakan
amendemen protokol.

```powershell
python scripts/build_madbase_splits.py
python scripts/build_madbase_workbook_templates.py
python scripts/check_arabic_readiness.py
```

Split memisahkan writer train, validation, dan pilot holdout. Lima manifest
konfirmatori memakai official-test sample IDs yang sama, sehingga pairing lintas
model dan seed dapat diaudit.

## 3. Smoke test

```powershell
python scripts/run_pilot.py smoke --dataset MADBASE --parallel 1
python scripts/check_pilot_runs.py --dataset MADBASE --tag smoke
```

Smoke memakai 20/6/8 sampel per kelas dari training pool dan budget 100 NFEV.
Angkanya hanya menunjukkan jalur kode bekerja.

## 4. Baseline, Jalur B, konsolidasi, dan analisis pilot

```powershell
python main_classical_baseline.py --dataset MADBASE --run-mode PILOT --seeds 42 --features PCA --ks 3 --sizes 20 6 8 --pilot-tag smoke
python main_selected_clustering.py --source-run-dir runs/arabic_digits/cls-0-1-2_ntrain20_nval6_ntest8_PCA_MORE-HD_seed42_smoke --n-parallel-declared 1
python scripts/consolidate_runs.py --dataset MADBASE --run-mode PILOT --pilot-tag smoke
python analysis/run_analysis.py --dataset MADBASE --run-mode PILOT --pilot-tag smoke
```

## 5. Pilot konvergensi wajib

Tahap ini telah selesai pada 2026-10-09. Jangan mengulang run yang sudah
COMPLETED; keputusan 900/940 telah dibekukan dari 14 kondisi lengkap.

Jalankan satu kondisi pertama untuk memeriksa mesin, lalu lanjutkan matriks
sisanya. Tahap ini dapat memakan waktu lama.

```powershell
python scripts/run_pilot.py convergence --dataset MADBASE --first-only
python scripts/run_pilot.py convergence --dataset MADBASE --skip-first --parallel 1
python scripts/analyze_pilot_budget.py --dataset MADBASE --tag convergence
python scripts/check_arabic_readiness.py --require-confirmatory
```

Budget dipilih oleh aturan plateau yang telah ditulis sebelumnya; jangan ubah
berdasarkan accuracy test. File keputusan harus berstatus `FROZEN` dan menyebut
dataset `MADBASE`.

## 6. Konfirmatori

Daftar perintah tanpa mengeksekusi:

```powershell
python scripts/print_run_plan.py --dataset MADBASE --pending --n-parallel 1
```

Entry point akan menolak run sebelum budget MADBase beku. Sesudah gate lulus,
jalankan satu proses per run ID. Jangan menimpa attempt yang ada; retry memakai
run UID, seed, dan konfigurasi yang sama. Official test hanya boleh dimuat di
tahap `final_evaluation`.

## 7. Jika run gagal

1. Pertahankan folder, traceback, status, dan stage terakhir.
2. Tandai kegagalan teknis; jangan memilih retry dari accuracy/F1 test.
3. Ulangi run ID dan seed yang sama sebagai attempt berikutnya.
4. Catat perubahan metode sebagai amendemen sebelum melanjutkan.
5. Hanya satu attempt `COMPLETED` yang boleh masuk agregasi.

## 8. Konsolidasi dan analisis final

```powershell
python scripts/consolidate_runs.py --dataset MADBASE --run-mode CONFIRMATORY
python analysis/run_analysis.py --dataset MADBASE --run-mode CONFIRMATORY
```

Sebelum pelaporan, pastikan admission report tidak memiliki mismatch dataset,
snapshot, split, input hash, pair hash, atau completed-attempt ganda.

## 9. Batas distribusi

Jangan commit atau membagikan arsip, BMP, cache array, atau contoh mentah
MADBase. Bagikan hanya kode, checksum, manifest ID, konfigurasi, dan hasil
turunan yang tidak merekonstruksi data mentah sampai izin resmi tersedia.

## 10. Penutupan G6, laporan, paper, dan dashboard

Semua langkah di bawah hanya membaca run/workbook yang sudah ada (tidak melatih ulang) dan tidak menyentuh file
studi MNIST.

```powershell
python research_arabic_digits/tools/finalize_g6.py                  # audit G6 + REPRODUCIBILITY_MANIFEST.json
python research_arabic_digits/paper/build/extract_workbooks.py     # sheet ringkas workbook -> paper/build/data/*.csv
python research_arabic_digits/paper/build/collect_jalurb.py        # 90 run Jalur B -> data/jalurB_runs.csv
python research_arabic_digits/paper/build/mechanism/pack_inputs.py # input analisis mekanistik (eksploratif)
python research_arabic_digits/paper/build/mechanism/mechanism_analysis_madbase.py --src <folder ekstrak mech_inputs.tar.gz> --out research_arabic_digits/paper/build/mechanism --workers 6
python research_arabic_digits/paper/build/analyze_madbase.py       # numbers.json, tables.json, runs_all.csv
python research_arabic_digits/paper/build/write_results.py         # RESULTS.md
python research_arabic_digits/paper/build/make_figures.py          # paper/figures/id/*.png
python research_arabic_digits/paper/build/build_paper.py           # paper/MORE-HD-C_MADBase_paper_ID.docx
python research_arabic_digits/research_data/dashboard/build_data.py  # data dashboard (atau refresh_data.ps1)
```

`build_paper.py` memakai `MML2OMML.XSL` milik Microsoft Office bila ada; bila tidak, perlu `pandoc` di PATH.
