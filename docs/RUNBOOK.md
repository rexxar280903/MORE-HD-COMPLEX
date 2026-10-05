# Runbook — menjalankan eksperimen MORE-HD-C dari awal

Panduan ini untuk menjalankan ulang seluruh training dari nol pada mesin Anda. Semua
keputusan protokol sudah dikunci di `MORE_HD_RESEARCH_READINESS_GATES (1).md`,
`MORE_HD_STATISTICAL_ANALYSIS_PLAN.md` (v2.0) dan `Pseudocode_2x3_manual_runs.md`;
runbook ini hanya urutan kerja.

## 0. Ringkasan beban kerja

| Track | Model | Kondisi | Seed | Run | Entry point |
|---|---|---|---:|---:|---|
| Primer | A = MORE-HD, D = MORE-HD-C | K=3..10 × PCA/HU/ZERNIKE | 5 | 240 | `main_train.py` |
| Ablation | B = MORE-HD-60P, C = MORE-HD-C-FixedRZ | K∈{3,6,10} × 3 fitur | 5 | 90 | `main_ablation.py` |
| Referensi MORE | M = MORE-REPRO | K=3..10 × 3 fitur | 5 | 120 | `main_more_reference.py` |
| Jalur B (sekunder) | A dan D | K∈{3,6,10} × 3 fitur | 5 | 90 | `main_selected_clustering.py` |
| Baseline klasik | NC, LR, chance | 120 cell | 5 | 240 fit | `main_classical_baseline.py` |

Proyeksi komputasi dari timing pilot (mesin referensi: Intel Xeon 2,1 GHz, 1 thread per proses):
lihat `research_data/pilot/timing_projection.json` (± 32 CPU-jam untuk 450 run + 90 Jalur B pada
budget 1.000). Dengan 4 proses paralel kira-kira 8–10 jam. Ulangi microbenchmark di mesin Anda
(langkah 2) karena kecepatan bergantung CPU.

## 1. Lingkungan

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt          # runtime + library acuan untuk test
python scripts/download_mnist.py              # file MNIST torchvision, diverifikasi MD5
python -m pytest tests -q                     # 73 test; semuanya harus lulus (~3 menit)
```

Setiap entry point mengunci `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` sebelum NumPy dimuat
(G4-01). Paralelisme dilakukan dengan menjalankan beberapa proses sekaligus, bukan multithread.

## 2. Verifikasi mesin (sekali, sebelum run konfirmatori)

```bash
python scripts/run_pilot.py smoke   --parallel 4   # seed 42, PILOT ONLY
python scripts/check_pilot_runs.py --tag smoke     # harus "ALL SMOKE CRITERIA OK"
python scripts/run_pilot.py timing  --parallel 1
python scripts/timing_projection.py                # proyeksi beban kerja di mesin ini
python scripts/circuit_microbenchmark.py           # biaya per eksekusi A/B/C/D/M (sheet 18)
```

Budget final **tidak** ditentukan ulang di mesin Anda: `FINAL_MAX_NFEV_CLUSTERING` dan
`FINAL_MAX_NFEV_SUPERVISED` sudah dibekukan di `core/constants.py` dari pilot konvergensi
seed 42 (`research_data/pilot/convergence_budget_decision.json`). Entry point menolak run
CONFIRMATORY dengan budget lain.

## 3. Run konfirmatori

Daftar perintah lengkap (450 run) dan statusnya:

```bash
python scripts/print_run_plan.py --n-parallel 4            # semua perintah + status per run_uid
python scripts/print_run_plan.py --pending --n-parallel 4  # hanya yang belum COMPLETED
```

Contoh satu run:

```bash
python main_train.py --architecture MORE-HD   --feature PCA --k 3 --seed 101 --run-mode CONFIRMATORY --n-parallel-declared 4
python main_train.py --architecture MORE-HD-C --feature PCA --k 3 --seed 101 --run-mode CONFIRMATORY --n-parallel-declared 4
python main_ablation.py --model MORE-HD-60P   --feature PCA --k 3 --seed 101 --run-mode CONFIRMATORY --n-parallel-declared 4
python main_more_reference.py                 --feature PCA --k 3 --seed 101 --run-mode CONFIRMATORY --n-parallel-declared 4
```

Aturan:

* `--n-parallel-declared` = jumlah run yang **sengaja** Anda jalankan bersamaan (wajib untuk
  CONFIRMATORY; hanya metadata audit).
* Satu proses = satu kondisi = satu folder `runs/<nama_run>/`. Folder yang sudah ada tidak
  pernah ditimpa.
* Urutan yang disarankan: jalankan run A (MORE-HD) suatu cell lebih dulu atau bersamaan dengan
  D/B/C/M; run B/C/M otomatis mencocokkan hash array input dengan run A bila A sudah ada
  (`paired_input_check`).
* Hasil official test setiap run baru dibaca di tahap `final_evaluation`; jangan membuka
  `logs/metrics_final.json` untuk mengubah keputusan apa pun (SAP §13).

## 4. Bila run gagal

* Exception → `logs/run_status.json` berstatus `FAILED` beserta traceback.
* Crash keras (listrik mati, kill) → status tertinggal `RUNNING`; tandai dulu:
  `python scripts/mark_run_failed.py runs/<nama_run> --reason "power loss"`.
* Ulangi dengan `--attempt 2` (folder baru `..._attempt2`; seed, split, pair, inisialisasi sama).
  Attempt baru ditolak bila ada attempt sebelumnya yang belum `FAILED`.

## 5. Jalur B dan baseline klasik

```bash
# Jalur B: setelah run sumber COMPLETED; scope K in {3,6,10}, A dan D, 5 seed (90 eksekusi)
python main_selected_clustering.py --source-run-dir runs/cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed101 --n-parallel-declared 4
# Baseline klasik (butuh tahap DATA_PIPELINE run A selesai)
python main_classical_baseline.py --run-mode CONFIRMATORY
```

## 6. Konsolidasi dan analisis

```bash
python - <<'EOF'
from pathlib import Path
from core import constants as C
from core.workbook import consolidate, discover_runs
for track, tpl, out in [
    ("PRIMARY", C.MASTER_SPREADSHEET_PATH, "research_data/consolidated/primary_240runs.xlsx"),
    ("ABLATION", C.ABLATION_SPREADSHEET_PATH, "research_data/consolidated/ablation_90runs.xlsx"),
    ("MORE_REFERENCE", C.MORE_REFERENCE_SPREADSHEET_PATH, "research_data/consolidated/more_reference_120runs.xlsx"),
]:
    print(consolidate(tpl, discover_runs("runs", track, "CONFIRMATORY"), out, track))
EOF
python analysis/run_analysis.py --run-mode CONFIRMATORY     # tabel SAP di research_data/analysis/
```

Konsolidasi selalu dibangun ulang dari template + artefak (idempoten) dan menolak duplikat
`run_uid`+`attempt`.

## 7. Kewajiban pelaporan setelah eksekusi (Gate D)

Lihat daftar `P-01…` di `MORE_HD_RESEARCH_READINESS_GATES (1).md`: semua run `COMPLETED`, tidak
ada duplikat/NaN/history terpotong, tabel SAP (primer, ablation, MORE, baseline, 6↔9) dilaporkan,
dan paragraf hasil ditulis mengikuti batas klaim di `docs/METHODS_DRAFT.md`.
