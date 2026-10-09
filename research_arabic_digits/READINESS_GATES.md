# Research Readiness Gates — MADBase

**Status keseluruhan:** `PILOT READY / CONFIRMATORY BLOCKED AT AD-G3-03`  
**Versi:** 1.0  
**Tanggal:** 2026-10-09

`CLOSED` berarti keputusan dan bukti tersedia. `BLOCKED` menghalangi
konfirmatori. G6 baru dapat ditutup setelah seluruh run konfirmatori selesai.

## G0 — Dataset

| ID | Status | Keputusan | Bukti |
|---|---|---|---|
| AD-G0-01 | `CLOSED` | Snapshot MADBase resmi dipin dengan SHA-256 | `DATASET_CARD.md`, `raw_files.sha256` |
| AD-G0-02 | `CLOSED WITH RESTRICTION` | Tidak ada lisensi redistribusi eksplisit; raw lokal saja | `DATASET_CARD.md`, `PROTOCOL_FREEZE.json` |
| AD-G0-03 | `CLOSED` | Label 0–9 = Arabic-Indic `٠١٢٣٤٥٦٧٨٩` | `label_mapping.json` |
| AD-G0-04 | `CLOSED` | 70.000 citra 28×28, seimbang, seluruhnya dapat didecode | `dataset_inventory.json` |
| AD-G0-05 | `CLOSED` | Writer overlap 0; kandidat duplikat dipertahankan dan dilaporkan | `duplicate_audit.csv`, `DATASET_CARD.md` |
| AD-G0-06 | `CLOSED` | Official split dan writer-disjoint subsplit dapat direkonstruksi | `splits/*.json` |

## G1 — Split dan anti-leakage

| ID | Status | Keputusan | Bukti |
|---|---|---|---|
| AD-G1-01 | `CLOSED` | 1000/100/200 sampel per kelas | `PROTOCOL_FREEZE.json` |
| AD-G1-02 | `CLOSED` | Pilot 42; konfirmatori 101/202/303/404/505 | `core/constants.py` |
| AD-G1-03 | `CLOSED` | Pilot test dari training pool; official test lazy dan vault-gated | `core/datasets.py`, `core/evaluation.py`, tes |
| AD-G1-04 | `CLOSED` | Sample/split hash paired dan mismatch ditolak | smoke matrix, `core/pairing.py` |
| AD-G1-05 | `CLOSED` | K kumulatif/nested 3..10 | `PROTOCOL_FREEZE.json` |
| AD-G1-06 | `CLOSED` | Retry memakai identitas sama; satu completed attempt diterima | `core/run_status.py`, tes |

## G2 — Preprocessing dan fitur

| ID | Status | Keputusan | Bukti |
|---|---|---|---|
| AD-G2-01 | `CLOSED` | Polaritas dibalik, 28×28 dipertahankan, tanpa augmentasi | `DATASET_CARD.md` |
| AD-G2-02 | `CLOSED` | PCA 8 komponen, full SVD, fit train-only | tes pipeline MADBase |
| AD-G2-03 | `CLOSED` | Hu 7 signed-log + kanal padding | tes fitur dan smoke |
| AD-G2-04 | `CLOSED` | Zernike delapan term pada geometri 28×28 | tes fitur dan workbook smoke |
| AD-G2-05 | `CLOSED` | Finite/NaN/constant/zero-norm guards diuji | unit tests |
| AD-G2-06 | `CLOSED` | Array dan transformer menyimpan SHA-256/provenance | run manifests |

## G3 — Desain dan statistik

| ID | Status | Keputusan | Bukti |
|---|---|---|---|
| AD-G3-01 | `CLOSED` | Pertanyaan, estimand, outcome, unit replikasi dibekukan | protocol/SAP v1.0 |
| AD-G3-02 | `CLOSED` | A/D, B/C, M; PCA/Hu/Zernike; K 3..10 | `PROTOCOL_FREEZE.json` |
| AD-G3-03 | `BLOCKED` | Budget optimizer harus berasal dari pilot konvergensi MADBase | decision file belum ada |
| AD-G3-04 | `CLOSED` | Ablation, NC/LR/chance, dan planned contrasts dibekukan | SAP v1.0 |
| AD-G3-05 | `CLOSED` | CI, effect size, sign-flip, Holm, dan batas n=5 ditulis | SAP v1.0 |
| AD-G3-06 | `CLOSED` | Confusion lengkap; fokus 6↔9 dipra-spesifikasi sebagai sekunder | SAP v1.0 |
| AD-G3-07 | `CLOSED` | Lintas dataset hanya deskriptif | protocol/SAP v1.0 |
| AD-G3-08 | `CLOSED` | Lima seed dan 200 test/kelas adalah batas feasibility eksplisit | SAP v1.0 |
| AD-G3-09 | `CLOSED` | Fixed labels, zero division, tie, dan separation dibekukan | `PROTOCOL_FREEZE.json`, tes |

## G4 — Implementasi

| ID | Status | Keputusan | Bukti |
|---|---|---|---|
| AD-G4-01 | `CLOSED` | Loader dan snapshot validation | `core/madbase.py`, tes |
| AD-G4-02 | `CLOSED` | Artefak generik `sample_id/source_index`; alias MNIST tetap kompatibel | `core/data_pipeline.py`, `core/workbook.py` |
| AD-G4-03 | `CLOSED` | Manifest deterministic dan writer-aware | `core/splits.py`, registry |
| AD-G4-04 | `CLOSED` | `--dataset` eksplisit dan mismatch ditolak | CLI/config tests |
| AD-G4-05 | `CLOSED` | Suite regresi lama tetap lulus; skip hanya data MNIST lokal yang tidak ada | `test_summary.json` |
| AD-G4-06 | `CLOSED` | Lima arsitektur lolos smoke dan vault tidak dibuka dini | `smoke_check.json` |
| AD-G4-07 | `CLOSED` | Environment lock dan pemeriksaan disk tersedia | requirements lock, readiness report |

## G5 — Artefak pra-run

| ID | Status | Keputusan | Bukti |
|---|---|---|---|
| AD-G5-01 | `CLOSED` | Run ID dataset-aware dan attempt auditable | smoke manifests |
| AD-G5-02 | `CLOSED` | Tiga template workbook MADBase tervalidasi | `research_data/*.xlsx` |
| AD-G5-03 | `CLOSED` | Status RUNNING/FAILED/COMPLETED dan retry diuji | test suite |
| AD-G5-04 | `CLOSED` | Consolidator dan analysis MADBase lulus pada batch pilot | `consolidated_pilot/`, `analysis_pilot/` |
| AD-G5-05 | `CLOSED` | Snapshot, split, environment, config, dan protocol provenance tersimpan | freeze/readiness/run manifests |

## G6 — Eksekusi final

Seluruh item berikut `NOT STARTED`: kelengkapan 450 run dan 90 Jalur B,
admission final, tabel primer/ablation/baseline, deviation log final, pengisian
hasil, dan paket reproduksi. G6 tidak menghalangi pilot, tetapi wajib ditutup
sebelum penelitian dinyatakan selesai.

## Ringkasan

| Gate | Closed | Blocked | Catatan |
|---|---:|---:|---|
| G0 | 6 | 0 | redistribusi raw dilarang oleh kebijakan risiko |
| G1 | 6 | 0 | official test lazy dan paired |
| G2 | 6 | 0 | tiga fitur tervalidasi |
| G3 | 8 | 1 | hanya budget konvergensi MADBase |
| G4 | 7 | 0 | implementasi dan smoke lulus |
| G5 | 5 | 0 | keluaran hilir pilot lulus |
| G6 | 0 | 0 | belum dimulai; pasca-konfirmatori |

## Log perubahan

| Tanggal | Perubahan |
|---|---|
| 2026-10-08 | Snapshot dan audit MADBase dibuat. |
| 2026-10-09 | Pipeline dataset-aware, split, vault, fitur, workbook, tes, smoke, baseline, Jalur B, konsolidasi, dan analisis ditutup. AD-G3-03 sengaja tetap memblokir konfirmatori sampai pilot konvergensi selesai. |

