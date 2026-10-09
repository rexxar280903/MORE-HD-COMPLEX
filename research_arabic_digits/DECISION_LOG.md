# Decision Log — Studi Digit Arab

**Status:** aktif
**Dibuat:** 2026-10-08

Dokumen ini mencatat keputusan, bukan hasil. Setiap keputusan memakai ID stabil,
tanggal dan zona waktu, status, alasan, alternatif, bukti, serta apakah
final-test outcome sudah tersedia.

Entri `LOCKED` adalah audit trail dan tidak mengubah protokol/SAP secara diam-diam.
Keputusan baru hanya operatif setelah dokumen otoritatif yang terdampak diberi
versi atau amendemen dan ditautkan kembali ke ID keputusan ini.

Status keputusan:

- `OPEN`: belum diputuskan;
- `PROPOSED`: rancangan awal, masih dapat berubah sebelum freeze;
- `LOCKED`: keputusan beku untuk fase yang disebut;
- `SUPERSEDED`: diganti keputusan baru yang ditautkan;
- `REOPENED`: dibuka kembali dengan alasan dan konsekuensi eksplisit.

## Keputusan awal

### AD-D001 — Isolasi studi dari protokol MNIST

- Tanggal: 2026-10-08 (Asia/Jakarta)
- Status: `LOCKED`
- Keputusan: seluruh dokumen studi dataset baru ditempatkan di
  `research_arabic_digits/`; dokumen/protokol MNIST di root tidak diubah menjadi
  dokumen digit Arab.
- Alasan: protokol MNIST sudah memiliki keputusan, split, status, dan hasil yang
  tidak sah bila diwariskan ke dataset lain.
- Alternatif: menyalin atau mengedit dokumen MNIST di tempat; ditolak karena
  mengaburkan provenance.
- Outcome digit Arab tersedia: tidak.
- Bukti: folder dan `README.md` studi ini.

### AD-D002 — Makna sementara “digit Arab”

- Tanggal: 2026-10-08 (Asia/Jakarta)
- Status: `SUPERSEDED` oleh AD-D006
- Keputusan sementara: dokumentasi memakai Arabic-Indic `٠١٢٣٤٥٦٧٨٩` sebagai
  asumsi kerja, bukan fakta dataset.
- Syarat penguncian: nama resmi, sumber, versi, lisensi, citation, mapping label,
  contoh per kelas, dan checksum harus tersedia.
- Alternatif: Western Arabic `0–9`, Eastern Arabic/Persian `۰–۹`, atau dataset
  lain; tetap terbuka sampai sumber ditentukan.
- Outcome digit Arab tersedia: tidak.

### AD-D006 — MADBase sebagai dataset kerja awal

- Tanggal: 2026-10-08 (Asia/Jakarta)
- Status: `LOCKED` untuk akuisisi, audit, dan pilot teknis; **belum mengizinkan run konfirmatori**
- Fase yang dipengaruhi: data acquisition, adapter, audit, dan pilot.
- Keputusan: gunakan snapshot resmi Modified Arabic Digits Database (MADBase)
  dari American University in Cairo, dataset ID `MADBASE-AUC-V1`.
- Alasan: 70.000 citra seimbang, 28×28, official split 60.000/10.000,
  writer-disjoint, dan format sengaja disejajarkan dengan MNIST.
- Batas: halaman resmi menyediakan unduhan publik tetapi tidak menampilkan teks
  lisensi standar; raw data tetap lokal dan konfirmasi penggunaan/redistribusi
  dibutuhkan sebelum publikasi konfirmatori.
- Bukti: `DATASET_CARD.md`, `research_data/manifests/raw_files.sha256`,
  `dataset_inventory.json`, dan `data_audit_report.md`.
- Final-test outcome sudah tersedia?: tidak; tidak ada model yang dijalankan.
- Dampak: AD-O001 dan AD-O003 diselesaikan. AD-O002 kemudian ditutup oleh
  kebijakan konservatif AD-D007.
- Bukti: `DATASET_CARD.md`.

### AD-D003 — Tidak mewarisi angka protokol MNIST

- Tanggal: 2026-10-08 (Asia/Jakarta)
- Status: `LOCKED`
- Keputusan: ukuran split, daftar seed, K, jumlah run, budget optimizer, runtime,
  dan status readiness MNIST tidak menjadi nilai default final studi ini.
- Alasan: nilai harus mengikuti inventaris dataset baru dan pilot yang relevan.
- Outcome digit Arab tersedia: tidak.
- Bukti: protocol/SAP versi awal dan keputusan budget gate AD-D009.

### AD-D004 — Batas perubahan tahap dokumentasi awal

- Tanggal: 2026-10-08 (Asia/Jakarta)
- Status: `LOCKED` untuk versi 0.1
- Keputusan: perubahan awal hanya membuat dossier penelitian. Loader, pipeline,
  split, workbook, dan eksperimen belum diimplementasikan atau dijalankan.
- Alasan: dataset belum disediakan dan kontrak datanya belum diketahui.
- Outcome digit Arab tersedia: tidak.
- Bukti: audit workspace dan `RUNBOOK.md` §0.

### AD-D005 — Perbandingan dengan MNIST

- Tanggal: 2026-10-08 (Asia/Jakarta)
- Status: `LOCKED`
- Keputusan: perbandingan lintas dataset dilaporkan deskriptif; sampel tidak
  diperlakukan berpasangan dan perbedaan tidak disebut efek kausal dataset.
- Batas: daftar perbedaan protokol dan output yang dibandingkan tercantum pada
  SAP v1.0; tidak ada inferensi kausal lintas dataset.
- Outcome digit Arab tersedia: tidak.

### AD-D007 — Kebijakan lisensi konservatif

- Tanggal: 2026-10-09 (Asia/Jakarta)
- Status: `LOCKED`
- Keputusan: karena halaman resmi tidak menyatakan lisensi redistribusi secara
  eksplisit, dataset dipakai hanya untuk riset lokal; arsip, BMP, cache array,
  dan contoh mentah tidak di-commit atau dibagikan.
- Alasan: unduhan publik bukan bukti izin redistribusi.
- Bukti: `DATASET_CARD.md`, `.gitignore`, `PROTOCOL_FREEZE.json`.
- Outcome official test tersedia?: tidak ada metrik model.

### AD-D008 — Desain, split, dan matriks dibekukan

- Tanggal: 2026-10-09 (Asia/Jakarta)
- Status: `LOCKED`
- Keputusan: K=3..10, fitur PCA/Hu/Zernike, model A/D/B/C/M, seed pilot 42,
  seed final 101/202/303/404/505, dan ukuran 1000/100/200 per kelas. Split
  train/validation writer-disjoint; official-test subset tetap lintas seed.
- Bukti: `PROTOCOL_FREEZE.json`, `splits/split_registry.json`, protocol/SAP v1.0.
- Outcome official test tersedia?: tidak ada metrik model konfirmatori.

### AD-D009 — Budget konfirmatori harus berasal dari MADBase

- Tanggal: 2026-10-09 (Asia/Jakarta)
- Status: `LOCKED`; nilai budget masih menunggu pilot
- Keputusan: budget 840/1000 dari MNIST tidak diwariskan. CLI menolak
  konfirmatori sampai pilot konvergensi MADBase menghasilkan
  `convergence_budget_decision.json` berstatus `FROZEN`.
- Bukti: `core/cli.py`, `scripts/analyze_pilot_budget.py`, readiness report.
- Outcome official test tersedia?: tidak.

### AD-D010 — Pipeline MADBase siap untuk pilot

- Tanggal: 2026-10-09 (Asia/Jakarta)
- Status: `LOCKED`
- Keputusan: pipeline dataset-aware, lazy official-test vault, fitur,
  workbook, baseline, Jalur B, konsolidasi, dan analisis diterima untuk pilot.
- Bukti: test summary, smoke check, consolidated/analysis pilot outputs, dan
  `PERBAIKAN_DAN_VERIFIKASI_2026-10-09.md`.
- Outcome official test tersedia?: tidak; smoke memakai holdout training pool.

## Keputusan terbuka

| ID | Pertanyaan | Pemilik | Target fase | Status |
|---|---|---|---|---|
| AD-O001 | Dataset resmi apa yang dipakai? | Pemilik studi | Sebelum implementasi | `CLOSED — AD-D006` |
| AD-O002 | Apakah lisensi mengizinkan penyimpanan/redistribusi? | Pemilik studi | Sebelum publikasi | `CLOSED — AD-D007: local-only, no raw redistribution` |
| AD-O003 | Apakah official split dan writer IDs tersedia? | Pemilik studi | Audit data | `CLOSED — tersedia dan writer-disjoint` |
| AD-O004 | Class regimes/K apa yang dipakai? | Pemilik studi | Protocol freeze | `CLOSED — AD-D008` |
| AD-O005 | Model/fitur/ablation/referensi mana yang dipertahankan? | Pemilik studi | Sebelum pilot | `CLOSED — AD-D008` |
| AD-O006 | Seed dan ukuran per split berapa? | Pemilik studi | Setelah inventaris | `CLOSED — AD-D008` |
| AD-O007 | Bagaimana budget optimizer dipilih? | Pemilik studi | Sebelum konfirmatori | `METHOD CLOSED / VALUE PENDING — AD-D009` |
| AD-O008 | Format artefak dan workbook final? | Pemilik studi | Sebelum confirmatory | `CLOSED — AD-D010` |

## Template entri berikutnya

```text
### AD-Dxxx — Judul

- Tanggal/waktu/zona:
- Status:
- Fase yang dipengaruhi:
- Keputusan:
- Alasan:
- Alternatif yang dipertimbangkan:
- Bukti:
- Final-test outcome sudah tersedia?:
- Dampak pada protocol/SAP/code/run:
- Menggantikan/digantikan oleh:
```
