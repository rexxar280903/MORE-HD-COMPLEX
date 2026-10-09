# Tata Kelola Artefak Penelitian Digit Arab

Folder ini menyimpan **manifest dan bukti penelitian**. Snapshot MADBase lokal
berada di direktori yang diabaikan Git karena izin redistribusi belum
dikonfirmasi; lihat `../DATASET_CARD.md`.

## Struktur aktif

```text
research_data/
  manifests/           checksum, inventory, label mapping, data audit (tersedia)
  pilot/               test summary, smoke checks, dan kelak keputusan convergence
  confirmatory_machine/ environment dan benchmark mesin final
  consolidated_pilot/  agregasi smoke yang lolos admission
  analysis_pilot/      tabel statistik smoke; bukan hasil penelitian
  consolidated/        agregasi konfirmatori setelah dijalankan
  analysis/            tabel statistik konfirmatori setelah dijalankan
  reports/              admission/deviation/reproducibility reports
```

Direktori lokal besar yang tidak direncanakan untuk Git:

```text
research_data/raw/
research_data/extracted/
research_data/processed/
research_data/runs/
research_data/local_cache/
```

## Artefak wajib

| Artefak | Format yang disarankan | Commit? |
|---|---|---|
| Raw file manifest | `.sha256`/JSON | Ya, jika tidak membocorkan data terbatas |
| Dataset inventory | JSON/CSV | Ya |
| Label mapping | JSON | Ya |
| Duplicate/leakage audit | CSV + Markdown | Ya, bila sample ID boleh dibagikan |
| Split manifests | JSON | Ya, bila lisensi mengizinkan ID |
| Pilot decisions | JSON/CSV/Markdown | Ya |
| Per-run large arrays | NPY/NPZ | Biasanya tidak; arsip eksternal |
| Consolidated tables | CSV/XLSX | Ya bila ukuran dan izin wajar |
| Analysis report | CSV/JSON/Markdown | Ya |

## Aturan provenance

Setiap artefak turunan harus menyimpan atau dapat ditautkan ke:

- dataset ID, versi, dan raw-manifest hash;
- split-manifest hash;
- preprocessing/feature config hash;
- seed dan namespace random;
- run ID dan attempt;
- code commit;
- environment/dependency lock;
- timestamp dan timezone;
- pembuat script/entry point.

File hasil tidak boleh diedit manual tanpa menyimpan sumber dan log perubahan.
Tabel publikasi dibangun dari output mesin-terbaca, bukan diketik ulang dari
tampilan terminal.

## Data sensitif atau tidak dapat dibagikan

Jika lisensi melarang redistribusi:

1. jangan commit citra/arsip;
2. commit hanya script akuisisi jika terms mengizinkan;
3. simpan checksum dan instruksi memperoleh snapshot;
4. gunakan sample ID pseudonim bila ID sumber sensitif;
5. jelaskan artefak mana yang tersedia kepada reviewer.
