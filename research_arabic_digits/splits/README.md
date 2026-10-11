# Kontrak Split Manifest — MADBase

**Status:** `FROZEN`  
**Versi:** 1.0  
**Tanggal:** 2026-10-09

Folder ini berisi satu manifest pilot, lima manifest konfirmatori, dan registry
hash. Manifest dibuat oleh `scripts/build_madbase_splits.py` dan divalidasi
sebelum training.

## Kontrak

| Bidang | Nilai/arti |
|---|---|
| `schema_version` | `1.0` |
| `dataset` / `dataset_id` | `MADBASE` / `MADBASE-AUC-V1` |
| `dataset_snapshot_sha256` | snapshot loader yang dipin |
| `master_seed` | 42 atau salah satu 101/202/303/404/505 |
| `run_mode` | `PILOT` atau `CONFIRMATORY` |
| `split_policy` | writer-disjoint |
| `train` / `validation` / `test` | source indices, sample IDs, writer IDs, counts, hashes |
| `test_origin` | pilot training-pool holdout atau `OFFICIAL_SOURCE` |
| `duplicate_policy` | retain and report; no silent exclusion |

Lihat manifest JSON aktual untuk struktur kanonik. Jangan membuat parser dari
contoh dokumentasi lama.

## Invariant yang diuji

- train, validation, dan test writer-disjoint;
- jumlah per kelas 1000/100/200;
- label dan sample IDs cocok dengan snapshot;
- seed yang sama menghasilkan manifest bit-identik;
- pairing lintas model/fitur memakai manifest yang sama;
- nestedness berlaku untuk kelas `0..K-1`;
- lima seed konfirmatori memakai official-test sample IDs identik;
- dataset/snapshot mismatch ditolak;
- official test tidak perlu dimuat untuk memvalidasi train/validation metadata.

Fixed confirmatory test sample hash:

```text
fece765dd5f7e16f51d53c2fa9038ba3279d0ec0459060298e26c873bf1d9a2e
```

Registry SHA-256 pada saat freeze:

```text
b6ad3ef7cc95e801afc1aeb3fc07ed7bdadf8443bc806a5da432c532abf27e90
```

Regenerasi yang mengubah hash adalah amendemen protokol dan tidak boleh
dilakukan setelah outcome official test tersedia.

