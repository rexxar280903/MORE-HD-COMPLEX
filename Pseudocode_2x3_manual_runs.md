# Pseudocode — Pilot Training Pipeline MORE-HD-C
 
Dokumen ini adalah **alur logika (pseudocode)**, bukan kode Python yang bisa dijalankan.
Tujuannya untuk direview dulu sebelum implementasi asli ditulis.

**Source of truth teknis:** seluruh desain implementasi eksperimen utama A/D, Jalur A/B, dan targeted ablation B/C berada di file ini.
 
---
 
## 0. Gambaran Umum Alur
 
```
CONFIG
  -> DATA_PIPELINE            (+ simpan artefak: pca, scaler_params, X/y transformed arrays train/val/test)
  -> CORRELATION_MATRIX       (+ simpan artefak: correlation_matrix)
  -> MODEL_SETUP              (+ simpan artefak: initial_params)
  -> CLUSTERING_LOOP          (+ simpan artefak: pair_manifest.json, pair_stats.json, clustering_log.jsonl, clustering_params.bin)
  -> QUANTUM_LABEL_EXTRACTION (+ simpan artefak: quantum_labels)
  -> SUPERVISED_LOOP          (+ simpan artefak: supervised_log.jsonl, supervised_params.bin)
  -> FINAL_EVALUATION         (+ simpan artefak: metrics_final, confusion_matrix)
  -> SAVE_ARTIFACT_BUNDLE     (kumpulkan semua + manifest config.json)
```
 
Prinsip: setiap tahap adalah fungsi terpisah, menerima output tahap sebelumnya,
dan langsung menuliskan artefaknya sendiri ke folder run — supaya kalau
program berhenti di tengah, tahap sebelumnya tidak perlu diulang.

**Update (G0-01, sesi 2026-09-22 — three-way split):** Pipeline sekarang membedakan
tiga split data secara eksplisit: **train** (dipakai COBYLA untuk optimasi),
**validation** (dipakai untuk semua monitoring pasif dan pemilihan checkpoint,
baik di `CLUSTERING_LOOP`, `SUPERVISED_LOOP`, maupun pemilihan checkpoint deterministik Jalur B), dan
**official test** (HANYA dipanggil oleh `FINAL_EVALUATION`, setelah protokol
dibekukan). `CLUSTERING_LOOP` dan `SUPERVISED_LOOP` tidak lagi menerima
`X_test`/`y_test` sama sekali di signature-nya — lihat bagian 2, 5, 7, 9, dan 10.

**Update (G0-03, sesi 2026-09-22 — ukuran dataset final dikunci):** Protokol
eksperimen final mengunci `n_train_per_class=1000`, `n_val_per_class=100`,
`n_test_per_class=200` — lihat bagian 1 (`CONFIG`). Ini menggantikan nilai
pilot sebelumnya (100/20/20) di seluruh dokumen ini, termasuk contoh `run_id`
dan pemanggilan `MAIN(config)`/Jalur B. **Revisi G0-01:** aturan lama
"`n_val_per_class = n_test_per_class`" (proporsi) DICABUT dan diganti angka
independen `n_val_per_class = 100` (separuh dari `n_test_per_class`), dengan
alasan validation hanya dipakai untuk monitoring/checkpoint selection
(bukan klaim akhir publikasi) sehingga tidak butuh presisi statistik setara
test set — lihat rasionalisasi di `MORE_HD_RESEARCH_READINESS_GATES.md`
log keputusan G0-03.
 

**Update (G1-02, sesi 2026-09-26 — terminologi dan provenance optimasi):**
Setiap pemanggilan fungsi objektif COBYLA sekarang disebut **objective-function
evaluation**, bukan iterasi. Counter utama adalah `eval_id`; progress optimizer yang
dilaporkan callback memakai `callback_id` terpisah; jumlah evaluasi resmi diambil dari
`result.nfev`. Titik akhir optimizer (`result.x`) disimpan terpisah dari titik terbaik
yang pernah teramati pada log objective (`best_observed_point`). Untuk clustering dan
supervised, hasil terminasi COBYLA juga wajib menyimpan `success`, `status`,
`message`, `fun`, dan `nfev`. Nama konfigurasi budget diubah menjadi
`max_nfev_clustering` dan `max_nfev_supervised`. Jalur B menyimpan
`selected_eval_id` yang dihasilkan fungsi selector deterministik, bukan "iteration". Spreadsheet per-run wajib memakai header
`Eval ID` / `Objective Evaluation` dan tidak boleh menyebut setiap objective call
sebagai `Iteration`.

**Update (G0-04, sesi 2026-09-26 — protokol multi-seed konfirmatori dikunci):**
Seed `42` hanya digunakan untuk `PILOT` (smoke test, debugging, pilot konvergensi,
dan penetapan budget COBYLA) dan **tidak pernah masuk agregasi hasil final**.
Eksperimen `CONFIRMATORY` menggunakan tepat lima seed yang dipra-tetapkan:
`[101, 202, 303, 404, 505]`. Desain tetap mempunyai 48 kondisi primer
(`2 architecture × 3 feature_method × 8 skenario K`), tetapi masing-masing
direplikasi pada lima seed sehingga totalnya **240 confirmatory runs**.

Pada setiap seed, identitas sampel MNIST mentah untuk train, validation, dan
official test dikunci dalam `splits/seed<SEED>.json`. Manifest split yang sama
dipakai oleh PCA, HU, ZERNIKE, MORE-HD, dan MORE-HD-C. Split juga bersifat
**nested terhadap K**: sampel digit yang sudah ada pada K lebih kecil tidak
diacak ulang ketika K bertambah; skenario K+1 hanya menambahkan sampel kelas
baru. Dengan demikian perubahan K tidak tercampur dengan resampling ulang
kelas yang sudah ada.

`condition_id` tetap R001–R048 dan merepresentasikan kondisi eksperimen,
bukan replikasi. Identitas eksekusi unik adalah
`run_uid = condition_id + "-S" + seed`, misalnya `R001-S101`.

**Update (G1-05/G1-07, sesi 2026-09-27 — targeted ablation dikunci):**
Eksperimen utama tetap `2 × 3 × 8` dengan 240 run konfirmatori A=MORE-HD dan
D=MORE-HD-C. Untuk memisahkan confound jumlah parameter dan akses state kompleks,
ditambahkan track ablation terpisah hanya pada `K={3,6,10}`, seluruh
`PCA/HU/ZERNIKE`, dan lima seed konfirmatori yang sama. A dan D **tidak dirun ulang**;
model baru B=MORE-HD-60P (6 layer RY-only, 60 trainable) dan
C=MORE-HD-C-FixedRZ (3 layer, 30 trainable RY + 30 fixed non-zero RZ)
menghasilkan 90 run tambahan. Total unique confirmatory executions menjadi 330.
Primary A/D diselaraskan ke structured paired initialization agar reuse A/D valid.
Spesifikasi penuh ablation berada di **Bagian 11** dokumen ini; primary `MAIN(config)` dan identity R001-R048 tidak diubah oleh keputusan ini.

**Update (G1-01/G1-06, sesi 2026-09-27 — protokol pilot budget dan fase simplex COBYLA):**
Smoke test dikunci pada `max_nfev = 10` dan pilot konvergensi pada cap
`max_nfev = 300` (clustering dan supervised), `cobyla_tol = 1e-4`,
`cobyla_rhobeg = 1.0` (ditulis eksplisit), seed 42. Definisi anggaran adil:
**total `nfev` sama** lintas arsitektur; jumlah evaluasi pasca-simplex dilaporkan
sebagai deskriptif. Setiap objective evaluation diberi kolom `phase`
(`initial_simplex` untuk `eval_id <= n_params`, `optimization` sesudahnya).
Jalur B hanya boleh memilih `eval_id` dengan `phase == "optimization"`.
Spesifikasi lengkap ada di **§1.1**.

**Update (G1-09, sesi 2026-09-27 — lingkup urutan kelas kumulatif):**
Skenario K tetap memakai urutan kumulatif `[0..K-1]` **tanpa** subset kelas acak,
demi komparabilitas dengan FRD-09 dan agar protokol paired/nested split G0-04 tidak
dibuka ulang. Tidak ada perubahan pada `CONFIG`, `DATA_PIPELINE`, identitas run
R001–R048, maupun jumlah run. Confound antara jumlah kelas K dan identitas digit yang
ditambahkan dinyatakan sebagai keterbatasan: kontras A vs D pada K yang sama tetap
bersih, sedangkan tren terhadap K dan interaksi arsitektur×K ditafsirkan spesifik
terhadap urutan kumulatif ini (lihat `MORE_HD_STATISTICAL_ANALYSIS_PLAN.md` §10.1).
Artefak `logs/confusion_matrix.npy`, transformer fitur, dan split manifest wajib tetap
tersimpan per run agar analisis identitas kelas eksploratif dapat dilakukan kemudian
tanpa run ulang.

**Update (G1-10, sesi 2026-09-27 — baseline klasik dan kebijakan loss adjuster R):**
Ditambahkan baseline klasik referensi: chance `1/K`, Nearest Centroid (NC), dan
Logistic Regression multinomial (LR, `C` dipilih di validation). Baseline dijalankan
oleh program **terpisah** `main_classical_baseline.py` (tidak digabung ke `main_train.py`),
tetapi **tidak** punya preprocessing sendiri: ia membaca `X_*_scaled.npy`/`y_*.npy` yang
sudah disimpan `DATA_PIPELINE` §2.7 pada run primer MORE-HD dengan `(seed, feature_method, K)`
yang sama, sehingga input baseline identik dengan input sirkuit. Tidak ada perubahan pada
`CONFIG`, `DATA_PIPELINE`, `MAIN(config)`, identitas R001–R048, atau jumlah run kuantum.
Loss adjuster R milik MORE **tidak** diimplementasikan (`loss_adjuster_policy="NONE"`);
acuan literatur memakai kolom MORE\R Tabel I Wu et al. (2023). Spesifikasi lengkap di
**Bagian 12** dan `MORE_HD_STATISTICAL_ANALYSIS_PLAN.md` §10.2.

**Update (crash-safe append-only log):** Parameter pada setiap **objective-function
evaluation** (baik di `CLUSTERING_LOOP` maupun `SUPERVISED_LOOP`) ditulis ke
**satu file binary append-only** (`clustering_params.bin`,
`supervised_params.bin`) dengan ukuran record tetap. Posisi record sama dengan
`eval_id`, sehingga evaluasi tertentu dapat diakses dengan
`READ_PARAM_RECORD(log, eval_id, record_size)` tanpa membaca ulang seluruh file.
Format `.npz` tetap tidak dipakai karena risiko *zip central directory corruption*
jika proses berhenti di tengah penulisan.

Metrik objective memakai **JSON Lines (`.jsonl`)** — satu baris JSON per
`eval_id`. Progress callback optimizer disimpan pada file terpisah
(`clustering_callback_log.jsonl` / `supervised_callback_log.jsonl`) dengan
`callback_id`; callback tidak boleh memanggil objective ulang. Dengan demikian,
`eval_id`, `callback_id`, dan `result.nfev` tidak pernah diperlakukan sebagai
istilah yang sama.

**Restart dari crash:** dilakukan manual di folder run baru yang kosong.
`TRUNCATE_INCOMPLETE_TAIL()` tidak diperlukan. Tabrakan nama folder saat
restart adalah tanggung jawab pengguna, karena `AUTO_GENERATE()` bersifat
deterministik (tidak memakai timestamp).
 
---
 
## 0.1 DESAIN EKSPERIMEN FAKTORIAL 2 × 3 DAN SKENARIO 3–10 KELAS

Eksperimen utama menggunakan dua faktor:

- Faktor A — `architecture`: `"MORE-HD"` (RY saja) dan `"MORE-HD-C"` (RY+RZ).
- Faktor B — `feature_method`: `"PCA"`, `"HU"`, dan `"ZERNIKE"`.

Setiap kombinasi faktor dijalankan pada delapan skenario jumlah kelas, mengikuti pola kumulatif kelas MNIST:

```
K=3  -> classes=[0,1,2]
K=4  -> classes=[0,1,2,3]
K=5  -> classes=[0,1,2,3,4]
K=6  -> classes=[0,1,2,3,4,5]
K=7  -> classes=[0,1,2,3,4,5,6]
K=8  -> classes=[0,1,2,3,4,5,6,7]
K=9  -> classes=[0,1,2,3,4,5,6,7,8]
K=10 -> classes=[0,1,2,3,4,5,6,7,8,9]
```

**Lingkup urutan kelas (G1-09, dikunci 2026-09-27):** urutan kumulatif di atas
dipertahankan tanpa subset kelas acak. Setiap transisi K→K+1 menambahkan satu digit
tertentu, sehingga efek jumlah kelas tidak terpisah dari identitas digit yang
ditambahkan. Perbandingan arsitektur dan feature method pada K yang sama tidak
terkena confound ini; tren lintas-K dilaporkan sebagai tren pada urutan kumulatif
`[0..K-1]`, bukan pada himpunan kelas sembarang berukuran K.

Total kondisi eksperimen per seed:

```
8 skenario kelas × 2 architecture × 3 feature_method = 48 kondisi

```

Replikasi konfirmatori:

```
48 kondisi × 5 confirmatory seeds = 240 run
confirmatory_seeds = [101, 202, 303, 404, 505]
pilot_seed = 42  # tidak masuk agregasi final
```

Targeted ablation G1-05/G1-07:

```
K = {3,6,10}
feature_method = {PCA, HU, ZERNIKE}
new ablation models = {MORE-HD-60P, MORE-HD-C-FixedRZ}
18 kondisi tambahan per seed × 5 confirmatory seeds = 90 run baru
A=MORE-HD dan D=MORE-HD-C direuse dari primary matrix, tidak dirun ulang
TOTAL = 240 primary + 90 ablation = 330 unique confirmatory executions
```

Ablation bukan level baru pada `GENERATE_RUN_ID()` primer. Ia memakai ID terpisah
`ABL001-ABL018` dan entry point `main_ablation.py`; definisi lengkapnya berada di **Bagian 11**.

Prinsip isolasi variabel:

1. `n_data_qubits` dikunci = 8 untuk SEMUA metode representasi.
2. PCA menghasilkan 8 fitur.
3. Hu menghasilkan 7 Hu Moments asli lalu ditambah 1 kanal netral bernilai 0, sehingga interface ke sirkuit tetap 8 dimensi.
4. Zernike menghasilkan tepat 8 koefisien yang dipilih dengan aturan deterministik dan dikunci sebelum eksperimen.
5. Arsitektur sirkuit, topologi entanglement, jumlah readout qubit, urutan 15 observable, optimizer, dan prosedur evaluasi tidak berubah hanya karena `feature_method` berbeda.
6. Matriks korelasi `S` dihitung dari representasi fitur masing-masing run. Jadi faktor `feature_method` menguji efek representasi fitur terhadap keseluruhan pipeline clustering + classification.

Satu pemanggilan `MAIN(config)` tetap hanya menghasilkan satu run. Seluruh 48 kondisi eksperimen dijalankan **secara manual satu per satu**, masing-masing dengan `run_dir` terpisah, agar proses clustering, supervised training, log, parameter, dan hasil akhir dapat dimonitor secara individual. Tidak digunakan batch runner otomatis untuk menjalankan seluruh matriks sekaligus.

---

## 0.2 STRUKTUR FOLDER PROYEK, DATASET LOKAL, DAN SPREADSHEET PER-RUN

Seluruh source code, dataset lokal, artefak per-run, dan data penelitian konsolidasi ditempatkan dalam satu folder proyek dengan pemisahan fungsi yang jelas. Struktur folder yang dipakai adalah:

```text
project/
│
├── main_train.py                  # Jalur A: satu run training lengkap
├── main_selected_clustering.py    # Jalur B: secondary path dari checkpoint clustering terpilih deterministik
├── main_ablation.py               # targeted ablation G1-05/G1-07; hanya model B/C
├── main_classical_baseline.py     # G1-10: chance/NC/LR; membaca array dari runs/, tanpa preprocessing ulang
│
├── core/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── correlation.py
│   ├── circuits.py
│   ├── clustering.py
│   ├── quantum_labels.py
│   ├── supervised.py
│   ├── evaluation.py
│   ├── classical_baselines.py     # G1-10: NC, LR, chance (dipakai hanya oleh main_classical_baseline.py)
│   └── artifact_io.py
│
├── data/
│   └── MNIST/                     # dataset diunduh sekali saat pilot, lalu dibaca lokal
│
├── splits/
│   ├── seed101.json
│   ├── seed202.json
│   ├── seed303.json
│   ├── seed404.json
│   └── seed505.json
│
├── research_data/
│   ├── MORE_HD_master_confirmatory_240runs.xlsx   # primary A/D
│   └── MORE_HD_master_ablation_90runs.xlsx        # targeted B/C; dibuat/diuji sebelum Gate C
│
└── runs/
    ├── cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed42/
    │   ├── artifacts/
    │   ├── logs/
    │   ├── config.json
    │   └── run_result.xlsx        # salinan lokal; hanya ditulis oleh run ini
    │
    ├── cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD-C_seed42/
    │   ├── artifacts/
    │   ├── logs/
    │   ├── config.json
    │   └── run_result.xlsx
    │
    ├── ...
    │
    ├── cls-0-1-2-3-4-5-6-7-8-9_ntrain1000_nval100_ntest200_ZERNIKE_MORE-HD-C_seed42/
    │   ├── artifacts/
    │   ├── logs/
    │   ├── config.json
    │   └── run_result.xlsx
    │
    ├── baselines/                 # G1-10; ditulis HANYA oleh main_classical_baseline.py
    │   ├── baseline_config.json
    │   ├── baseline_results.csv   # 1 baris per (seed, feature_method, K, baseline)
    │   └── S101_cls-0-1-2_PCA/
    │       ├── baseline_cell_manifest.json
    │       ├── NC_confusion_matrix.npy
    │       ├── LR_confusion_matrix.npy
    │       ├── NC_test_predictions.npy
    │       ├── LR_test_predictions.npy
    │       └── LR_cv_log.json     # akurasi validation per nilai C
    │
    └── ablation/
        ├── ABL001-S101_cls-0-1-2_PCA_MORE-HD-60P/
        │   ├── artifacts/
        │   ├── logs/
        │   ├── config.json
        │   └── run_result.xlsx
        └── ABL002-S101_cls-0-1-2_PCA_MORE-HD-C-FixedRZ/
            ├── artifacts/
            ├── logs/
            ├── config.json
            └── run_result.xlsx
```

Prinsip penyimpanan untuk eksekusi paralel:

1. `runs/` tetap menjadi sumber utama artefak mentah. Setiap kombinasi `classes × feature_method × architecture × seed` mempunyai `run_dir` sendiri dan **satu proses hanya boleh menulis ke satu `run_dir` miliknya**.
2. File `research_data/MORE_HD_master_confirmatory_240runs.xlsx` **tidak pernah ditulis oleh proses training**. Selama 240 run konfirmatori berlangsung, file ini hanya dipakai sebagai sumber/template untuk membuat salinan lokal per-run.
3. Saat sebuah run baru berhasil membuat `run_dir`, program menyalin file master tersebut menjadi `runs/<run_name>/run_result.xlsx`. Setelah itu, seluruh penulisan spreadsheet oleh run tersebut hanya diarahkan ke salinan lokal `run_result.xlsx`.
4. Karena setiap proses paralel menulis file Excel yang berbeda, tidak ada dua proses yang melakukan concurrent write ke workbook yang sama.
5. Setelah seluruh 240 run konfirmatori selesai, isi hasil dari masing-masing `run_result.xlsx` dapat dikonsolidasikan kembali ke `MORE_HD_master_confirmatory_240runs.xlsx`. Tahap konsolidasi akhir berada **di luar proses training paralel** dan belum diotomatisasi pada pseudocode ini.
6. Artefak JSON/JSONL/NPY/BIN tetap menjadi sumber data paling dasar. `run_result.xlsx` adalah representasi tabel lokal dari satu run dan tidak menggantikan artefak mentah.
7. Pengelolaan paralel Jalur B belum diubah pada revisi ini; collision guard tambahan khusus Jalur B ditunda sesuai keputusan eksperimen saat ini.

Path spreadsheet dan dataset dikunci pada level proyek:

```
MASTER_SPREADSHEET_PATH     = "research_data/MORE_HD_master_confirmatory_240runs.xlsx"
LOCAL_RUN_SPREADSHEET_NAME  = "run_result.xlsx"
MNIST_ROOT                  = "data/"
```

Master spreadsheet konfirmatori memuat **48 condition ID per seed**, bukan 48 run total. Lima seed konfirmatori yang sudah dibekukan (`101, 202, 303, 404, 505`) menghasilkan:

```
8 skenario kelas × 3 feature_method × 2 architecture = 48 kondisi per seed
48 kondisi × 5 confirmatory seeds = 240 confirmatory runs
```

`R001` sampai `R048` tetap dipakai sebagai **condition_id**. Eksekusi unik memakai `run_uid = condition_id-S<seed>` (contoh `R001-S101`). Seed `42` hanya untuk `PILOT`/smoke test dan tidak masuk agregasi konfirmatori. Urutan `condition_id` tetap mengikuti `K=3` sampai `K=10`; di dalam setiap skenario kelas urutannya adalah `PCA`, `HU`, `ZERNIKE`, dan pada setiap metode fitur urutannya `MORE-HD` kemudian `MORE-HD-C`.

```
FUNCTION GENERATE_RUN_ID(classes, feature_method, architecture):
    K = LENGTH(classes)

    ASSERT classes == [0, 1, 2, ..., K-1]
    ASSERT 3 <= K <= 10

    class_block = (K - 3) * 6

    feature_offset = {
        "PCA": 0,
        "HU": 2,
        "ZERNIKE": 4
    }[feature_method]

    architecture_offset = {
        "MORE-HD": 0,
        "MORE-HD-C": 1
    }[architecture]

    run_number = class_block + feature_offset + architecture_offset + 1
    RETURN "R" + ZERO_PAD(run_number, 3)
```

---


## 0.3 PROTOKOL MULTI-SEED, SPLIT MANIFEST, DAN PROPAGASI RNG

Konstanta protokol yang dibekukan:

```
PILOT_SEED = 42
CONFIRMATORY_SEEDS = [101, 202, 303, 404, 505]
N_CONFIRMATORY_REPLICATIONS = 5
RUN_MODE = "PILOT" | "CONFIRMATORY"
```

Aturan validasi:

```
FUNCTION VALIDATE_SEED_PROTOCOL(config):
    IF config.run_mode == "PILOT":
        ASSERT config.seed == PILOT_SEED
    ELSE IF config.run_mode == "CONFIRMATORY":
        ASSERT config.seed IN CONFIRMATORY_SEEDS
    ELSE:
        ERROR("run_mode tidak dikenal")
```

Satu `master_seed` tidak dipakai langsung untuk seluruh sumber randomness.
Sub-seed diturunkan secara deterministik berdasarkan namespace agar setiap
sumber randomness dapat direproduksi dan diaudit secara terpisah:

```
FUNCTION DERIVE_SUBSEED(master_seed, namespace):
    digest = SHA256(STRING(master_seed) + ":" + namespace)
    RETURN UINT32(FIRST_8_HEX_DIGITS(digest))

data_seed      = DERIVE_SUBSEED(seed, "data")
ry_core_seed   = DERIVE_SUBSEED(seed, "init:ry_core")
rz_phase_seed  = DERIVE_SUBSEED(seed, "init:rz_phase")
ry_extra_seed  = DERIVE_SUBSEED(seed, "ablation:init:ry_extra")  # hanya B
pair_seed      = DERIVE_SUBSEED(seed, "cluster_pairs")
label_seed     = DERIVE_SUBSEED(seed, "quantum_labels")
```

Manifest split dibuat satu kali per confirmatory seed dan tidak dibuat ulang
per feature method atau per architecture:

```
FUNCTION CREATE_OR_LOAD_SPLIT_MANIFEST(master_seed):
    ASSERT master_seed IN CONFIRMATORY_SEEDS
    path = "splits/seed" + master_seed + ".json"

    IF EXISTS(path):
        manifest = LOAD_JSON(path)
        VALIDATE_SPLIT_MANIFEST(manifest, master_seed)
        RETURN manifest

    FOR digit IN 0..9:
        train_pool = ALL_INDICES_OF_DIGIT(mnist_train_raw, digit)
        test_pool  = ALL_INDICES_OF_DIGIT(mnist_test_raw, digit)

        shuffled_train = SHUFFLE(train_pool, RNG(DERIVE_SUBSEED(master_seed, "data:train:" + digit)))
        shuffled_test  = SHUFFLE(test_pool,  RNG(DERIVE_SUBSEED(master_seed, "data:test:" + digit)))

        train_idx[digit] = FIRST 1000 OF shuffled_train
        val_idx[digit]   = NEXT 100 OF shuffled_train
        test_idx[digit]  = FIRST 200 OF shuffled_test

        ASSERT DISJOINT(train_idx[digit], val_idx[digit])

    manifest = {
        "master_seed": master_seed,
        "train_idx_by_class": train_idx,
        "val_idx_by_class": val_idx,
        "test_idx_by_class": test_idx,
        "nested_k_rule": "reuse same class-specific indices for every K containing that class"
    }

    SAVE_ATOMIC_JSON(manifest, path)
    RETURN manifest
```

Untuk skenario K tertentu, `DATA_PIPELINE` hanya mengambil indeks kelas
`0..K-1` dari manifest tersebut. Karena indeks per kelas tidak berubah,
K=4 mempertahankan seluruh sampel kelas 0,1,2 dari K=3 dan hanya menambahkan
kelas 3; pola yang sama berlaku sampai K=10.

```
FUNCTION BUILD_RUN_IDENTITY(config):
    condition_id = GENERATE_CONDITION_ID(config.K, config.feature_method, config.architecture)
    run_uid = condition_id + "-S" + STRING(config.seed)
    RETURN condition_id, run_uid
```

Untuk `PILOT`, seed 42 boleh menggunakan split sementara/pilot, tetapi artefak
pilot harus ditandai `run_mode="PILOT"` dan tidak boleh dimasukkan ke workbook
agregasi konfirmatori maupun klaim publikasi.

---

## 0.4 TARGETED ABLATION G1-05/G1-07 — B + C PADA K={3,6,10}

Track ini **tidak mengubah** `GENERATE_RUN_ID()` atau 48 kondisi primer.
A dan D dibaca dari primary run yang sudah ada; hanya B dan C dieksekusi sebagai run baru.

| Code | Model | State family | Trainable | Fixed | Scope |
|---|---|---|---:|---:|---|
| A | MORE-HD | real | 30 RY | 0 | reuse primary |
| B | MORE-HD-60P | real | 60 RY | 0 | new ablation run |
| C | MORE-HD-C-FixedRZ | complex allowed | 30 RY | 30 non-zero RZ | new ablation run |
| D | MORE-HD-C | complex allowed | 30 RY + 30 RZ | 0 | reuse primary |

Aturan structured paired initialization:

```
ry_core_seed  = DERIVE_SUBSEED(seed, "init:ry_core")
rz_phase_seed = DERIVE_SUBSEED(seed, "init:rz_phase")
ry_extra_seed = DERIVE_SUBSEED(seed, "ablation:init:ry_extra")

A/C/D: RY core awal sama (30 nilai)
D:     RY core + RZ phase di-PACK per layer sesuai ordering RY+RZ
B:     30 RY pertama = RY core yang sama; 30 RY tambahan dari ry_extra_seed
C/D:   initial RZ vector sama dan seluruh elemennya non-zero
C:     RZ dibekukan selama clustering + supervised
D:     RZ dilatih
```

Planned paired contrasts adalah A-B, A-C, B-D, dan C-D. B-vs-D hanya disebut
**equal-trainable-parameter control**, bukan perfect architecture/depth match, karena
B mempunyai 6 variational layers sehingga jumlah entangling blocks/CNOT juga lebih besar.
Gate count, depth, runtime, fixed-RZ hash, split hash, pair-manifest hash, dan Y-odd
activity wajib disimpan.

Builder B/C, ID `ABL001-ABL018`, manifest, artefak, dan analysis linkage didefinisikan penuh di **Bagian 11**.

## 1. CONFIG

```
STRUCT Config:
    classes                = [0, 1, 2]        # dipilih user; benchmark utama memakai 0..K-1 untuk K=3..10
    n_train_per_class      = 1000              # DIKUNCI (G0-03, sesi 2026-09-22): protokol final, mengikuti skala FRD-09
    n_val_per_class         = 100               # DIKUNCI (G0-03, sesi 2026-09-22): angka INDEPENDEN, bukan lagi mengikuti
                                                # n_test_per_class (revisi atas aturan proporsi G0-01 sebelumnya).
                                                # Validation diambil dari POOL TRAINING (bukan pool test), disjoint dari
                                                # n_train_per_class, dan HANYA dipakai untuk monitoring pasif /
                                                # checkpoint selection selama CLUSTERING_LOOP dan SUPERVISED_LOOP,
                                                # serta untuk pemilihan checkpoint deterministik di Jalur B.
    n_test_per_class       = 200               # DIKUNCI (G0-03, sesi 2026-09-22): official test -- TIDAK diakses
                                                # sebelum FINAL_EVALUATION (G0-01)
    max_nfev_clustering      = 10               # SMOKE TEST = 10; PILOT G1-01 = 300 (lihat §1.1); final dikunci dari pilot
    max_nfev_supervised      = 10               # jumlah maksimum objective-function evaluations, BUKAN iterasi
                                                # budget SAMA untuk semua arsitektur (aturan total-nfev G1-06)

    architecture           = "MORE-HD"        # "MORE-HD" | "MORE-HD-C"
    feature_method         = "PCA"            # "PCA" | "HU" | "ZERNIKE"

    n_data_qubits          = 8                # DIKUNCI sama untuk semua feature_method
    n_readout_qubits       = 2
    n_input_channels       = 8                # interface akhir DATA_PIPELINE -> quantum encoding

    # konfigurasi PCA
    pca_n_components       = 8
    pca_random_state       = 42               # = config.seed; menjaga determinisme FIT_PCA
    pca_svd_solver         = "auto"           # default sklearn; deterministik selama random_state dikunci

    # konfigurasi Hu
    hu_raw_dim             = 7                # sifat intrinsik Hu Moments
    hu_input_mode          = "grayscale"      # DIKUNCI: grayscale ternormalisasi, BUKAN biner/Otsu
    hu_padding_value        = 0.0              # kanal ke-8 netral -> RY(0); DIKUNCI final (menang atas pi/2 di noa3.ipynb)
    hu_use_signed_log       = TRUE             # stabilisasi dynamic range sebelum scaler
    hu_signed_log_epsilon   = 1e-30            # DIKUNCI: h' = -sign(h) * log10(|h| + epsilon)
    hu_clip_after_scaling   = FALSE            # DIKUNCI: tidak clip, konsisten dengan PCA

    # konfigurasi Zernike -- 8 term tetap, tidak dipilih berdasarkan hasil test
    zernike_terms          = [(2,0), (2,2), (3,1), (3,3),
                              (4,0), (4,2), (5,1), (5,5)]   # DIREVISI: (0,0) & (1,1) dibuang (kurang diskriminatif)
    zernike_use_magnitude  = TRUE             # |Z_nm| untuk invariansi rotasi
    zernike_clip_after_scaling = FALSE        # DIKUNCI: tidak clip, konsisten dengan PCA dan Hu

    cobyla_tol             = 1e-4
    cobyla_rhobeg          = 1.0              # DIKUNCI untuk pilot (G1-01/G1-06): default SciPy, ditulis eksplisit
    seed                   = 42
    run_mode                = "PILOT"      # "PILOT" | "CONFIRMATORY"
    confirmatory_seeds       = [101,202,303,404,505]
    n_cluster_pair_samples = 5                # DIKUNCI G1-03/G1-08: mengikuti MORE asli (5 instance/kelas)
    pair_balance_policy    = "NATURAL_FULL_PAIRING"  # semua unordered unique pairs dipakai
    pair_weighting         = "NONE"            # tidak ada balancing/reweighting same-vs-different
    active_dim_threshold   = 1e-6

    # dataset lokal / pilot download
    mnist_root              = "data/"
    mnist_download          = FALSE             # TRUE hanya saat pilot/initial download

    run_id = GENERATE_RUN_ID(classes, feature_method, architecture)

    run_name = AUTO_GENERATE(
        classes, n_train_per_class, n_val_per_class, n_test_per_class,
        architecture, feature_method, seed
    )
    run_dir = "runs/" + run_name
    local_spreadsheet_path = run_dir + "/" + LOCAL_RUN_SPREADSHEET_NAME


FUNCTION AUTO_GENERATE(classes, n_train, n_val, n_test, architecture, feature_method, seed):
    # contoh (dengan protokol final G0-03: n_train=1000, n_val=100, n_test=200):
    # cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed42
    # cls-0-1-2_ntrain1000_nval100_ntest200_HU_MORE-HD-C_seed42
    RETURN "cls-" + JOIN(classes, "-") +
           "_ntrain" + n_train +
           "_nval" + n_val +
           "_ntest" + n_test +
           "_" + feature_method +
           "_" + architecture +
           "_seed" + seed


FUNCTION VALIDATE_CONFIG(config):
    IF config.architecture NOT IN ["MORE-HD", "MORE-HD-C"]:
        RAISE_ERROR("architecture tidak dikenali")

    IF config.feature_method NOT IN ["PCA", "HU", "ZERNIKE"]:
        RAISE_ERROR("feature_method harus PCA, HU, atau ZERNIKE")

    IF config.n_data_qubits != 8 OR config.n_input_channels != 8:
        RAISE_ERROR("eksperimen utama mengunci interface input pada 8 channel / 8 data qubit")

    IF config.n_val_per_class <= 0:
        RAISE_ERROR("n_val_per_class harus lebih besar dari 0 -- validation wajib ada untuk G0-01")

    IF config.max_nfev_clustering <= 0 OR config.max_nfev_supervised <= 0:
        RAISE_ERROR("max_nfev_clustering dan max_nfev_supervised harus > 0")

    IF LENGTH(config.classes) < 3 OR LENGTH(config.classes) > 10:
        PRINT_WARNING("benchmark utama dirancang untuk 3 sampai 10 kelas")


FUNCTION CREATE_RUN_DIRECTORY_EXCLUSIVE(config):
    # WAJIB dilakukan sebelum training dan sebelum menulis artefak apa pun.
    # Operasi pembuatan direktori harus atomic/exclusive (setara mkdir exist_ok=False).
    # Dua proses dengan run_name identik tidak boleh pernah menulis ke folder yang sama.

    TRY:
        CREATE_DIRECTORY_EXCLUSIVE(config.run_dir)
    CATCH DIRECTORY_ALREADY_EXISTS:
        RAISE_ERROR(
            "Run directory sudah ada. Training dibatalkan untuk mencegah " +
            "artefak/log run lama tertimpa: " + config.run_dir
        )

    CREATE_DIRECTORY(config.run_dir + "/artifacts")
    CREATE_DIRECTORY(config.run_dir + "/logs")


FUNCTION INITIALIZE_LOCAL_RUN_SPREADSHEET(config):
    # Master TIDAK dibuka untuk write oleh training.
    # Setiap run memperoleh file Excel lokal yang independen.
    IF NOT FILE_EXISTS(MASTER_SPREADSHEET_PATH):
        RAISE_ERROR("Master/template spreadsheet tidak ditemukan: " + MASTER_SPREADSHEET_PATH)

    COPY_FILE(
        source      = MASTER_SPREADSHEET_PATH,
        destination = config.local_spreadsheet_path
    )

    RETURN config.local_spreadsheet_path


FUNCTION UPDATE_LOCAL_RUN_SPREADSHEET(local_spreadsheet_path, run_id, run_dir):
    # Dipanggil setelah artefak akhir run tersimpan.
    # Fungsi ini TIDAK PERNAH membuka MASTER_SPREADSHEET_PATH dalam mode write.

    ASSERT local_spreadsheet_path STARTS_WITH run_dir
    ASSERT FILE_EXISTS(local_spreadsheet_path)

    manifest                    = LOAD_JSON(run_dir + "/config.json")
    metrics_final               = LOAD_JSON(run_dir + "/logs/metrics_final.json")
    clustering_log              = READ_JSONL(run_dir + "/logs/clustering_log.jsonl")
    supervised_log              = READ_JSONL(run_dir + "/logs/supervised_log.jsonl")
    clustering_callback_log     = READ_JSONL(run_dir + "/logs/clustering_callback_log.jsonl")
    supervised_callback_log     = READ_JSONL(run_dir + "/logs/supervised_callback_log.jsonl")
    clustering_optimizer_result = LOAD_JSON(run_dir + "/logs/clustering_optimizer_result.json")
    supervised_optimizer_result = LOAD_JSON(run_dir + "/logs/supervised_optimizer_result.json")
    quantum_labels              = LOAD_JSON(run_dir + "/artifacts/quantum_labels.json")
    correlation_mat             = LOAD(run_dir + "/artifacts/correlation_matrix.npy")
    confusion_matrix            = LOAD(run_dir + "/logs/confusion_matrix.npy")

    workbook = OPEN_WORKBOOK(local_spreadsheet_path, mode="write_local_only")

    # G1-02: terminology spreadsheet harus membedakan objective evaluation vs callback.
    NORMALIZE_OPTIMIZATION_HEADERS(
        workbook,
        objective_evaluation_header = "Eval ID",
        callback_header             = "Callback ID",
        nfev_header                 = "nfev",
        forbid_legacy_objective_header = "Iteration"
    )

    # Hanya bagian/row milik run_id ini yang diisi; run lain tidak disentuh.
    WRITE_RUN_DATA_TO_EXISTING_TEMPLATE(
        workbook=workbook,
        run_id=run_id,
        manifest=manifest,
        metrics_final=metrics_final,
        clustering_log=clustering_log,
        supervised_log=supervised_log,
        clustering_callback_log=clustering_callback_log,
        supervised_callback_log=supervised_callback_log,
        clustering_optimizer_result=clustering_optimizer_result,
        supervised_optimizer_result=supervised_optimizer_result,
        quantum_labels=quantum_labels,
        correlation_matrix=correlation_mat,
        confusion_matrix=confusion_matrix
    )

    SAVE_WORKBOOK(workbook, local_spreadsheet_path)
    CLOSE_WORKBOOK(workbook)


```

**Catatan determinisme:** `SET_RANDOM_SEED(config.seed)` dipanggil ulang
secara eksplisit di awal `DATA_PIPELINE`, `MODEL_SETUP`, dan
`CLUSTERING_LOOP` (bukan hanya sekali secara global), supaya hasil tetap
deterministik terlepas dari urutan pemanggilan fungsi.

### 1.1 Protokol smoke test, pilot budget, dan fase simplex COBYLA (G1-01/G1-06, dikunci 2026-09-27)

**Latar belakang.** COBYLA selalu memakai `n_params + 1` objective evaluation
pertama untuk membangun simplex awal (`x0` ditambah satu probe per koordinat
sejauh `rhobeg`). `n_params` di sini adalah panjang vektor yang dioptimasi
COBYLA (trainable saja): A=MORE-HD 30 → simplex 31; B=MORE-HD-60P 60 → 61;
C=MORE-HD-C-FixedRZ 30 trainable (RZ beku tidak dihitung) → 31;
D=MORE-HD-C 60 → 61.

**Tahap 0 — Smoke test** (`run_mode="PILOT"`, seed 42)

| Setting | Nilai |
|---|---|
| `max_nfev_clustering` / `max_nfev_supervised` | 10 / 10 |
| Kondisi | K=3, PCA, MORE-HD dan MORE-HD-C (2 run) |
| Tujuan | Pipeline berjalan end-to-end; artefak/log tertulis; `LENGTH(log) == result.nfev`; kolom `phase` ada |
| Batasan | Seluruh 10 evaluasi berada di fase `initial_simplex`; label `optimization` **belum** teruji; hasil tidak dianalisis |

**Tahap 1 — Pilot konvergensi** (`run_mode="PILOT"`, seed 42)

| Setting | Nilai |
|---|---|
| `max_nfev_clustering` / `max_nfev_supervised` | 300 / 300 (cap pilot, sama untuk semua arsitektur) |
| `cobyla_tol` | `1e-4` |
| `cobyla_rhobeg` | `1.0` |
| Kondisi | K ∈ {3, 10} × {PCA, ZERNIKE} × {MORE-HD, MORE-HD-C} = 8 run; opsional MORE-HD-60P pada K=10 |
| Urutan eksekusi | Run pertama K=3/PCA/MORE-HD-C dijalankan sendiri dan diperiksa (`phase` = `initial_simplex` untuk `eval_id` 0–60 dan `optimization` mulai `eval_id` 61; `LENGTH(log) == result.nfev`; summary optimizer lengkap; waktu per evaluasi tercatat) sebelum 7 run sisanya dijalankan |
| Dicatat per run | best-observed `train_loss` vs `eval_id` + `phase`; `pseudo_accuracy_val`/`val_loss`; waktu per objective evaluation; `success`/`status`/`message` (berhenti karena `tol` atau budget) |

**Aturan keputusan budget final** (dikunci sebelum pilot dijalankan):

1. Untuk setiap run pilot dan setiap loop (clustering, supervised), hitung kurva
   best-observed `train_loss` pada fase `optimization`.
2. `N*` per run = `eval_id` terkecil (dihitung sebagai total `nfev`) di mana
   best-observed loss sudah berada dalam **2%** dari total perbaikan yang dicapai
   pada akhir run (`L(N) - L_end <= 0.02 × (L_x0 - L_end)`).
3. `max_nfev` final per loop = maksimum `N*` dari seluruh 8 run pilot, dibulatkan ke
   atas ke kelipatan 10, dengan batas atas 300.
4. Sebuah run dinyatakan **belum plateau** jika berhenti karena budget (bukan `tol`)
   **dan** perbaikan best-observed loss pada 30 evaluasi terakhir melebihi 2% dari
   total perbaikan run (`L(nfev-30) - L_end > 0.02 × (L_x0 - L_end)`). Jika ada run
   yang belum plateau pada 300, `max_nfev` final tetap
   300 atas dasar biaya komputasi; kondisi tersebut dilaporkan sebagai limitation
   dan klaim publikasi dirumuskan sebagai "pada budget objective evaluation yang
   sama", bukan "pada konvergensi".
5. Untuk run konfirmatori, `max_nfev` final wajib `> n_params + 1` untuk semua
   model (A/B/C/D); `VALIDATE_OPTIMIZER_BUDGET` menolak konfigurasi yang
   melanggar.

**Aturan anggaran adil (G1-06).** Primary: **total `nfev` sama** untuk semua
arsitektur. Karena simplex D (61) lebih panjang dari A (31), D mendapat 30
evaluasi fase `optimization` lebih sedikit; ini konservatif terhadap MORE-HD-C
dan dilaporkan eksplisit melalui `n_optimization_evals` di summary optimizer.
Kontras ablation B–D dan A–C memiliki panjang simplex identik sehingga bebas dari
confound ini.

```
FUNCTION OBJECTIVE_PHASE(eval_id, n_params):
    # eval_id 0-based; eval_id 0 = x0, eval_id 1..n_params = probe simplex.
    IF eval_id <= n_params:
        RETURN "initial_simplex"
    RETURN "optimization"


FUNCTION VALIDATE_OPTIMIZER_BUDGET(max_nfev, n_params, run_mode):
    simplex_size = n_params + 1
    IF max_nfev <= simplex_size:
        IF run_mode == "CONFIRMATORY":
            RAISE_ERROR("max_nfev harus > n_params + 1 untuk run konfirmatori (G1-06)")
        PRINT_WARNING("budget tidak keluar dari fase initial_simplex; hanya valid sebagai smoke test")
```

Unit test wajib: pada fixture COBYLA nyata, untuk `eval_id` 1..`n_params`, titik
yang dievaluasi berbeda dari vertex terbaik sebelumnya pada tepat satu koordinat
sebesar `±rhobeg`. Test ini memverifikasi label `phase` terhadap versi SciPy yang
dipakai (versi SciPy dicatat di summary optimizer).
 
---
 
## 2. DATA PIPELINE

Semua metode representasi harus berakhir pada matriks fitur berukuran `(N, 8)` supaya sirkuit kuantum tetap identik lintas metode. Perbedaan hanya terjadi pada cara membentuk representasi fitur sebelum scaling ke `[0, PI]`. Sejak revisi G0-01 (sesi 2026-09-22), fungsi ini menghasilkan **tiga** split -- train, validation, dan official test -- bukan dua.

```
FUNCTION DATA_PIPELINE(config):
    SET_RANDOM_SEED(config.seed)
    VALIDATE_CONFIG(config)

    # 2.1 Load MNIST mentah dalam bentuk citra 28x28
    # Pilot pertama: config.mnist_download = TRUE untuk menyiapkan data di project/data.
    # Parallel run berikutnya: config.mnist_download = FALSE; program hanya membaca data lokal.
    IF config.mnist_download == TRUE:
        mnist_train_raw, mnist_test_raw = LOAD_MNIST_FROM_TORCHVISION(
            root=config.mnist_root,
            download=TRUE
        )
    ELSE:
        IF NOT MNIST_DATASET_EXISTS(config.mnist_root):
            RAISE_ERROR(
                "MNIST lokal belum tersedia di " + config.mnist_root +
                ". Jalankan satu pilot run dengan mnist_download=TRUE terlebih dahulu."
            )

        mnist_train_raw, mnist_test_raw = LOAD_MNIST_FROM_TORCHVISION(
            root=config.mnist_root,
            download=FALSE
        )

    # 2.2 Filter kelas dan sampling -- tiga split: train, validation, official test (G0-01, sesi 2026-09-22)
    # Train dan validation SAMA-SAMA berasal dari mnist_train_raw; validation WAJIB disjoint
    # dari train (exclude eksplisit setelah sampling train, bukan sampling independen dari pool
    # penuh, supaya tidak ada satu sampel pun yang bocor ke dua split sekaligus).
    # Official test berasal dari mnist_test_raw asli, sumber terpisah total, dan TIDAK diakses
    # oleh CLUSTERING_LOOP, SUPERVISED_LOOP, maupun Jalur B -- hanya oleh FINAL_EVALUATION.
    X_train_img, y_train = FILTER_AND_SAMPLE(
        mnist_train_raw, classes=config.classes, n_per_class=config.n_train_per_class
    )

    mnist_train_pool_sisa = EXCLUDE_SAMPLED(
        mnist_train_raw, already_sampled_indices=INDICES_OF(X_train_img)
    )
    X_val_img, y_val = FILTER_AND_SAMPLE(
        mnist_train_pool_sisa, classes=config.classes, n_per_class=config.n_val_per_class
    )
    ASSERT NO_OVERLAP(INDICES_OF(X_train_img), INDICES_OF(X_val_img))

    X_test_img, y_test = FILTER_AND_SAMPLE(
        mnist_test_raw, classes=config.classes, n_per_class=config.n_test_per_class
    )

    # 2.3 Normalisasi citra ke [0,1] -- common preprocessing
    X_train_norm = NORMALIZE_PIXELS(X_train_img, range=[0,1])
    X_val_norm   = NORMALIZE_PIXELS(X_val_img,   range=[0,1])
    X_test_norm  = NORMALIZE_PIXELS(X_test_img,  range=[0,1])

    # 2.4 Ekstraksi / reduksi fitur menurut metode yang dipilih user
    IF config.feature_method == "PCA":
        # PCA bersifat data-driven: FIT hanya pada train
        X_train_flat = FLATTEN_IMAGES(X_train_norm)      # (N, 784)
        X_val_flat   = FLATTEN_IMAGES(X_val_norm)
        X_test_flat  = FLATTEN_IMAGES(X_test_norm)

        feature_model = FIT_PCA(
            X_train_flat,
            n_components=config.pca_n_components,
            svd_solver=config.pca_svd_solver,
            random_state=config.pca_random_state
        )
        X_train_feat = feature_model.TRANSFORM(X_train_flat)
        X_val_feat   = feature_model.TRANSFORM(X_val_flat)
        X_test_feat  = feature_model.TRANSFORM(X_test_flat)
        # shape akhir sebelum scaler: (N, 8)
        # Catatan: svd_solver="auto" pada sklearn kemungkinan memilih randomized SVD
        # untuk n_components=8 << 784 dimensi; determinisme tetap terjaga karena
        # random_state dikunci = config.seed (mengikuti kode referensi noa3.ipynb).
        # Tidak ada clipping eksplisit yang diterapkan setelah scaling untuk PCA --
        # X_val_scaled dan X_test_scaled boleh sedikit keluar dari [0, PI] akibat
        # proyeksi val/test di luar rentang train, konsisten dengan kode referensi tersebut.

        SAVE(feature_model, config.run_dir + "/artifacts/pca_model.joblib")

    ELSE IF config.feature_method == "HU":
        # Hu Moments tidak di-fit ke distribusi data.
        # Input citra DIKUNCI: grayscale ternormalisasi (X_*_norm langsung),
        # BUKAN dibinerkan/Otsu -- lihat 2.8.2.
        # Dihitung langsung per gambar -> tepat 7 descriptor.
        X_train_hu7 = [HU_MOMENTS(img) FOR img IN X_train_norm]
        X_val_hu7   = [HU_MOMENTS(img) FOR img IN X_val_norm]
        X_test_hu7  = [HU_MOMENTS(img) FOR img IN X_test_norm]

        IF config.hu_use_signed_log == TRUE:
            X_train_hu7 = SIGNED_LOG_TRANSFORM(X_train_hu7, epsilon=config.hu_signed_log_epsilon)
            X_val_hu7   = SIGNED_LOG_TRANSFORM(X_val_hu7,   epsilon=config.hu_signed_log_epsilon)
            X_test_hu7  = SIGNED_LOG_TRANSFORM(X_test_hu7,  epsilon=config.hu_signed_log_epsilon)

        # Tambahkan SATU kanal netral; bukan Hu moment ke-8.
        X_train_feat = APPEND_CONSTANT_COLUMN(
            X_train_hu7, value=config.hu_padding_value
        )
        X_val_feat = APPEND_CONSTANT_COLUMN(
            X_val_hu7, value=config.hu_padding_value
        )
        X_test_feat = APPEND_CONSTANT_COLUMN(
            X_test_hu7, value=config.hu_padding_value
        )
        # shape: (N, 8) = 7 Hu + 1 neutral channel

    ELSE IF config.feature_method == "ZERNIKE":
        # Zernike juga tidak di-fit ke data.
        # Semua gambar dipetakan ke disk satuan dengan prosedur identik.
        X_train_disk = [MAP_IMAGE_TO_UNIT_DISK(img) FOR img IN X_train_norm]
        X_val_disk   = [MAP_IMAGE_TO_UNIT_DISK(img) FOR img IN X_val_norm]
        X_test_disk  = [MAP_IMAGE_TO_UNIT_DISK(img) FOR img IN X_test_norm]

        X_train_feat = [
            EXTRACT_ZERNIKE_TERMS(
                img, terms=config.zernike_terms,
                use_magnitude=config.zernike_use_magnitude
            )
            FOR img IN X_train_disk
        ]
        X_val_feat = [
            EXTRACT_ZERNIKE_TERMS(
                img, terms=config.zernike_terms,
                use_magnitude=config.zernike_use_magnitude
            )
            FOR img IN X_val_disk
        ]
        X_test_feat = [
            EXTRACT_ZERNIKE_TERMS(
                img, terms=config.zernike_terms,
                use_magnitude=config.zernike_use_magnitude
            )
            FOR img IN X_test_disk
        ]
        # shape: (N, 8)

    ELSE:
        RAISE_ERROR("feature_method tidak dikenali")

    ASSERT NUM_COLUMNS(X_train_feat) == config.n_input_channels
    ASSERT NUM_COLUMNS(X_val_feat)   == config.n_input_channels
    ASSERT NUM_COLUMNS(X_test_feat)  == config.n_input_channels

    # 2.5 Scaling akhir ke sudut quantum encoding.
    # FIT scaler HANYA pada TRAIN untuk SEMUA metode -- val dan test sama-sama
    # hanya di-TRANSFORM, tidak pernah ikut fitting (G0-01 / G2-03).
    # Penting: pada HU, kolom padding netral TIDAK diikutkan ke fitting scaler.
    IF config.feature_method == "HU":
        scaler_params = FIT_MINMAX(
            X_train_feat[:, 0:7], range=[0, PI]
        )
        X_train_scaled7 = scaler_params.TRANSFORM(X_train_feat[:, 0:7])
        X_val_scaled7   = scaler_params.TRANSFORM(X_val_feat[:, 0:7])
        X_test_scaled7  = scaler_params.TRANSFORM(X_test_feat[:, 0:7])

        X_train_scaled = APPEND_CONSTANT_COLUMN(X_train_scaled7, value=0.0)
        X_val_scaled   = APPEND_CONSTANT_COLUMN(X_val_scaled7,   value=0.0)
        X_test_scaled  = APPEND_CONSTANT_COLUMN(X_test_scaled7,  value=0.0)
    ELSE:
        scaler_params = FIT_MINMAX(X_train_feat, range=[0, PI])
        X_train_scaled = scaler_params.TRANSFORM(X_train_feat)
        X_val_scaled   = scaler_params.TRANSFORM(X_val_feat)
        X_test_scaled  = scaler_params.TRANSFORM(X_test_feat)

    # 2.5.1 Kebijakan clipping -- DIKUNCI SERAGAM untuk ketiga feature_method:
    # TIDAK ADA clipping eksplisit setelah scaling (PCA, HU, ZERNIKE semua sama).
    # X_val_scaled / X_test_scaled boleh sedikit keluar dari [0, PI] akibat proyeksi/rentang
    # val/test yang berbeda dari train. Ini keputusan sadar supaya feature_method
    # tidak jadi confound tersembunyi lewat perlakuan clipping yang berbeda-beda.

    # 2.6 Simpan artefak preprocessing
    SAVE(scaler_params, config.run_dir + "/artifacts/scaler_params.joblib")

    feature_metadata = {
        "feature_method": config.feature_method,
        "raw_descriptor_dim": (
            8 IF config.feature_method == "PCA" ELSE
            7 IF config.feature_method == "HU" ELSE
            LENGTH(config.zernike_terms)
        ),
        "quantum_input_dim": config.n_input_channels,
        "pca_svd_solver": config.pca_svd_solver IF config.feature_method == "PCA" ELSE NULL,
        "pca_random_state": config.pca_random_state IF config.feature_method == "PCA" ELSE NULL,
        "hu_input_mode": config.hu_input_mode IF config.feature_method == "HU" ELSE NULL,
        "hu_padding_value": config.hu_padding_value IF config.feature_method == "HU" ELSE NULL,
        "hu_signed_log_epsilon": config.hu_signed_log_epsilon IF config.feature_method == "HU" ELSE NULL,
        "n_non_finite_hu": n_non_finite_hu_counter IF config.feature_method == "HU" ELSE NULL,
        "zernike_terms": config.zernike_terms IF config.feature_method == "ZERNIKE" ELSE NULL,
        "clip_after_scaling": FALSE
    }
    SAVE(feature_metadata, config.run_dir + "/artifacts/feature_metadata.json")

    # 2.7 Simpan array final yang benar-benar diberikan ke circuit
    # Array ini juga menjadi SATU-SATUNYA sumber input baseline klasik G1-10 (Bagian 12);
    # main_classical_baseline.py membacanya tanpa menjalankan preprocessing ulang.
    SAVE(X_train_scaled, config.run_dir + "/artifacts/X_train_scaled.npy")
    SAVE(y_train,        config.run_dir + "/artifacts/y_train.npy")
    SAVE(X_val_scaled,   config.run_dir + "/artifacts/X_val_scaled.npy")
    SAVE(y_val,          config.run_dir + "/artifacts/y_val.npy")
    SAVE(X_test_scaled,  config.run_dir + "/artifacts/X_test_scaled.npy")
    SAVE(y_test,         config.run_dir + "/artifacts/y_test.npy")

    RETURN X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
```

Catatan metodologis:

- PCA: model PCA dan scaler hanya di-fit dari data train.
- Hu dan Zernike: extractor bersifat deterministik per gambar dan tidak belajar distribusi train; namun scaler tetap di-fit hanya pada train.
- Kanal ke-8 Hu adalah kanal netral `0.0`, sehingga angle encoding pada qubit ke-8 adalah `RY(0)`. Kanal ini tidak dianggap sebagai Hu Moment baru.
- Kebijakan clipping (2.5.1) sudah diseragamkan: tidak ada clipping untuk PCA, HU, maupun ZERNIKE.
- `feature_method` memengaruhi juga matriks korelasi `S`, karena `CORRELATION_MATRIX` menerima `X_train_scaled` dari representasi yang dipilih.
- **Validation** (`X_val_scaled`, `y_val`) diambil dari pool `mnist_train_raw`, disjoint dari train, BUKAN dipotong dari pool test. Validation dipakai untuk semua monitoring/checkpoint selection selama `CLUSTERING_LOOP`, `SUPERVISED_LOOP`, dan selector deterministik Jalur B (G0-01/G0-02).
- **Official test** (`X_test_scaled`, `y_test`) hanya boleh dipanggil oleh `FINAL_EVALUATION`. `DATA_PIPELINE` tetap men-load dan menyimpan array test di sini (2.7) karena fungsi ini dijalankan sekali per run, tapi tidak ada pemanggilan `circuit_fn` atau perhitungan metrik apa pun terhadap `X_test_scaled`/`y_test` sebelum `FINAL_EVALUATION` dipanggil.

---

## 2.8 RINGKASAN PENGATURAN PER METODE REPRESENTASI FITUR

Bagian ini merinci pengaturan teknis tiap `feature_method` secara terpisah,
supaya setiap keputusan preprocessing bisa direview dan dikunci satu per
satu (sejalan dengan Gate G2 pada `MORE_HD_RESEARCH_READINESS_GATES`).
Status **DIKUNCI** berarti keputusan desain sudah final; untuk Hu dan
Zernike, status ini dicapai pada sesi 2026-09-22 dan masih menunggu
implementasi kode nyata + unit test sebelum Gate G2-01/G2-02 bisa ditutup
(`CLOSED`) — pseudocode fungsi ekstraksi sudah tersedia di 2.8.4.

### 2.8.1 PCA — Status: DIKUNCI

| Aspek | Pengaturan |
|---|---|
| Preprocessing sebelum fit | Flatten citra `(N, 784)`; normalisasi piksel `/255.0` ke `[0,1]`. Tidak ada standardisasi z-score — PCA hanya mean-center otomatis. |
| Jumlah komponen | `pca_n_components = 8` |
| `svd_solver` | `"auto"` (default sklearn; untuk `n_components=8 << 784` kemungkinan besar memilih randomized SVD) |
| `random_state` | `42` (= `config.seed`), menjaga determinisme meski solver randomized |
| Fit | Hanya pada train (`fit_transform` di train, `transform` di val dan test) — tidak ada data leakage |
| Scaler akhir | `MinMaxScaler(feature_range=(0, PI))`, di-fit hanya pada `X_train_pca` |
| Clipping setelah scaling | **Tidak ada.** `X_val_scaled`/`X_test_scaled` boleh sedikit keluar dari `[0, PI]` akibat proyeksi val/test di luar rentang train |
| Sumber keputusan | Disamakan dengan kode referensi `noa3.ipynb` |
| Artefak yang disimpan | `pca_model.joblib`, dicatat di `feature_metadata.json` dan manifest run (`pca_svd_solver`, `pca_random_state`) |

### 2.8.2 HU MOMENTS — Status: DIKUNCI (menunggu implementasi & unit test — lihat Gate G2-01)

| Aspek | Pengaturan final | Keputusan diambil |
|---|---|---|
| Input citra | Grayscale ternormalisasi (`X_train_norm`/`X_val_norm`/`X_test_norm` langsung, piksel `[0,1]`) dipakai sebagai peta massa ke `HU_MOMENTS`. **Bukan** dibinerkan/Otsu. | Sesi 2026-09-22: dipilih grayscale karena mempertahankan info ketebalan stroke digit dan tidak menambah hyperparameter threshold yang belum dikunci. |
| Fungsi ekstraksi `HU_MOMENTS(img)` | Didefinisikan penuh di 2.8.4 — raw image moments -> central moments -> 7 Hu descriptors standar (Hu, 1962). | Pseudocode fungsi ditambahkan 2026-09-22, sebelumnya black-box. |
| Formula signed-log | `h' = -sign(h) x log10(\|h\| + epsilon)`, `epsilon = 1e-30`. Lihat `SIGNED_LOG_TRANSFORM` di 2.8.4. | Formula standar Hu Moments log-stabilization; epsilon sekecil ini hanya berpengaruh saat `h` benar-benar nol eksak. |
| Penanganan non-finite | Jika `h'` NaN/Inf, diganti `0.0` dan dicatat sebagai `n_non_finite_hu` di `feature_metadata.json` (counter warning, bukan error yang menghentikan proses). | Selaras dengan kriteria penerimaan G2-01 ("nilai finite"). |
| Kanal ke-8 (padding) | `hu_padding_value = 0.0` — **final, menggantikan referensi pi/2 di `noa3.ipynb`.** | BAB 1 sudah menulis 0.0 sebagai keputusan formal (RY(0) = identitas, benar-benar netral secara rotasi); `noa3.ipynb` dianggap versi lama yang belum disinkronkan. |
| Scaler | `MinMaxScaler(0, PI)` fit hanya 7 kolom Hu asli (kolom padding dikecualikan dari fitting). | Sudah dikunci sebelumnya, tidak berubah. |
| Clipping setelah scaling | **Tidak ada** — konsisten dengan PCA. | Sesi 2026-09-22: kebijakan clipping diseragamkan lintas ketiga `feature_method` supaya tidak jadi confound tersembunyi. |

### 2.8.3 ZERNIKE MOMENTS — Status: DIKUNCI (menunggu implementasi & unit test — lihat Gate G2-02)

| Aspek | Pengaturan final | Keputusan diambil |
|---|---|---|
| Fungsi `MAP_IMAGE_TO_UNIT_DISK(img)` | Didefinisikan di 2.8.4 — pusat = pusat geometris citra `((W-1)/2, (H-1)/2)`; radius = `MIN(H,W)/2 = 14`; piksel dengan `r > 1` (di luar disk satuan) di-mask ke `0`; tanpa interpolasi (koordinat piksel dipakai langsung). | Konvensi standar `mahotas.features.zernike_moments`, konsisten dengan eksplorasi awal (`radius=14`) di `/areas/more-hd.md`. |
| Fungsi `EXTRACT_ZERNIKE_TERMS(...)` | Didefinisikan di 2.8.4 — piksel di luar disk sudah `0` sejak tahap mapping; magnitude `\|Z_nm\|` diambil per pasangan `(n,m)` dari hasil komputasi Zernike hingga derajat maksimum yang dibutuhkan. | Pseudocode fungsi ditambahkan 2026-09-22, sebelumnya black-box. |
| 8 pasangan `(n,m)` | **Direvisi**: `[(2,0), (2,2), (3,1), (3,3), (4,0), (4,2), (5,1), (5,5)]`. | Sesi 2026-09-22: `(0,0)` (momen area, nyaris konstan pada citra ternormalisasi) dan `(1,1)` (terkait pusat massa, mendekati nol karena citra sudah disentralkan ke pusat disk) dibuang dan diganti derajat 5, supaya seluruh 8 dimensi berpotensi diskriminatif antar kelas digit. |
| Magnitude | `zernike_use_magnitude = TRUE` — dikunci (pakai `\|Z_nm\|` untuk invariansi rotasi). | Tidak berubah. |
| Scaler | `MinMaxScaler(0, PI)` fit pada seluruh 8 kolom — konsisten dengan PCA (tidak ada perlakuan khusus seperti Hu). | Tidak berubah. |
| Clipping setelah scaling | **Tidak ada** — sama seperti Hu dan PCA. | Sesi 2026-09-22: kebijakan clipping diseragamkan lintas ketiga `feature_method`. |

**Catatan penyelarasan dengan eksplorasi lama:** eksplorasi Zernike sebelumnya (`/areas/more-hd.md`) memakai skema 7 magnitude + 1 kanal padding konstan, mirip perlakuan Hu. Desain final di dokumen ini **sengaja berbeda**: Zernike menghasilkan 8 koefisien asli tanpa kanal padding (hanya Hu yang memakai interface-compromise padding), sesuai `D-01` di `MORE_HD_RESEARCH_READINESS_GATES`. BAB 1 dan FRD-09 perlu disamakan ke versi ini (lihat `D-01`, `D-03`).

### 2.8.4 PSEUDOCODE FUNGSI EKSTRAKSI HU DAN ZERNIKE

Sebelumnya `HU_MOMENTS`, `MAP_IMAGE_TO_UNIT_DISK`, dan `EXTRACT_ZERNIKE_TERMS`
dipanggil sebagai black-box di bagian 2 (`DATA_PIPELINE`). Berikut definisi
pseudocode-nya, mengikuti keputusan yang dikunci di 2.8.2 dan 2.8.3.

```
FUNCTION HU_MOMENTS(img):
    # img: array grayscale ternormalisasi (H, W), nilai piksel di [0,1].
    # Tidak dibinerkan -- nilai piksel dipakai langsung sebagai "massa".
    raw_moments = IMAGE_MOMENTS(img)              # m00, m10, m01, mu20, mu11, mu02, ... (central moments)
    hu = CENTRAL_MOMENTS_TO_HU(raw_moments)        # h1..h7, formula standar Hu (1962)
    RETURN hu   # vector panjang 7; float64; bisa sangat kecil dan bertanda


FUNCTION SIGNED_LOG_TRANSFORM(hu_batch, epsilon):
    # Menstabilkan dynamic range Hu Moments yang bisa sangat kecil (~1e-20) dan bertanda.
    n_non_finite = 0
    FOR each h IN hu_batch (elementwise):
        h_transformed = -SIGN(h) * LOG10(ABS(h) + epsilon)
        IF NOT IS_FINITE(h_transformed):
            h_transformed = 0.0
            n_non_finite = n_non_finite + 1

    RECORD_COUNTER("n_non_finite_hu", n_non_finite)   # diteruskan ke feature_metadata.json
    RETURN h_transformed_batch


FUNCTION MAP_IMAGE_TO_UNIT_DISK(img):
    # img: citra grayscale ternormalisasi (H, W); untuk MNIST, H = W = 28.
    center_x, center_y = (WIDTH(img) - 1) / 2, (HEIGHT(img) - 1) / 2   # pusat geometris, BUKAN pusat massa
    radius = MIN(HEIGHT(img), WIDTH(img)) / 2                          # = 14 untuk 28x28

    masked_img = ZEROS_LIKE(img)
    FOR each pixel (x, y) IN img:
        r = DISTANCE((x, y), (center_x, center_y)) / radius
        IF r <= 1.0:
            masked_img[y][x] = img[y][x]     # tanpa interpolasi -- koordinat piksel dipakai langsung
        # ELSE: tetap 0 (di luar disk satuan, tidak berkontribusi ke integral momen)

    RETURN { "pixels": masked_img, "center": (center_x, center_y), "radius": radius }


FUNCTION EXTRACT_ZERNIKE_TERMS(disk_image, terms, use_magnitude):
    max_degree = MAX(n FOR (n, m) IN terms)

    all_moments = COMPUTE_ZERNIKE_MOMENTS(
        disk_image["pixels"],
        radius = disk_image["radius"],
        degree = max_degree,
        center = disk_image["center"]
    )
    # all_moments: dict {(n,m): complex_value}, dihitung sekali hingga max_degree

    result = []
    FOR each (n, m) IN terms:
        z = all_moments[(n, m)]
        IF use_magnitude == TRUE:
            result.APPEND( ABS(z) )     # |Z_nm| -- invarian rotasi
        ELSE:
            result.APPEND( z )

    RETURN result   # vector panjang 8
```

Catatan implementasi: `IMAGE_MOMENTS` / `CENTRAL_MOMENTS_TO_HU` dan
`COMPUTE_ZERNIKE_MOMENTS` di atas mewakili library yang dipakai saat kode
asli ditulis (mis. `cv2.moments` + `cv2.HuMoments` untuk Hu, atau
`mahotas.features.zernike_moments` untuk Zernike) — pemilihan library
persis akan dikunci saat implementasi Python dimulai, bukan di tahap
pseudocode ini.

---

## 3. CORRELATION MATRIX
 
```
FUNCTION CORRELATION_MATRIX(X_train_scaled, y_train, classes):
    K = LENGTH(classes)
    S = ZEROS(K, K)
 
    # 3.1 Hitung rata-rata fitur per kelas
    class_means = {}
    FOR each c IN classes:
        class_means[c] = MEAN(X_train_scaled WHERE y_train == c)
 
    # 3.2 Isi matriks S dengan MSE antar rata-rata kelas
    FOR i IN range(K):
        FOR j IN range(K):
            IF i == j:
                S[i][j] = -1                      # placeholder, diisi ulang di 3.3
            ELSE:
                S[i][j] = MEAN_SQUARED_ERROR(class_means[classes[i]], class_means[classes[j]])
 
    # 3.3 Normalisasi sel non-diagonal ke rentang [0.5, 1]
    off_diag_values = ALL S[i][j] WHERE i != j
    S_normalized = NORMALIZE(off_diag_values, range=[0.5, 1])
    PUT_BACK S_normalized INTO S (kecuali diagonal)
 
    # 3.4 Set diagonal = -1 (menandakan "kelas sama, harus didekatkan")
    FOR i IN range(K):
        S[i][i] = -1
 
    SAVE(S, run_dir + "/artifacts/correlation_matrix.npy")
    RETURN S
```
 
---
 
## 4. MODEL SETUP
 
Sirkuit dipisah jadi **dua fungsi pembangun berbeda** — `BUILD_CIRCUIT_MORE_HD`
dan `BUILD_CIRCUIT_MORE_HD_C`. MORE-HD adalah arsitektur **tetap** (acuan
pembanding, harus selalu identik dengan skripsi lama). MORE-HD-C adalah
arsitektur baru hasil keputusan kita: **hanya varian V1 (RY+RZ)** yang
dipakai — bukan lagi tiga cabang V1/V2/V3. V2 dan V3 tetap didokumentasikan
di bagian 4.4 sebagai cadangan riset lanjutan, tapi tidak dipanggil di
alur aktif manapun sekarang.
 
```
FUNCTION BUILD_PAIRED_PRIMARY_INITIALIZATION(master_seed):
    rng_ry = RNG(DERIVE_SUBSEED(master_seed, "init:ry_core"))
    rng_rz = RNG(DERIVE_SUBSEED(master_seed, "init:rz_phase"))

    ry_core  = RANDOM_UNIFORM_RNG(rng_ry, low=0, high=2*PI, size=30)
    rz_phase = RANDOM_UNIFORM_RNG(rng_rz, low=0, high=2*PI, size=30)

    # RZ=0 tidak dipakai untuk phase-control. Exact zero dire-sample deterministik.
    FOR i IN range(30):
        WHILE rz_phase[i] == 0.0:
            rz_phase[i] = RANDOM_UNIFORM_RNG(rng_rz, low=0, high=2*PI, size=1)

    RETURN {
        "ry_core": ry_core,
        "rz_phase": rz_phase,
        "ry_core_seed": DERIVE_SUBSEED(master_seed, "init:ry_core"),
        "rz_phase_seed": DERIVE_SUBSEED(master_seed, "init:rz_phase")
    }


FUNCTION PACK_RY_RZ_BY_LAYER(ry_core, rz_phase, n_layers=3, n_qubits=10):
    ry_layers = RESHAPE(ry_core,  (n_layers, n_qubits))
    rz_layers = RESHAPE(rz_phase, (n_layers, n_qubits))
    packed = []

    FOR layer_idx IN range(n_layers):
        packed.EXTEND(ry_layers[layer_idx])   # 10 RY parameter
        packed.EXTEND(rz_layers[layer_idx])   # 10 RZ parameter

    RETURN ARRAY(packed)   # length 60; cocok dengan BUILD_CIRCUIT_MORE_HD_C


FUNCTION MODEL_SETUP(config):
    init_bundle = BUILD_PAIRED_PRIMARY_INITIALIZATION(config.seed)

    IF config.architecture == "MORE-HD":
        circuit_fn, n_params = BUILD_CIRCUIT_MORE_HD(
            n_data_qubits=config.n_data_qubits,
            n_readout_qubits=config.n_readout_qubits
        )
        initial_params = COPY(init_bundle.ry_core)

    ELSE IF config.architecture == "MORE-HD-C":
        circuit_fn, n_params = BUILD_CIRCUIT_MORE_HD_C(
            n_data_qubits=config.n_data_qubits,
            n_readout_qubits=config.n_readout_qubits
        )
        initial_params = PACK_RY_RZ_BY_LAYER(
            init_bundle.ry_core,
            init_bundle.rz_phase
        )

    ELSE:
        RAISE_ERROR("architecture harus 'MORE-HD' atau 'MORE-HD-C'")

    ASSERT LENGTH(initial_params) == n_params

    SAVE(initial_params, run_dir + "/artifacts/initial_params.npy")
    SAVE_JSON(
        {
            "ry_core_seed": init_bundle.ry_core_seed,
            "rz_phase_seed": init_bundle.rz_phase_seed,
            "ry_core_sha256": SHA256_ARRAY(init_bundle.ry_core),
            "rz_phase_sha256": SHA256_ARRAY(init_bundle.rz_phase),
            "packing_rule": (
                "RY_ONLY_3x10" IF config.architecture == "MORE-HD"
                ELSE "PER_LAYER_[10_RY,10_RZ]_x3"
            )
        },
        run_dir + "/artifacts/initialization_manifest.json"
    )

    RETURN circuit_fn, initial_params
```
 
### 4.1 BUILD_CIRCUIT_MORE_HD — Arsitektur Baseline (Tetap, Tidak Ada Varian)
 
```
FUNCTION BUILD_CIRCUIT_MORE_HD(n_data_qubits=8, n_readout_qubits=2, n_layers=3):
    # Arsitektur ini TIDAK menerima parameter varian -- selalu identik
    # dengan MORE-HD di skripsi lama, jadi acuan pembanding yang stabil.
 
    ASSERT n_data_qubits == 8
    ASSERT n_readout_qubits == 2
    n_qubits         = n_data_qubits + n_readout_qubits   # 10
    params_per_layer = n_qubits                           # 1 RY / qubit / layer
    n_params         = n_layers * params_per_layer         # 3 * 10 = 30 (sama persis skripsi lama)
 
    FUNCTION circuit_fn(x, theta):
        theta_layers = RESHAPE(theta, (n_layers, params_per_layer))
 
        # --- Encoding ---
        FOR i IN range(n_data_qubits):
            APPLY_GATE( RY(x[i]), wire=i )
 
        # --- Variational layers ---
        FOR layer_idx IN range(n_layers):
            RY_ONLY_LAYER(theta_layers[layer_idx], n_qubits)
            ENTANGLING_BLOCK_CNOT(n_data_qubits, n_readout_qubits)
 
        # --- Measurement ---
        RETURN MEASURE_15_OBSERVABLES(readout_wires=[n_data_qubits, n_data_qubits + 1])
 
    RETURN circuit_fn, n_params
```
 
### 4.2 BUILD_CIRCUIT_MORE_HD_C — Arsitektur Baru (V1: RY+RZ)
 
```
FUNCTION BUILD_CIRCUIT_MORE_HD_C(n_data_qubits=8, n_readout_qubits=2, n_layers=3):
    # Sama seperti MORE-HD di semua hal, KECUALI lapisan variational:
    # tiap qubit dapat RY *dan* RZ (bukan RY saja). Entangling & measurement
    # tidak berubah sama sekali dari MORE-HD.
 
    ASSERT n_data_qubits == 8
    ASSERT n_readout_qubits == 2
    n_qubits         = n_data_qubits + n_readout_qubits   # tetap 10, sama seperti MORE-HD
 
    params_per_layer = n_qubits * 2                        # RY + RZ per qubit -> 20
    n_params         = n_layers * params_per_layer          # 3 * 20 = 60
 
    FUNCTION circuit_fn(x, theta):
        theta_layers = RESHAPE(theta, (n_layers, params_per_layer))
 
        # --- Encoding (SAMA dengan MORE-HD) ---
        FOR i IN range(n_data_qubits):
            APPLY_GATE( RY(x[i]), wire=i )
 
        # --- Variational layers: RY + RZ (INI SATU-SATUNYA PERBEDAAN dari MORE-HD) ---
        FOR layer_idx IN range(n_layers):
            RY_RZ_LAYER(theta_layers[layer_idx], n_qubits)
            ENTANGLING_BLOCK_CNOT(n_data_qubits, n_readout_qubits)   # SAMA dengan MORE-HD
 
        # --- Measurement (SAMA dengan MORE-HD, urutan observable identik) ---
        RETURN MEASURE_15_OBSERVABLES(readout_wires=[n_data_qubits, n_data_qubits + 1])
 
    RETURN circuit_fn, n_params
```
 
**Perbandingan jumlah parameter** (n_layers=3, n_qubits=10):
 
| Arsitektur | params_per_layer | Total (× 3 layer) |
|---|---|---|
| MORE-HD (baseline) | 10 (RY saja) | **30** |
| MORE-HD-C (V1: RY+RZ) | 20 (RY + RZ) | **60** |
 
Perbandingannya dengan MORE asli (91 parameter): MORE-HD-C masih memakai kurang dari 2/3-nya, jadi keunggulan efisiensi parameter yang jadi kontribusi utama skripsi lama tetap terjaga.
 
### 4.3 Fungsi Bantu Bersama (Dipakai oleh Kedua Arsitektur)
 
```
# --- Lapisan variational: dipakai MORE-HD ---
FUNCTION RY_ONLY_LAYER(theta_layer, n_qubits):
    FOR q IN range(n_qubits):
        APPLY_GATE( RY(theta_layer[q]), wire=q )
 
 
# --- Lapisan variational: dipakai MORE-HD-C (V1) ---
FUNCTION RY_RZ_LAYER(theta_layer, n_qubits):
    FOR q IN range(n_qubits):
        APPLY_GATE( RY(theta_layer[q]),            wire=q )
        APPLY_GATE( RZ(theta_layer[n_qubits + q]), wire=q )
 
 
# --- Entangling: dipakai KEDUA arsitektur, tidak berubah sama sekali ---
FUNCTION ENTANGLING_BLOCK_CNOT(n_data_qubits, n_readout_qubits):
    # (a) circular entanglement di antara data qubit
    FOR i IN range(n_data_qubits - 1):
        APPLY_GATE( CNOT, control=i, target=i+1 )
    APPLY_GATE( CNOT, control=n_data_qubits - 1, target=0 )
 
    # (b) konektivitas data qubit terakhir -> kedua readout qubit
    APPLY_GATE( CNOT, control=n_data_qubits - 1, target=n_data_qubits )
    APPLY_GATE( CNOT, control=n_data_qubits - 1, target=n_data_qubits + 1 )
 
    # (c) readout correlation
    APPLY_GATE( CNOT, control=n_data_qubits, target=n_data_qubits + 1 )
 
 
# --- Pengukuran: dipakai KEDUA arsitektur, urutan dikunci tetap ---
FUNCTION MEASURE_15_OBSERVABLES(readout_wires):
    q8, q9 = readout_wires
    pauli_symbols = ["I", "X", "Y", "Z"]
    observables = []
    dimension_labels = []
 
    FOR a IN pauli_symbols:
        FOR b IN pauli_symbols:
            IF NOT (a == "I" AND b == "I"):          # II dibuang (tidak informatif)
                observables.APPEND( PAULI_TENSOR(a, wire=q8, b, wire=q9) )
                dimension_labels.APPEND(a + b)
    # urutan HASIL TETAP: [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]
    # dikunci sama persis dgn Tabel III-2 skripsi lama, supaya index dimensi
    # bisa dibandingkan lintas MORE-HD vs MORE-HD-C
 
    RETURN [ EXPECTATION_VALUE(obs) FOR obs IN observables ]
```
 
### 4.4 Cadangan Riset Lanjutan — V2 & V3 (Belum Aktif, Tidak Dipanggil)
 
Kalau nanti V1 terbukti belum cukup mengatasi kegagalan di 8-10 kelas,
dua kandidat ini sudah didokumentasikan sebagai langkah lanjutan —
**tidak dipanggil di kode aktif manapun sekarang**, murni catatan:
 
```
# CADANGAN -- belum dipanggil di manapun
FUNCTION U3_LAYER(theta_layer, n_qubits):        # untuk V2: RZ-RY-RZ per qubit (30 param/layer)
    FOR q IN range(n_qubits):
        idx = q * 3
        APPLY_GATE( RZ(theta_layer[idx]),     wire=q )
        APPLY_GATE( RY(theta_layer[idx + 1]), wire=q )
        APPLY_GATE( RZ(theta_layer[idx + 2]), wire=q )
 
# CADANGAN -- belum dipanggil di manapun
FUNCTION ENTANGLING_BLOCK_CRZ(n_data_qubits, n_readout_qubits, theta_entangle):   # untuk V3
    # Topologi SAMA PERSIS dengan ENTANGLING_BLOCK_CNOT, tapi tiap CNOT
    # diganti CRZ(theta) berparameter.
    idx = 0
    FOR i IN range(n_data_qubits - 1):
        APPLY_GATE( CRZ(theta_entangle[idx]), control=i, target=i+1 )
        idx = idx + 1
    APPLY_GATE( CRZ(theta_entangle[idx]), control=n_data_qubits - 1, target=0 )
    idx = idx + 1
    APPLY_GATE( CRZ(theta_entangle[idx]), control=n_data_qubits - 1, target=n_data_qubits )
    idx = idx + 1
    APPLY_GATE( CRZ(theta_entangle[idx]), control=n_data_qubits - 1, target=n_data_qubits + 1 )
    idx = idx + 1
    APPLY_GATE( CRZ(theta_entangle[idx]), control=n_data_qubits, target=n_data_qubits + 1 )
```
 
### 4.5 Penjelasan Tiap Layer: MORE-HD vs MORE-HD-C (V1)
 
Kedua arsitektur dibangun dari 4 layer yang sama, dengan `n_layers = 3`
(layer Variational dan Entangling diulang 3 kali). **Cuma Layer 2
(Variational) yang berbeda** — layer lain menyumbang nol parameter yang
dilatih, dengan alasan berbeda-beda per layer, sebagaimana dirinci pada
tabel berikut.
 
| Layer | Diterapkan pada | Jenis gerbang | Param/qubit | Jumlah qubit | Param per 1× layer | Diulang | Kontribusi total |
|---|---|---|---|---|---|---|---|
| **1. Encoding** | 8 data qubit | RY(x) — sudut = nilai fitur data, **bukan** parameter yang dilatih | 0 | 8 | 0 | 1× (sekali di depan, di luar loop layer) | **0** (kedua arsitektur) |
| **2. Variational** | 10 qubit (8 data + 2 readout) | RY saja *(MORE-HD)* / RY+RZ *(MORE-HD-C)* | **1** *(MORE-HD)* / **2** *(MORE-HD-C)* | 10 | 10 *(MORE-HD)* / 20 *(MORE-HD-C)* | **3×** (`n_layers`) | **30** *(MORE-HD)* / **60** *(MORE-HD-C)* |
| **3. Entangling** | pasangan qubit (control→target) | CNOT — gerbang permutasi tetap, tidak punya sudut rotasi | 0 | — | 0 | 3× (dalam loop yang sama) | **0** (kedua arsitektur) |
| **4. Measurement** | 2 readout qubit | Observable Pauli — ini pengukuran, bukan gerbang berparameter | 0 | — | 0 | 1× (sekali di akhir) | **0** (kedua arsitektur) |
| | | | | | | **TOTAL** | **30 vs 60** |
 
**Formula umum:** `total_parameter = n_layers × n_qubits × n_rotasi_per_qubit`
 
- MORE-HD: `3 × 10 × 1 = 30` (1 rotasi/qubit karena cuma RY)
- MORE-HD-C: `3 × 10 × 2 = 60` (2 rotasi/qubit karena RY **dan** RZ, masing-masing punya sudut sendiri)
Selisih tepat 2× ini murni berasal dari Layer 2 — Layer 1, 3, dan 4 dipakai
**identik** oleh kedua arsitektur sehingga tidak menyumbang selisih sama
sekali. Ini pembuktian ulang secara numerik bahwa "isolasi variabel" yang
kita rancang benar-benar tercermin di hitungan parameter, bukan cuma
klaim di teks.
 
**Kenapa cuma Layer 2 yang beda:** akar masalah MORE-HD (6 dari 15 dimensi
selalu nol) disebabkan murni karena *state* kuantum selalu riil — dan itu
konsekuensi dari gerbang RY dan CNOT yang keduanya bermatriks riil.
Menambahkan RZ di layer variational adalah cara paling langsung menyuntik fase kompleks **tanpa mengubah** cara data masuk (layer 1), cara qubit saling terhubung (layer 3), atau cara hasil diukur (layer 4). Namun, A-vs-D juga menaikkan jumlah parameter terlatih 30→60, sehingga perubahan performa tidak boleh langsung diatribusikan hanya kepada RZ/fase kompleks; targeted ablation B/C pada Bagian 11 digunakan untuk memisahkan confound tersebut.
 
**Catatan pemilihan run:** satu kali menjalankan `MAIN(config)` hanya
menjalankan **satu kombinasi** `architecture × feature_method × classes`.
Karena itu perbandingan faktorial dilakukan melalui pemanggilan `MAIN(config)`
secara manual untuk setiap kondisi, dengan `run_dir` unik. Benchmark lengkap
3–10 kelas tetap terdiri dari 48 kondisi per seed, tetapi setiap kondisi dijalankan
dan dimonitor secara terpisah, bukan melalui batch runner otomatis.
 
---

## 4.6 PROTOKOL PEMBENTUKAN PASANGAN CLUSTERING — G1-03/G1-08 (DIKUNCI 2026-09-27)

Protokol utama sengaja dipertahankan sedekat mungkin dengan MORE asli: tepat
**5 instance training per kelas** dipilih untuk membentuk clustering dataset.
Perbedaannya, implementasi penelitian ini mengunci sampling secara deterministik
dan menyimpan manifest pasangan agar prosedur dapat diaudit dan direproduksi.

Aturan yang dibekukan:

1. Sumber pasangan hanya `X_train`/`y_train`; validation dan official test tidak pernah
   dipakai untuk membentuk pasangan.
2. `n_cluster_pair_samples = 5` untuk setiap kelas.
3. Sampling dilakukan **tanpa replacement** dan tepat satu kali sebelum COBYLA
   dimulai. Tidak ada resampling pada setiap objective-function evaluation.
4. Root RNG pairing adalah `pair_seed = DERIVE_SUBSEED(master_seed, "cluster_pairs")`.
   Untuk menjaga nested-K secara eksplisit, setiap kelas memakai substream
   `DERIVE_SUBSEED(master_seed, "cluster_pairs:" + class_id)`. Karena itu lima
   sampel digit yang sudah ada tetap identik ketika K bertambah.
5. Untuk seed dan K yang sama, identitas sampel pasangan harus sama lintas
   `PCA/HU/ZERNIKE` dan `MORE-HD/MORE-HD-C`. Feature extraction hanya mengubah
   representasi sampel, bukan identitas sampel yang dipilih.
6. Dari union 5K sampel terpilih, pair builder membentuk **seluruh unordered unique
   pairs** dengan aturan `i < j`. Self-pair `(i,i)`, duplicate pair, dan pasangan
   terbalik ganda `(i,j)/(j,i)` dilarang.
7. Distribusi pasangan alami dipertahankan: tidak ada downsampling, oversampling,
   balanced mean, atau bobot tambahan antara same-class dan different-class.
   Dengan kata lain `pair_balance_policy="NATURAL_FULL_PAIRING"` dan
   `pair_weighting="NONE"`.
8. Ketidakseimbangan pair diakui sebagai karakteristik protokol MORE. Statistiknya
   wajib disimpan dan dipertimbangkan saat menginterpretasikan efek K; hasil tidak
   boleh mengatribusikan seluruh perubahan terhadap curse of density tanpa
   mempertimbangkan perubahan komposisi pasangan.

Untuk m=5 sampel per kelas:

```
n_selected_samples = 5K
n_pairs_total       = C(5K, 2)
n_pairs_same        = K * C(5, 2) = 10K
n_pairs_different   = C(K, 2) * 25
```

Contoh: K=3 menghasilkan 105 pair (30 same, 75 different), sedangkan K=10
menghasilkan 1.225 pair (100 same, 1.125 different).

```
FUNCTION BUILD_PAIRING_DATASET(X_train, y_train, config):
    ASSERT config.n_cluster_pair_samples == 5
    ASSERT config.pair_balance_policy == "NATURAL_FULL_PAIRING"
    ASSERT config.pair_weighting == "NONE"

    selected = []
    selected_manifest = {}

    FOR each class_id IN config.classes:
        class_positions = INDICES_WHERE(y_train == class_id)
        ASSERT LENGTH(class_positions) >= 5

        class_pair_seed = DERIVE_SUBSEED(
            config.seed,
            "cluster_pairs:" + STRING(class_id)
        )

        chosen_positions = SAMPLE_WITHOUT_REPLACEMENT(
            class_positions,
            n=5,
            rng=RNG(class_pair_seed)
        )
        chosen_positions = SORT_ASCENDING(chosen_positions)

        selected_manifest[class_id] = {
            "class_pair_seed": class_pair_seed,
            "train_positions": chosen_positions
        }

        FOR each pos IN chosen_positions:
            selected.APPEND({
                "train_position": pos,
                "class_id": class_id,
                "x": X_train[pos]
            })

    selected = SORT_BY(selected, keys=["class_id", "train_position"])

    pairing_dataset = []
    FOR a IN range(0, LENGTH(selected)):
        FOR b IN range(a + 1, LENGTH(selected)):
            left  = selected[a]
            right = selected[b]

            pairing_dataset.APPEND({
                "x_i": left.x,
                "x_j": right.x,
                "class_i": left.class_id,
                "class_j": right.class_id,
                "train_position_i": left.train_position,
                "train_position_j": right.train_position
            })

    K = LENGTH(config.classes)
    expected_same      = K * COMBINATION(5, 2)
    expected_different = COMBINATION(K, 2) * 25
    expected_total     = COMBINATION(5 * K, 2)

    n_same = COUNT(pair IN pairing_dataset WHERE pair.class_i == pair.class_j)
    n_different = LENGTH(pairing_dataset) - n_same

    ASSERT LENGTH(pairing_dataset) == expected_total
    ASSERT n_same == expected_same
    ASSERT n_different == expected_different
    ASSERT COUNT_SELF_PAIRS(pairing_dataset) == 0
    ASSERT COUNT_DUPLICATE_UNORDERED_PAIRS(pairing_dataset) == 0

    pair_manifest = {
        "master_seed": config.seed,
        "pair_seed": DERIVE_SUBSEED(config.seed, "cluster_pairs"),
        "n_samples_per_class": 5,
        "selected_by_class": selected_manifest,
        "pair_order": "unordered_i_lt_j",
        "sampling": "without_replacement",
        "frozen_before_optimizer": TRUE,
        "balance_policy": "NATURAL_FULL_PAIRING",
        "pair_weighting": "NONE"
    }

    pair_stats = {
        "K": K,
        "n_selected_samples": 5 * K,
        "n_pairs_total": LENGTH(pairing_dataset),
        "n_pairs_same_class": n_same,
        "n_pairs_different_class": n_different,
        "same_class_ratio": n_same / LENGTH(pairing_dataset),
        "different_class_ratio": n_different / LENGTH(pairing_dataset),
        "duplicate_pairs": 0,
        "self_pairs": 0
    }

    SAVE_ATOMIC_JSON(pair_manifest, run_dir + "/artifacts/pair_manifest.json")
    SAVE_ATOMIC_JSON(pair_stats,    run_dir + "/artifacts/pair_stats.json")

    RETURN pairing_dataset, pair_stats
```

Unit test implementasi Python nantinya wajib memverifikasi rumus jumlah pair,
ketiadaan self/duplicate pair, determinisme seed, nested-K, dan kesamaan identitas
sampel pairing lintas architecture/feature_method untuk seed dan K yang sama.

---

## 5. CLUSTERING LOOP
 
Tujuan tahap ini: melatih parameter agar *output* sirkuit dari kelas yang sama
saling berdekatan, dan kelas berbeda saling menjauh — **bukan** untuk
meminimalkan "loss test", makanya evaluasi generalisasinya diganti jadi
cek struktur centroid (pseudo-accuracy + margin), bukan angka loss test.

**G0-01 (sesi 2026-09-22):** fungsi ini hanya menerima `X_val`/`y_val`, TIDAK
`X_test`/`y_test` sama sekali. **G1-02 (2026-09-26):** satu pemanggilan
`objective_clustering(theta)` adalah satu **objective-function evaluation** dan
dicatat dengan `eval_id`; ini tidak disebut sebagai iterasi COBYLA. **G1-03/G1-08 (2026-09-27):**
`pairing_dataset` memakai 5 sampel/kelas, seluruh unordered unique pairs, tanpa
balancing/reweighting, dan dibentuk hanya sekali sebelum optimizer dimulai.
 
```
FUNCTION CLUSTERING_LOOP(circuit_fn, initial_params, X_train, y_train, X_val, y_val, S, config):
    SET_RANDOM_SEED(config.seed)
 
    # Dibangun SATU KALI dan dibekukan sebelum objective pertama.
    pairing_dataset, pair_stats = BUILD_PAIRING_DATASET(
        X_train, y_train, config
    )
 
    n_objective_evals = 0
    callback_id = 0
    best_observed_fun = +INFINITY
    best_observed_eval_id = NULL
    best_observed_theta = NULL

    n_params = LENGTH(initial_params)          # trainable vector yang dioptimasi COBYLA
    VALIDATE_OPTIMIZER_BUDGET(config.max_nfev_clustering, n_params, config.run_mode)

    CREATE_EMPTY_BINARY_LOG(
        run_dir + "/artifacts/clustering_params.bin",
        record_size = SIZEOF(initial_params)
    )
    CREATE_EMPTY_FILE(run_dir + "/logs/clustering_log.jsonl")
    CREATE_EMPTY_FILE(run_dir + "/logs/clustering_callback_log.jsonl")
 
    FUNCTION objective_clustering(theta):
        NONLOCAL n_objective_evals
        NONLOCAL best_observed_fun, best_observed_eval_id, best_observed_theta

        eval_id = n_objective_evals
 
        # --- (a) objective train_loss yang dibaca COBYLA ---
        pair_losses = []
        FOR each (x_i, x_j, class_i, class_j) IN pairing_dataset:
            v_i = circuit_fn(x_i, theta)
            v_j = circuit_fn(x_j, theta)
            dist = COSINE_DISTANCE(v_i, v_j)
            s_ij = S[class_i][class_j]
            pair_losses.APPEND(-s_ij * dist)
        train_loss = MEAN(pair_losses)
 
        # --- (b) centroid sementara dari seluruh train ---
        temp_centroids = {}
        FOR each c IN config.classes:
            outputs_c = [circuit_fn(x, theta) FOR x IN X_train WHERE y_train == c]
            temp_centroids[c] = NORMALIZE_VECTOR(MEAN(outputs_c))
 
        # --- (c) monitoring VALIDATION, pasif; tidak masuk objective ---
        correct = 0
        margins = []
        true_distances_val = []
        val_outputs_by_class = { c: [] FOR c IN config.classes }

        FOR each (x, true_class) IN ZIP(X_val, y_val):
            v = circuit_fn(x, theta)
            val_outputs_by_class[true_class].APPEND(v)

            distances = { c: COSINE_DISTANCE(v, temp_centroids[c]) FOR c IN config.classes }
            predicted_class = ARGMIN(distances)
            IF predicted_class == true_class:
                correct += 1
            dist_true  = distances[true_class]
            dist_other = MIN(distances[c] FOR c IN config.classes IF c != true_class)
            true_distances_val.APPEND(dist_true)
            margins.APPEND(dist_other - dist_true)
 
        pseudo_accuracy_val = correct / LENGTH(X_val)
        avg_margin_val = MEAN(margins)
        mean_true_distance_val = MEAN(true_distances_val)

        val_centroids = {}
        FOR each c IN config.classes:
            val_centroids[c] = NORMALIZE_VECTOR(MEAN(val_outputs_by_class[c]))

        min_separation_val = MIN(
            COSINE_DISTANCE(val_centroids[a], val_centroids[b])
            FOR all pairs (a, b) IN config.classes WHERE a != b
        )

        min_separation = MIN(
            COSINE_DISTANCE(temp_centroids[a], temp_centroids[b])
            FOR all pairs (a, b) IN config.classes WHERE a != b
        )
        correlation_consistency = SPEARMAN_CORRELATION(
            pairwise_values_of(S),
            pairwise_cosine_distances_of(temp_centroids)
        )
        active_dimensions = COUNT(
            dim IN range(15)
            WHERE ANY(
                ABS(temp_centroids[c][dim]) > config.active_dim_threshold
                FOR c IN config.classes
            )
        )

        # Record ke-eval_id di binary log berisi theta yang benar-benar dievaluasi.
        APPEND_PARAM_RECORD(run_dir + "/artifacts/clustering_params.bin", theta)

        IF train_loss < best_observed_fun:
            best_observed_fun = train_loss
            best_observed_eval_id = eval_id
            best_observed_theta = COPY(theta)

        log_entry = {
            "eval_id": eval_id,
            "phase": OBJECTIVE_PHASE(eval_id, n_params),   # G1-06
            "train_loss": train_loss,
            "pseudo_accuracy_val": pseudo_accuracy_val,
            "avg_margin_val": avg_margin_val,
            "mean_true_distance_val": mean_true_distance_val,
            "min_separation_val": min_separation_val,
            "min_separation": min_separation,
            "correlation_consistency": correlation_consistency,
            "active_dimensions": active_dimensions
        }
        APPEND_LINE(run_dir + "/logs/clustering_log.jsonl", TO_JSON(log_entry))

        n_objective_evals = n_objective_evals + 1
        RETURN train_loss

    FUNCTION clustering_callback(callback_state):
        NONLOCAL callback_id
        theta_callback = EXTRACT_THETA_FROM_COBYLA_CALLBACK(callback_state)

        callback_entry = {
            "callback_id": callback_id,
            "objective_evals_seen": n_objective_evals,
            "theta_source": "optimizer_callback"
        }
        APPEND_LINE(
            run_dir + "/logs/clustering_callback_log.jsonl",
            TO_JSON(callback_entry)
        )
        callback_id = callback_id + 1
 
    # maxiter milik wrapper SciPy diperlakukan sebagai budget maksimum nfev.
    result = COBYLA_MINIMIZE(
        objective_clustering,
        x0=initial_params,
        callback=clustering_callback,
        options={
            "maxiter": config.max_nfev_clustering,
            "tol": config.cobyla_tol,
            "rhobeg": config.cobyla_rhobeg
        }
    )

    trained_params_clustering = result.x
    ASSERT result.nfev == n_objective_evals

    # final point resmi dari optimizer dan best observed point disimpan TERPISAH.
    SAVE(result.x, run_dir + "/artifacts/clustering_params_final.npy")
    SAVE(best_observed_theta, run_dir + "/artifacts/clustering_params_best_observed.npy")

    optimizer_summary = {
        "optimizer": "COBYLA",
        "max_nfev": config.max_nfev_clustering,
        "tol": config.cobyla_tol,
        "rhobeg": config.cobyla_rhobeg,
        "scipy_version": SCIPY_VERSION(),
        "n_params": n_params,
        "simplex_size": n_params + 1,
        "n_initial_simplex_evals": MIN(result.nfev, n_params + 1),
        "n_optimization_evals": MAX(0, result.nfev - (n_params + 1)),
        "success": result.success,
        "status": result.status,
        "message": STRING(result.message),
        "fun": result.fun,
        "nfev": result.nfev,
        "callback_count": callback_id,
        "final_point_path": "artifacts/clustering_params_final.npy",
        "best_observed_eval_id": best_observed_eval_id,
        "best_observed_fun": best_observed_fun,
        "best_observed_point_path": "artifacts/clustering_params_best_observed.npy"
    }
    SAVE_JSON(
        optimizer_summary,
        run_dir + "/logs/clustering_optimizer_result.json"
    )

    RETURN trained_params_clustering
```
 
### Penjelasan Metrik Evaluasi pada Clustering Loop

Delapan nilai pada `clustering_log.jsonl` dicatat **per objective evaluation
(`eval_id`)**, bukan per iterasi optimizer. Dari delapan metrik tersebut,
hanya `train_loss` yang secara langsung digunakan COBYLA sebagai fungsi
objektif. Tujuh metrik lainnya merupakan **monitoring pasif** untuk mengamati
kualitas representasi clustering pada data train/validation tanpa memengaruhi
langkah optimasi COBYLA.

| Metrik | Fungsi (untuk apa dipakai) | Satuan / Rentang Nilai | Arah yang diharapkan | Interpretasi nilai yang lebih baik |
|---|---|---|---|---|
| `train_loss` | Objective utama yang diminimalkan COBYLA. | Skalar tak berdimensi dari `-S_ij × cosine_distance`. | ↓ **Semakin kecil semakin baik** | Nilai yang lebih kecil menunjukkan objective clustering semakin terpenuhi. |
| `pseudo_accuracy_val` | Nearest-centroid accuracy pada validation menggunakan centroid yang dibangun hanya dari TRAIN. | Proporsi 0–1. | ↑ **Semakin besar semakin baik** | Nilai mendekati 1 menunjukkan semakin banyak sampel validation yang paling dekat dengan centroid kelas benar. Ini menjadi kriteria pertama selector Jalur B. |
| `avg_margin_val` | Margin validation antara centroid kelas benar dan centroid kelas salah terdekat. | Selisih cosine distance; teoretis -2 sampai +2. | ↑ **Semakin besar semakin baik** | Margin positif yang lebih besar menunjukkan keputusan nearest-centroid lebih aman. Metrik ini tetap diagnostik dan tidak dipakai sebagai kriteria selector G0-02. |
| `mean_true_distance_val` | Rata-rata cosine distance sampel validation ke centroid TRAIN kelas benarnya. | 0–2. | ↓ **Semakin kecil semakin baik** | Mengukur compactness validation terhadap label/centroid kelas benar. Ini menjadi kriteria ketiga selector Jalur B. |
| `min_separation_val` | Jarak cosine minimum antar centroid yang dibangun dari output validation per kelas. | 0–2. | ↑ **Semakin besar semakin baik** | Menilai apakah kelas tetap terpisah pada validation. Ini menjadi kriteria kedua selector Jalur B. |
| `min_separation` | Jarak cosine minimum antar centroid kelas yang dibangun dari TRAIN. | 0–2. | ↑ **Semakin besar semakin baik** | Nilai lebih besar menunjukkan pasangan centroid TRAIN yang paling berdekatan masih mempunyai pemisahan lebih lebar. Tetap dipakai sebagai outcome representasi/diagnostik. |
| `correlation_consistency` | Konsistensi urutan jarak centroid TRAIN terhadap matriks korelasi `S`. | Spearman -1 sampai +1. | ↑ **Semakin mendekati +1 semakin baik** | Mengukur kesesuaian struktur jarak dengan dissimilarity antarkelas. Metrik diagnostik, bukan kriteria pemilihan checkpoint. |
| `active_dimensions` | Jumlah dimensi observable aktif pada centroid. | Integer 0–15. | ↑ **Semakin besar umumnya semakin baik untuk pemanfaatan ruang observable** | Dipakai untuk diagnosis structural zeros. **Tidak boleh** dipakai untuk memilih checkpoint Jalur B karena dapat secara sistematis menguntungkan MORE-HD-C terhadap MORE-HD. |

Ringkasan arah interpretasi:

```text
train_loss                  -> lebih kecil lebih baik
pseudo_accuracy_val         -> lebih besar lebih baik
avg_margin_val              -> lebih besar lebih baik (diagnostik)
mean_true_distance_val      -> lebih kecil lebih baik
min_separation_val          -> lebih besar lebih baik
min_separation              -> lebih besar lebih baik (diagnostik)
correlation_consistency     -> lebih besar / lebih dekat ke +1 lebih baik (diagnostik)
active_dimensions           -> lebih besar umumnya lebih baik; DIAGNOSTIK SAJA
```

Untuk G0-02, selector Jalur B tidak menggunakan weighted score. Pemilihan
dilakukan secara **lexicographic deterministik** pada objective evaluation yang
eligible: (1) maksimum `pseudo_accuracy_val`; (2) jika tie, maksimum
`min_separation_val`; (3) jika tie, minimum `mean_true_distance_val`; (4)
jika tie, minimum `train_loss`; dan (5) jika masih tie, pilih `eval_id`
paling awal. `active_dimensions`, `correlation_consistency`, dan
`avg_margin_val` tidak memengaruhi keputusan selector.

Domain kandidat dikunci oleh G1-06 (2026-09-27): hanya objective evaluation
dengan `phase == "optimization"` yang eligible. Seluruh fase `initial_simplex`
(`eval_id` 0..`n_params`, termasuk `x0`) dikecualikan karena titik-titik itu
adalah probe konstruksi simplex, bukan keputusan optimizer, dan jumlahnya
asimetris antar arsitektur (31 vs 61). G0-02 mengunci aturan ranking,
sedangkan G1-06 mengunci domain kandidat yang boleh diranking.

Kolom `phase` juga dicatat untuk setiap baris `clustering_log.jsonl` dan
`supervised_log.jsonl` (lihat §1.1). Langkah geometry-improvement COBYLA di
tengah optimasi tidak dapat dibedakan tanpa instrumentasi internal dan tetap
berlabel `optimization`; ini dicatat sebagai keterbatasan.

Seluruh monitoring pada bagian ini hanya memakai train/validation. Official test
tetap hanya digunakan di `FINAL_EVALUATION`. Grafik loss, separation, margin,
pseudo-accuracy, correlation consistency, active dimensions, dan metrik validation
tambahan harus memakai sumbu-X **Objective Evaluation (`eval_id`)**, bukan
"Iteration".

---

## 6. EKSTRAKSI LABEL KUANTUM
 
```
FUNCTION QUANTUM_LABEL_EXTRACTION(circuit_fn, trained_params_clustering, X_train, y_train, config):
 
    quantum_labels = {}
    FOR each c IN config.classes:
        outputs_c = [circuit_fn(x, trained_params_clustering) FOR x IN X_train WHERE y_train == c]
        centroid = MEAN(outputs_c)
        quantum_labels[c] = NORMALIZE_VECTOR(centroid)
 
    SAVE(quantum_labels, run_dir + "/artifacts/quantum_labels.json")
    RETURN quantum_labels
```
 
Fungsi ini dipakai identik oleh Jalur A (otomatis, bagian 9) maupun
Jalur B (pemilihan manual, bagian 10) — parameter `trained_params_clustering`
yang diterima bisa berasal dari `result.x` COBYLA (Jalur A) atau dari
`READ_PARAM_RECORD` pada iterasi pilihan manual (Jalur B); fungsi ini
sendiri tidak tahu dan tidak perlu tahu sumbernya.
 
---
 
## 7. SUPERVISED LOOP
 
Berbeda dari clustering, objective supervised adalah jarak keluaran ke quantum
label. `val_loss` tetap hanya monitoring pasif. G1-02 menerapkan aturan yang sama:
setiap pemanggilan `objective_supervised(theta)` adalah satu objective-function
evaluation dengan `eval_id`, sedangkan callback optimizer memakai
`callback_id` terpisah.
 
```
FUNCTION SUPERVISED_LOOP(circuit_fn, trained_params_clustering, quantum_labels,
                         X_train, y_train, X_val, y_val, config):

    n_objective_evals = 0
    callback_id = 0
    best_observed_fun = +INFINITY
    best_observed_eval_id = NULL
    best_observed_theta = NULL

    n_params = LENGTH(trained_params_clustering)   # trainable vector yang dioptimasi COBYLA
    VALIDATE_OPTIMIZER_BUDGET(config.max_nfev_supervised, n_params, config.run_mode)

    CREATE_EMPTY_BINARY_LOG(
        run_dir + "/artifacts/supervised_params.bin",
        record_size = SIZEOF(trained_params_clustering)
    )
    CREATE_EMPTY_FILE(run_dir + "/logs/supervised_log.jsonl")
    CREATE_EMPTY_FILE(run_dir + "/logs/supervised_callback_log.jsonl")
 
    FUNCTION objective_supervised(theta):
        NONLOCAL n_objective_evals
        NONLOCAL best_observed_fun, best_observed_eval_id, best_observed_theta

        eval_id = n_objective_evals
 
        train_losses = []
        FOR each (x, c) IN ZIP(X_train, y_train):
            v = circuit_fn(x, theta)
            train_losses.APPEND(COSINE_DISTANCE(v, quantum_labels[c]))
        train_loss = MEAN(train_losses)
 
        val_losses = []
        FOR each (x, c) IN ZIP(X_val, y_val):
            v = circuit_fn(x, theta)
            val_losses.APPEND(COSINE_DISTANCE(v, quantum_labels[c]))
        val_loss = MEAN(val_losses)
 
        APPEND_PARAM_RECORD(run_dir + "/artifacts/supervised_params.bin", theta)

        IF train_loss < best_observed_fun:
            best_observed_fun = train_loss
            best_observed_eval_id = eval_id
            best_observed_theta = COPY(theta)
 
        log_entry = {
            "eval_id": eval_id,
            "phase": OBJECTIVE_PHASE(eval_id, n_params),   # G1-06
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        APPEND_LINE(run_dir + "/logs/supervised_log.jsonl", TO_JSON(log_entry))
 
        n_objective_evals = n_objective_evals + 1
        RETURN train_loss

    FUNCTION supervised_callback(callback_state):
        NONLOCAL callback_id
        theta_callback = EXTRACT_THETA_FROM_COBYLA_CALLBACK(callback_state)

        callback_entry = {
            "callback_id": callback_id,
            "objective_evals_seen": n_objective_evals,
            "theta_source": "optimizer_callback"
        }
        APPEND_LINE(
            run_dir + "/logs/supervised_callback_log.jsonl",
            TO_JSON(callback_entry)
        )
        callback_id = callback_id + 1
 
    result = COBYLA_MINIMIZE(
        objective_supervised,
        x0=trained_params_clustering,
        callback=supervised_callback,
        options={
            "maxiter": config.max_nfev_supervised,
            "tol": config.cobyla_tol,
            "rhobeg": config.cobyla_rhobeg
        }
    )

    trained_params_final = result.x
    ASSERT result.nfev == n_objective_evals

    SAVE(result.x, run_dir + "/artifacts/supervised_params_final.npy")
    SAVE(best_observed_theta, run_dir + "/artifacts/supervised_params_best_observed.npy")

    optimizer_summary = {
        "optimizer": "COBYLA",
        "max_nfev": config.max_nfev_supervised,
        "tol": config.cobyla_tol,
        "rhobeg": config.cobyla_rhobeg,
        "scipy_version": SCIPY_VERSION(),
        "n_params": n_params,
        "simplex_size": n_params + 1,
        "n_initial_simplex_evals": MIN(result.nfev, n_params + 1),
        "n_optimization_evals": MAX(0, result.nfev - (n_params + 1)),
        "success": result.success,
        "status": result.status,
        "message": STRING(result.message),
        "fun": result.fun,
        "nfev": result.nfev,
        "callback_count": callback_id,
        "final_point_path": "artifacts/supervised_params_final.npy",
        "best_observed_eval_id": best_observed_eval_id,
        "best_observed_fun": best_observed_fun,
        "best_observed_point_path": "artifacts/supervised_params_best_observed.npy"
    }
    SAVE_JSON(
        optimizer_summary,
        run_dir + "/logs/supervised_optimizer_result.json"
    )
 
    RETURN trained_params_final
```

`supervised_log.jsonl` harus diplot terhadap `eval_id`. `result.x` adalah
**final point yang dikembalikan COBYLA**; ia tidak boleh otomatis disebut
"best observed point". Titik terbaik yang benar-benar terlihat selama evaluasi
disimpan terpisah sebagai `supervised_params_best_observed.npy`.
 
---
 
## 8. FINAL EVALUATION

**G0-01 (sesi 2026-09-22):** ini SATU-SATUNYA fungsi dalam seluruh pipeline
yang boleh memanggil `circuit_fn` terhadap `X_test`/`y_test`. Baik Jalur A
maupun Jalur B memanggil fungsi ini persis sekali, di akhir, setelah
`SUPERVISED_LOOP` selesai — sesuai dengan definisi "protokol dibekukan"
pada kriteria penerimaan G0-01.
 
```
FUNCTION FINAL_EVALUATION(circuit_fn, trained_params_final, quantum_labels, X_test, y_test, config):
 
    y_pred = []
    FOR each x IN X_test:
        v = circuit_fn(x, trained_params_final)
        distances = { c: COSINE_DISTANCE(v, quantum_labels[c]) FOR c IN config.classes }
        y_pred.APPEND(ARGMIN(distances))
 
    accuracy         = ACCURACY_SCORE(y_test, y_pred)
    precision        = PRECISION_SCORE(y_test, y_pred, average="macro")
    recall           = RECALL_SCORE(y_test, y_pred, average="macro")
    f1               = F1_SCORE(y_test, y_pred, average="macro")
    confusion_matrix = CONFUSION_MATRIX(y_test, y_pred)
 
    metrics = {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1
    }
 
    SAVE(metrics,          run_dir + "/logs/metrics_final.json")
    SAVE(confusion_matrix, run_dir + "/logs/confusion_matrix.npy")
 
    RETURN metrics, confusion_matrix
```
 
---
 
## 9. JALUR A — MAIN (Satu Run, Eksekusi Manual per Kondisi)

### 9.1 MAIN(config) — menjalankan satu kombinasi eksperimen

`MAIN(config)` adalah unit eksperimen utama dan selalu menjalankan tepat **satu kondisi**.
Input budget optimasi dinyatakan sebagai maksimum objective-function evaluations:

- `classes`
- `n_train_per_class`
- `n_val_per_class`
- `n_test_per_class`
- `max_nfev_clustering`
- `max_nfev_supervised`
- `architecture`
- `feature_method`
- `seed`

```
FUNCTION MAIN(config):
    VALIDATE_CONFIG(config)
    CREATE_RUN_DIRECTORY_EXCLUSIVE(config)
    INITIALIZE_LOCAL_RUN_SPREADSHEET(config)

    VALIDATE_SEED_PROTOCOL(config)
    config.split_manifest_path = RESOLVE_SPLIT_MANIFEST_PATH(config)
    IF config.run_mode == "CONFIRMATORY":
        CREATE_OR_LOAD_SPLIT_MANIFEST(config.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = DATA_PIPELINE(config)
    S = CORRELATION_MATRIX(X_train, y_train, config.classes)
    circuit_fn, initial_params = MODEL_SETUP(config)

    params_after_clustering = CLUSTERING_LOOP(
        circuit_fn, initial_params, X_train, y_train, X_val, y_val, S, config
    )

    quantum_labels = QUANTUM_LABEL_EXTRACTION(
        circuit_fn, params_after_clustering, X_train, y_train, config
    )

    params_final = SUPERVISED_LOOP(
        circuit_fn, params_after_clustering, quantum_labels,
        X_train, y_train, X_val, y_val, config
    )

    metrics, confusion_matrix = FINAL_EVALUATION(
        circuit_fn, params_final, quantum_labels, X_test, y_test, config
    )

    SAVE_ARTIFACT_BUNDLE(config, metrics)

    UPDATE_LOCAL_RUN_SPREADSHEET(
        local_spreadsheet_path = config.local_spreadsheet_path,
        run_id                 = config.run_id,
        run_dir                = config.run_dir
    )

    PRINT("Jalur A selesai. Hasil ada di: " + config.run_dir)


FUNCTION SAVE_ARTIFACT_BUNDLE(config, metrics):

    artifact_paths = {
        "feature_metadata": "artifacts/feature_metadata.json",
        "scaler_params": "artifacts/scaler_params.joblib",
        "X_train_scaled": "artifacts/X_train_scaled.npy",
        "y_train": "artifacts/y_train.npy",
        "X_val_scaled": "artifacts/X_val_scaled.npy",
        "y_val": "artifacts/y_val.npy",
        "X_test_scaled": "artifacts/X_test_scaled.npy",
        "y_test": "artifacts/y_test.npy",
        "correlation_matrix": "artifacts/correlation_matrix.npy",
        "initial_params": "artifacts/initial_params.npy",
        "initialization_manifest": "artifacts/initialization_manifest.json",

        "clustering_params_bin": "artifacts/clustering_params.bin",
        "clustering_params_final": "artifacts/clustering_params_final.npy",
        "clustering_params_best_observed": "artifacts/clustering_params_best_observed.npy",
        "clustering_log": "logs/clustering_log.jsonl",
        "clustering_callback_log": "logs/clustering_callback_log.jsonl",
        "clustering_optimizer_result": "logs/clustering_optimizer_result.json",

        "quantum_labels": "artifacts/quantum_labels.json",

        "supervised_params_bin": "artifacts/supervised_params.bin",
        "supervised_params_final": "artifacts/supervised_params_final.npy",
        "supervised_params_best_observed": "artifacts/supervised_params_best_observed.npy",
        "supervised_log": "logs/supervised_log.jsonl",
        "supervised_callback_log": "logs/supervised_callback_log.jsonl",
        "supervised_optimizer_result": "logs/supervised_optimizer_result.json",

        "metrics_final": "logs/metrics_final.json",
        "confusion_matrix": "logs/confusion_matrix.npy",
        "run_spreadsheet": "run_result.xlsx"
    }

    IF config.feature_method == "PCA":
        artifact_paths["pca_model"] = "artifacts/pca_model.joblib"

    clustering_optimizer_result = LOAD_JSON(
        config.run_dir + "/logs/clustering_optimizer_result.json"
    )
    supervised_optimizer_result = LOAD_JSON(
        config.run_dir + "/logs/supervised_optimizer_result.json"
    )

    manifest = {
        "run_id": config.run_id,
        "run_name": config.run_name,
        "path_type": "jalur_a_automatic",
        "timestamp": NOW(),

        "classes": config.classes,
        "n_classes": LENGTH(config.classes),
        "n_train_per_class": config.n_train_per_class,
        "n_val_per_class": config.n_val_per_class,
        "n_test_per_class": config.n_test_per_class,
        "max_nfev_clustering": config.max_nfev_clustering,
        "max_nfev_supervised": config.max_nfev_supervised,

        "architecture": config.architecture,
        "feature_method": config.feature_method,
        "n_data_qubits": config.n_data_qubits,
        "n_readout_qubits": config.n_readout_qubits,
        "n_input_channels": config.n_input_channels,

        "pca_n_components": config.pca_n_components IF config.feature_method == "PCA" ELSE NULL,
        "pca_svd_solver": config.pca_svd_solver IF config.feature_method == "PCA" ELSE NULL,
        "pca_random_state": config.pca_random_state IF config.feature_method == "PCA" ELSE NULL,
        "hu_raw_dim": config.hu_raw_dim IF config.feature_method == "HU" ELSE NULL,
        "hu_input_mode": config.hu_input_mode IF config.feature_method == "HU" ELSE NULL,
        "hu_padding_value": config.hu_padding_value IF config.feature_method == "HU" ELSE NULL,
        "zernike_terms": config.zernike_terms IF config.feature_method == "ZERNIKE" ELSE NULL,

        "cobyla_tol": config.cobyla_tol,
        "cobyla_rhobeg": config.cobyla_rhobeg,
        "seed": config.seed,
        "master_seed": config.seed,
        "run_mode": config.run_mode,
        "data_seed": DERIVE_SUBSEED(config.seed, "data"),
        "ry_core_seed": DERIVE_SUBSEED(config.seed, "init:ry_core"),
        "rz_phase_seed": DERIVE_SUBSEED(config.seed, "init:rz_phase"),
        "pair_seed": DERIVE_SUBSEED(config.seed, "cluster_pairs"),
        "label_seed": DERIVE_SUBSEED(config.seed, "quantum_labels"),
        "split_manifest": config.split_manifest_path,
        "n_cluster_pair_samples": config.n_cluster_pair_samples,
        "active_dim_threshold": config.active_dim_threshold,
        "mnist_root": config.mnist_root,
        "mnist_download": config.mnist_download,
        "local_spreadsheet_path": config.local_spreadsheet_path,

        "optimizer_results": {
            "clustering": clustering_optimizer_result,
            "supervised": supervised_optimizer_result
        },

        "final_metrics": metrics,
        "artifact_paths": artifact_paths
    }

    SAVE(manifest, config.run_dir + "/config.json")
```

### 9.2 Contoh satu run berdasarkan input user

Contoh berikut adalah kondisi pertama: klasifikasi 3 kelas (`0,1,2`) menggunakan
PCA dan arsitektur MORE-HD, memakai protokol final G0-03 (`n_train=1000`,
`n_val=100`, `n_test=200`). Hanya kondisi ini yang dijalankan ketika `MAIN(config)`
dipanggil.

```
config_user = Config(
    classes=[0,1,2],
    n_train_per_class=1000,
    n_val_per_class=100,
    n_test_per_class=200,
    max_nfev_clustering=MAX_NFEV_FINAL_G1_01,   # diisi dari hasil pilot §1.1; 10 hanya untuk smoke test PILOT
    max_nfev_supervised=MAX_NFEV_FINAL_G1_01,   # VALIDATE_OPTIMIZER_BUDGET menolak <= n_params+1 pada CONFIRMATORY
    architecture="MORE-HD",
    feature_method="PCA",
    seed=101,
    run_mode="CONFIRMATORY",
    mnist_download=FALSE      # dataset sudah tersedia setelah pilot/initial download
)

MAIN(config_user)
```

Run tersebut akan mempunyai identitas unik, misalnya:

```
cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed101
```

Untuk kondisi ini: `condition_id=R001`, `run_uid=R001-S101`, dan split dibaca dari `splits/seed101.json`.

Pilot seed 42 **tidak boleh** dipromosikan menjadi run konfirmatori. Setelah pipeline, budget optimizer, dan protokol dibekukan pada fase pilot, run publikasi dimulai ulang menggunakan seed konfirmatori yang sudah dipra-tetapkan.

```
mnist_download = FALSE
```

Dengan demikian proses paralel hanya membaca dataset lokal dari `data/` dan tidak mencoba melakukan download bersamaan.

Setelah run selesai dan hasilnya diperiksa, pengguna membuat konfigurasi berikutnya
secara manual, misalnya `PCA + MORE-HD-C`, lalu `HU + MORE-HD`, dan seterusnya.
Pola yang sama dilakukan untuk skenario kelas 3 sampai 10. Dengan demikian desain
penelitian tetap terdiri dari 48 kondisi per seed. Lima seed konfirmatori menghasilkan total 240 run, tetapi eksekusinya tetap bersifat
**controlled manual per run** agar setiap proses dapat dimonitor secara individual.

Contoh urutan enam kondisi untuk skenario 3 kelas adalah:

| Feature representation | MORE-HD | MORE-HD-C |
|---|---|---|
| PCA | satu run terpisah | satu run terpisah |
| HU | satu run terpisah | satu run terpisah |
| ZERNIKE | satu run terpisah | satu run terpisah |

Setelah keenam kondisi untuk 3 kelas selesai, pengguna melanjutkan secara manual ke
4 kelas, kemudian 5 kelas, sampai 10 kelas. Tidak ada fungsi batch yang menjalankan
seluruh 48 kondisi secara otomatis.

---

## 9.3 KEBIJAKAN EKSEKUSI PARALEL UNTUK JALUR A

Parallel execution dilakukan pada level **run**, bukan di dalam satu run. Setiap proses menjalankan `MAIN(config)` dengan konfigurasi yang berbeda dan menulis hanya ke `run_dir` miliknya sendiri.

```
Process A -> runs/<run_A>/...
Process B -> runs/<run_B>/...
Process C -> runs/<run_C>/...
...
```

Aturan yang dikunci untuk tahap eksperimen saat ini:

1. Program tidak menentukan otomatis berapa banyak run yang boleh aktif bersamaan. Jumlah proses paralel ditentukan manual oleh pengguna setelah memantau CPU dan RAM pada mesin yang digunakan.
2. Tidak ada auto-throttling CPU/RAM dan belum ada pembatasan thread internal NumPy/SciPy/PennyLane pada revisi ini.
3. Overhead I/O storage dari beberapa run yang menulis log secara bersamaan belum diberi penanganan khusus dan untuk sementara diterima sebagai bagian dari eksperimen.
4. `CREATE_RUN_DIRECTORY_EXCLUSIVE(config)` wajib dieksekusi sebelum training. Run dengan `run_dir` yang sudah ada langsung dibatalkan; program tidak melakukan overwrite atau resume otomatis.
5. Master spreadsheet tidak pernah menjadi shared writable resource selama training. Setiap run hanya menulis `run_result.xlsx` lokal miliknya.
6. Pengelolaan parallel collision khusus Jalur B belum menjadi fokus revisi ini dan tetap mengikuti desain sebelumnya.

Dengan aturan ini, sumber daya yang dibagi antar proses hanya berupa resource read-only seperti source code dan dataset MNIST lokal. Semua output yang mutable tetap dipisahkan per `run_dir`.

---

## 10. JALUR B — Deterministic Validation-Selected Clustering Checkpoint
 
### 10.1 Posisi Jalur B terhadap Jalur A
 
Jalur A tetap merupakan **primary execution path**. Setelah `CLUSTERING_LOOP`
selesai, Jalur A menggunakan `result.x` — final point resmi yang dikembalikan
COBYLA — untuk `QUANTUM_LABEL_EXTRACTION`, `SUPERVISED_LOOP`, dan
`FINAL_EVALUATION`. Keputusan G0-02 **tidak mengganti** `result.x` pada
Jalur A.

Jalur B adalah **secondary/sensitivity path** yang dijalankan dari artefak run
Jalur A yang sudah selesai. Jalur B tidak menjalankan ulang `DATA_PIPELINE`
atau `CLUSTERING_LOOP`. Tujuannya adalah menguji apakah trajectory clustering
mengandung objective evaluation yang, menurut aturan validation yang
dipra-tetapkan, menghasilkan titik awal supervised yang berbeda dari final
`result.x`.

Jalur B tidak termasuk dalam hitungan **330 unique confirmatory executions**
yang telah dibekukan (240 primary A/D + 90 targeted ablation B/C). Jika Jalur B
nanti dieksekusi pada subset atau seluruh run, eksekusi tersebut dicatat sebagai
secondary analysis tambahan dan tidak mengganti hasil primer Jalur A.

### 10.2 Aturan G0-02: selector otomatis dan deterministik

Tidak ada lagi input manual `selected_eval_id`. Fungsi selector membaca
`clustering_log.jsonl` dan memilih satu `eval_id` secara lexicographic.
Urutan kriteria dibekukan sebagai berikut:

1. maksimum `pseudo_accuracy_val`;
2. jika tie, maksimum `min_separation_val`;
3. jika tie, minimum `mean_true_distance_val`;
4. jika tie, minimum `train_loss`;
5. jika masih tie, `eval_id` paling awal.

Tidak digunakan weighted score. `active_dimensions`,
`correlation_consistency`, dan `avg_margin_val` tetap disimpan untuk analisis,
tetapi **tidak boleh** memengaruhi pemilihan checkpoint. Pengecualian
`active_dimensions` penting agar selector tidak secara struktural
menguntungkan MORE-HD-C hanya karena arsitektur tersebut memang dirancang untuk
mengaktifkan dimensi Y-odd.

Domain objective evaluation yang boleh menjadi kandidat ditentukan oleh
`ELIGIBLE_CLUSTERING_EVAL_IDS`. **Dikunci G1-06 (2026-09-27):** hanya baris
dengan `phase == "optimization"`; seluruh `initial_simplex` termasuk `x0`
dikecualikan. Jika domain kosong (budget tidak melewati simplex), selector
menolak run tersebut.

```
FUNCTION ELIGIBLE_CLUSTERING_EVAL_IDS(log_rows, source_clustering_result):
    n_params = source_clustering_result.n_params
    eligible = []
    FOR each row IN log_rows:
        ASSERT row.phase == OBJECTIVE_PHASE(row.eval_id, n_params)
        IF row.phase == "optimization":
            eligible.APPEND(row.eval_id)
    RETURN eligible
```

Catatan asimetri: di bawah aturan total-`nfev` sama, jumlah kandidat Jalur B untuk
MORE-HD-C 30 lebih sedikit daripada MORE-HD. Jumlah kandidat per run dicatat
di `checkpoint_selection_log.json` (`len(eligible_eval_ids)`).

`max_nfev_supervised` Jalur B **diwarisi dari run sumber**, bukan input manual.
Dengan demikian satu-satunya perbedaan yang disengaja antara Jalur A dan Jalur B
adalah sumber parameter clustering awal untuk supervised:

```text
Jalur A : theta_clustering = result.x COBYLA
Jalur B : theta_clustering = checkpoint terpilih deterministic selector
```

### 10.3 Pseudocode selector

```
FUNCTION SELECT_CLUSTERING_CHECKPOINT_DETERMINISTIC(source_run_dir):
    source_clustering_result = LOAD_JSON(
        source_run_dir + "/logs/clustering_optimizer_result.json"
    )
    log_rows = LOAD_JSONL(
        source_run_dir + "/logs/clustering_log.jsonl"
    )

    ASSERT LENGTH(log_rows) == source_clustering_result.nfev

    eligible_eval_ids = ELIGIBLE_CLUSTERING_EVAL_IDS(
        log_rows,
        source_clustering_result
    )

    IF LENGTH(eligible_eval_ids) == 0:
        RAISE_ERROR("Tidak ada objective evaluation eligible untuk Jalur B")

    candidates = [
        row FOR row IN log_rows
        WHERE row.eval_id IN eligible_eval_ids
    ]

    SORT candidates BY:
        pseudo_accuracy_val     DESCENDING,
        min_separation_val      DESCENDING,
        mean_true_distance_val  ASCENDING,
        train_loss              ASCENDING,
        eval_id                 ASCENDING

    selected = candidates[0]

    selection_record = {
        "selection_method": "deterministic_lexicographic_validation",
        "criteria_order": [
            "pseudo_accuracy_val DESC",
            "min_separation_val DESC",
            "mean_true_distance_val ASC",
            "train_loss ASC",
            "eval_id ASC"
        ],
        "selected_eval_id": selected.eval_id,
        "selected_metrics": selected,
        "eligible_eval_ids": eligible_eval_ids,
        "test_used_for_selection": FALSE,
        "diagnostics_excluded_from_selection": [
            "active_dimensions",
            "correlation_consistency",
            "avg_margin_val"
        ]
    }

    SAVE_JSON(
        selection_record,
        source_run_dir + "/logs/selected_checkpoint.json"
    )

    SAVE_JSON(
        {
            "candidates": candidates,
            "selection_method": selection_record.selection_method,
            "criteria_order": selection_record.criteria_order
        },
        source_run_dir + "/logs/checkpoint_selection_log.json"
    )

    RETURN selected.eval_id
```

Untuk input artefak dan aturan eligibility yang identik, fungsi ini wajib
menghasilkan `selected_eval_id` yang sama. Unit test G0-02 harus menjalankan
selector minimal dua kali pada fixture yang sama dan membuktikan hasil serta
urutan kandidat identik.

### 10.4 Pseudocode Jalur B

```
FUNCTION MAIN_FROM_SELECTED_CLUSTERING(source_run_dir):
 
    source_manifest = LOAD_JSON(source_run_dir + "/config.json")
    ASSERT source_manifest.path_type == "jalur_a_automatic"
    ASSERT EXISTS(source_run_dir + "/artifacts/clustering_params.bin")
    ASSERT EXISTS(source_run_dir + "/logs/clustering_log.jsonl")
    ASSERT EXISTS(source_run_dir + "/logs/clustering_optimizer_result.json")

    config = Config(
        classes                = source_manifest.classes,
        n_train_per_class      = source_manifest.n_train_per_class,
        n_val_per_class        = source_manifest.n_val_per_class,
        n_test_per_class       = source_manifest.n_test_per_class,
        max_nfev_clustering    = source_manifest.max_nfev_clustering,
        max_nfev_supervised    = source_manifest.max_nfev_supervised,
        architecture           = source_manifest.architecture,
        feature_method         = source_manifest.feature_method,
        n_data_qubits          = source_manifest.n_data_qubits,
        n_readout_qubits       = source_manifest.n_readout_qubits,
        n_input_channels       = source_manifest.n_input_channels,
        cobyla_tol             = source_manifest.cobyla_tol,
        cobyla_rhobeg          = source_manifest.cobyla_rhobeg,
        seed                   = source_manifest.seed,
        run_mode               = source_manifest.run_mode,
        n_cluster_pair_samples = source_manifest.n_cluster_pair_samples
    )

    selected_eval_id = SELECT_CLUSTERING_CHECKPOINT_DETERMINISTIC(
        source_run_dir
    )

    config.run_name = (
        source_manifest.run_name +
        "_jalurB_auto_eval" +
        ZERO_PAD(selected_eval_id, 4)
    )
    config.run_dir = "runs/" + config.run_name

    CREATE_RUN_DIRECTORY_EXCLUSIVE(config)
    CREATE_DIRECTORY(config.run_dir + "/artifacts")
    CREATE_DIRECTORY(config.run_dir + "/logs")

    X_train_scaled = LOAD(source_run_dir + "/artifacts/X_train_scaled.npy")
    y_train        = LOAD(source_run_dir + "/artifacts/y_train.npy")
    X_val_scaled   = LOAD(source_run_dir + "/artifacts/X_val_scaled.npy")
    y_val          = LOAD(source_run_dir + "/artifacts/y_val.npy")

    # Official test SENGAJA belum dimuat di tahap ini.

    IF config.architecture == "MORE-HD":
        circuit_fn, _ = BUILD_CIRCUIT_MORE_HD(
            n_data_qubits=config.n_data_qubits,
            n_readout_qubits=config.n_readout_qubits
        )
    ELSE IF config.architecture == "MORE-HD-C":
        circuit_fn, _ = BUILD_CIRCUIT_MORE_HD_C(
            n_data_qubits=config.n_data_qubits,
            n_readout_qubits=config.n_readout_qubits
        )
    ELSE:
        RAISE_ERROR("architecture tidak dikenali di manifest run sumber")

    record_size = GET_RECORD_SIZE(
        source_run_dir + "/artifacts/clustering_params.bin"
    )
    selected_theta = READ_PARAM_RECORD(
        source_run_dir + "/artifacts/clustering_params.bin",
        eval_id = selected_eval_id,
        record_size = record_size
    )

    quantum_labels = QUANTUM_LABEL_EXTRACTION(
        circuit_fn, selected_theta, X_train_scaled, y_train, config
    )

    trained_params_final = SUPERVISED_LOOP(
        circuit_fn, selected_theta, quantum_labels,
        X_train_scaled, y_train, X_val_scaled, y_val, config
    )

    X_test_scaled = LOAD(source_run_dir + "/artifacts/X_test_scaled.npy")
    y_test        = LOAD(source_run_dir + "/artifacts/y_test.npy")

    metrics, confusion_matrix = FINAL_EVALUATION(
        circuit_fn, trained_params_final, quantum_labels,
        X_test_scaled, y_test, config
    )

    SAVE_ARTIFACT_BUNDLE_JALUR_B(
        config, metrics, source_run_dir, selected_eval_id
    )

    RETURN metrics, confusion_matrix


FUNCTION SAVE_ARTIFACT_BUNDLE_JALUR_B(
    config, metrics, source_run_dir, selected_eval_id
):
    selection_record = LOAD_JSON(
        source_run_dir + "/logs/selected_checkpoint.json"
    )
    supervised_optimizer_result = LOAD_JSON(
        config.run_dir + "/logs/supervised_optimizer_result.json"
    )

    manifest = {
        "run_name": config.run_name,
        "path_type": "jalur_b_deterministic_validation_selection",
        "analysis_role": "secondary_sensitivity",
        "timestamp": NOW(),
        "source_run_dir": source_run_dir,
        "source_primary_theta": "result.x",
        "selected_eval_id": selected_eval_id,
        "selection_method": selection_record.selection_method,
        "selection_criteria_order": selection_record.criteria_order,
        "test_used_for_selection": FALSE,
        "inherits_supervised_budget_from_source": TRUE,
        "max_nfev_supervised": config.max_nfev_supervised,
        "architecture": config.architecture,
        "feature_method": config.feature_method,
        "classes": config.classes,
        "seed": config.seed,
        "run_mode": config.run_mode,
        "supervised_optimizer_result": supervised_optimizer_result,
        "final_metrics": metrics,
        "artifact_paths": {
            "quantum_labels": "artifacts/quantum_labels.json",
            "supervised_params_bin": "artifacts/supervised_params.bin",
            "supervised_params_final": "artifacts/supervised_params_final.npy",
            "supervised_params_best_observed": "artifacts/supervised_params_best_observed.npy",
            "supervised_log": "logs/supervised_log.jsonl",
            "supervised_callback_log": "logs/supervised_callback_log.jsonl",
            "supervised_optimizer_result": "logs/supervised_optimizer_result.json",
            "metrics_final": "logs/metrics_final.json",
            "confusion_matrix": "logs/confusion_matrix.npy",
            "source_selected_checkpoint": source_run_dir + "/logs/selected_checkpoint.json",
            "source_checkpoint_selection_log": source_run_dir + "/logs/checkpoint_selection_log.json"
        }
    }

    SAVE(manifest, config.run_dir + "/config.json")


# Contoh:
MAIN_FROM_SELECTED_CLUSTERING(
    source_run_dir = "runs/R001-S101_..."
)
```
 
### 10.5 Ringkasan Perbedaan Jalur A vs Jalur B
 
| Aspek | Jalur A — primary | Jalur B — secondary sensitivity |
|---|---|---|
| Sumber `theta` clustering | `result.x` = final point resmi COBYLA | `READ_PARAM_RECORD` pada `selected_eval_id` hasil selector deterministik |
| Pemilihan checkpoint clustering | Tidak ada; memakai final optimizer point | Lexicographic train/validation rule yang dibekukan |
| Input manual checkpoint | Tidak ada | **Tidak ada** |
| Budget supervised | `max_nfev_supervised` dari config | Diwarisi identik dari source Jalur A |
| DATA_PIPELINE/CLUSTERING dijalankan ulang | Ya, sebagai run normal | Tidak |
| Akses official test sebelum keputusan model selesai | Tidak | Tidak |
| `active_dimensions` dipakai memilih checkpoint | Tidak | **Tidak; diagnostik saja** |
| Nama run | normal | suffix `_jalurB_auto_evalXXXX` |
| Peran dalam 330 execution plan | Termasuk primary 240 A/D | **Tidak termasuk**; secondary analysis tambahan |
| Provenance | manifest Jalur A | `source_run_dir` + `selected_eval_id` + selection log deterministik |

---

## 11. TARGETED ABLATION G1-05/G1-07 — IMPLEMENTASI TERPADU A/B/C/D

Bagian ini adalah **source of truth teknis** untuk targeted ablation. Isi yang sebelumnya berada pada dokumen terpisah dipindahkan ke sini agar spesifikasi implementasi hanya memiliki satu sumber acuan.

#### 11.11.1 Purpose

The primary 2 × 3 × 8 experiment remains unchanged and compares MORE-HD against MORE-HD-C across K=3..10, PCA/HU/ZERNIKE, and five confirmatory seeds. The targeted ablation defined here is an additional confirmatory experiment designed to separate four effects that are confounded in the direct MORE-HD versus MORE-HD-C comparison: (1) additional real-valued trainable capacity, (2) access to a complex-valued readout state, (3) optimization of the complex phase, and (4) the combined full MORE-HD-C modification.

The ablation is intentionally restricted to representative class-count regimes K ∈ {3, 6, 10}. It uses all three feature representations and the same five confirmatory seeds used by the primary experiment: 101, 202, 303, 404, and 505. Seed 42 remains PILOT ONLY.

#### 11.11.2 Four-model ablation family

| Code | Model | State constraint | Variational structure | Trainable parameters | Fixed phase parameters | Role |
|---|---|---|---|---:|---:|---|
| A | MORE-HD | real | 3 × RY-only layers | 30 | 0 | primary baseline; reused from the 240-run main experiment |
| B | MORE-HD-60P | real | 6 × RY-only layers | 60 | 0 | parameter-budget / real-capacity control |
| C | MORE-HD-C-FixedRZ | complex allowed | 3 × (RY + fixed RZ) layers | 30 RY | 30 RZ | complex-state access at the same trainable-parameter count as A |
| D | MORE-HD-C | complex allowed | 3 × (RY + trainable RZ) layers | 60 | 0 | primary proposed model; reused from the 240-run main experiment |

Models A and D are not rerun for the ablation. Their matching main-experiment runs are referenced by `run_uid` and reused. Only B and C generate new executions.

#### 11.11.3 Ablation matrix and run count

The targeted conditions are:

```text
K = {3, 6, 10}
feature_method = {PCA, HU, ZERNIKE}
new ablation variants = {B, C}
confirmatory seeds = {101, 202, 303, 404, 505}
```

Therefore:

```text
3 K levels × 3 feature methods × 2 new variants = 18 ablation conditions per seed
18 conditions × 5 confirmatory seeds = 90 additional confirmatory runs
```

The complete confirmatory workload becomes:

```text
240 primary runs + 90 additional ablation runs = 330 unique confirmatory runs
```

The four-model analysis still contains A/B/C/D for each ablation cell, but A and D are linked to existing primary runs and are not counted again.

#### 11.11.4 Pairing and data-control rules

For a fixed `(seed, K, feature_method)` cell, A, B, C, and D must use the same raw train/validation/official-test split manifest. The same preprocessing fit policy is retained: all learned transformations and scalers are fit only on the training partition.

The clustering-pair protocol is also paired. The selected five training instances per class and the resulting unordered unique pair manifest must be identical across A/B/C/D for the same `(seed, K)` cell. B and C therefore reuse the same deterministic pair-selection namespace and sample identities used by A/D.

No official-test information may be used to choose the architecture, initialization, optimizer checkpoint, ablation subset, or any other design decision.

#### 11.11.5 Deterministic paired initialization

Paired initialization is shared with the primary A/D runs and derived from the confirmatory master seed using independent deterministic namespaces. This is required so that A and D can be reused from the primary matrix without breaking the ablation pairing.

```text
ry_core_seed  = DERIVE_SUBSEED(master_seed, "init:ry_core")
ry_extra_seed = DERIVE_SUBSEED(master_seed, "ablation:init:ry_extra")
rz_phase_seed = DERIVE_SUBSEED(master_seed, "init:rz_phase")
```

The common 30-element `theta_RY_core` is used to initialize the corresponding RY parameters in A, C, and D. Primary `MODEL_SETUP` must therefore build A and D from this same initialization bundle before any confirmatory run is executed. Model B uses the same 30-element core initialization for its first three RY layers and receives 30 additional RY parameters from `ry_extra_seed` for layers four through six.

Models C and D share the same 30-element initial RZ vector produced by `rz_phase_seed`. Every RZ angle must be non-zero. If deterministic generation produces an exact zero due to numerical representation, that element is deterministically redrawn from the same RNG stream until it is non-zero.

For model C, the RZ vector is frozen for both clustering and supervised optimization. Only the 30 RY parameters are optimized. For model D, the same initial RZ vector is trainable. This gives C-versus-D a common phase starting point.

#### 11.11.6 Circuit builders

#### 11.6.1 Model B — MORE-HD-60P

```text
FUNCTION BUILD_CIRCUIT_MORE_HD_60P(
    n_data_qubits=8,
    n_readout_qubits=2,
    n_layers=6
):
    ASSERT n_data_qubits == 8
    ASSERT n_readout_qubits == 2

    n_qubits = 10
    n_trainable = 6 * 10 = 60

    FUNCTION circuit_fn(x, theta):
        theta_layers = RESHAPE(theta, (6, 10))

        FOR i IN range(8):
            APPLY_GATE(RY(x[i]), wire=i)

        FOR layer_idx IN range(6):
            RY_ONLY_LAYER(theta_layers[layer_idx], n_qubits)
            ENTANGLING_BLOCK_CNOT(8, 2)

        RETURN MEASURE_15_OBSERVABLES(readout_wires=[8, 9])

    RETURN circuit_fn, n_trainable
```

Model B remains in the real-state family because it uses only RY rotations and CNOT gates. It is a parameter-budget control, but it is not a perfect depth-matched control: increasing from three to six RY layers also increases variational depth and the number of repeated entangling blocks. Therefore B-versus-D must be described as an equal-trainable-parameter comparison, not as a claim that all other circuit-resource dimensions are identical. Gate counts, variational depth, and runtime are recorded as secondary diagnostics.

#### 11.6.2 Model C — MORE-HD-C-FixedRZ

```text
FUNCTION BUILD_CIRCUIT_MORE_HD_C_FIXED_RZ(
    fixed_rz,
    n_data_qubits=8,
    n_readout_qubits=2,
    n_layers=3
):
    ASSERT LENGTH(fixed_rz) == 30
    ASSERT ALL(angle != 0.0 FOR angle IN fixed_rz)

    n_qubits = 10
    n_trainable = 3 * 10 = 30

    FUNCTION circuit_fn(x, theta_ry):
        ry_layers = RESHAPE(theta_ry, (3, 10))
        rz_layers = RESHAPE(fixed_rz, (3, 10))

        FOR i IN range(8):
            APPLY_GATE(RY(x[i]), wire=i)

        FOR layer_idx IN range(3):
            FOR q IN range(10):
                APPLY_GATE(RY(ry_layers[layer_idx][q]), wire=q)
                APPLY_GATE(RZ(rz_layers[layer_idx][q]), wire=q)
            ENTANGLING_BLOCK_CNOT(8, 2)

        RETURN MEASURE_15_OBSERVABLES(readout_wires=[8, 9])

    RETURN circuit_fn, n_trainable
```

The fixed RZ vector must be saved as an artifact and hashed in the run manifest. It must not be updated by COBYLA during either training stage.

#### 11.11.7 Ablation configuration and execution

Ablation runs use a dedicated entry point so that the 48 primary conditions and their R001-R048 identity remain unchanged.

```text
main_ablation.py

STRUCT AblationConfig EXTENDS Config:
    experiment_track = "ABLATION"
    ablation_model    = "MORE-HD-60P" | "MORE-HD-C-FixedRZ"
    K                 = 3 | 6 | 10
    feature_method    = "PCA" | "HU" | "ZERNIKE"
    seed              = one of [101,202,303,404,505]

FUNCTION ABLATION_MAIN(config):
    VALIDATE_ABLATION_CONFIG(config)
    CREATE_RUN_DIRECTORY_EXCLUSIVE(config)
    LOAD_OR_COPY_ABLATION_RUN_SPREADSHEET(config)

    split_manifest = CREATE_OR_LOAD_SPLIT_MANIFEST(config.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = DATA_PIPELINE(config)
    S = CORRELATION_MATRIX(X_train, y_train, config.classes)

    init_bundle = BUILD_ABLATION_INITIALIZATION(config.seed)

    IF config.ablation_model == "MORE-HD-60P":
        circuit_fn, n_trainable = BUILD_CIRCUIT_MORE_HD_60P()
        initial_params = CONCAT(init_bundle.ry_core, init_bundle.ry_extra)
        fixed_rz = NULL

    ELSE IF config.ablation_model == "MORE-HD-C-FixedRZ":
        fixed_rz = init_bundle.rz_phase
        circuit_fn, n_trainable = BUILD_CIRCUIT_MORE_HD_C_FIXED_RZ(fixed_rz)
        initial_params = init_bundle.ry_core
        SAVE(fixed_rz, run_dir + "/artifacts/fixed_rz.npy")

    RUN_STANDARD_TWO_STAGE_PIPELINE_WITH_EXISTING_SPLIT_AND_PAIRS(...)
    FINAL_EVALUATION(...)
    SAVE_ABLATION_MANIFEST(...)
```

The standard clustering and supervised losses remain unchanged. The ablation does not introduce a new optimizer, new feature representation, new label definition, or new test-selection rule.

#### 11.11.8 Ablation IDs and folder names

Ablation condition IDs are separate from R001-R048. Use `ABL001` through `ABL018` with the following deterministic order:

```text
K: 3 -> 6 -> 10
within K: PCA -> HU -> ZERNIKE
within feature: MORE-HD-60P -> MORE-HD-C-FixedRZ
```

A unique ablation execution uses:

```text
ablation_run_uid = ablation_condition_id + "-S" + seed
```

Recommended folder examples:

```text
runs/ablation/ABL001-S101_cls-0-1-2_PCA_MORE-HD-60P/
runs/ablation/ABL002-S101_cls-0-1-2_PCA_MORE-HD-C-FixedRZ/
```

Every ablation manifest must also record the matching primary run IDs for A and D so that the four-model cell can be assembled without rerunning them.

#### 11.11.9 Required ablation artifacts

Each B/C run must save the same core artifacts as a primary run plus:

```text
ablation_condition_id
ablation_run_uid
ablation_model
matching_run_uid_A
matching_run_uid_D
n_trainable_params
n_fixed_rz_params
n_variational_layers
variational_gate_count
entangling_block_count
cnot_count
ry_core_seed
ry_extra_seed            # B only
rz_phase_seed            # C only
fixed_rz.npy             # C only
fixed_rz_sha256          # C only
paired_pair_manifest_sha256
paired_split_manifest_sha256
```

Structural diagnostics must explicitly record the six Y-odd observables `[IY, XY, YI, YX, YZ, ZY]`, their mean/max absolute magnitudes, and the Y-odd norm fraction.

#### 11.11.10 Planned comparisons

The ablation is interpreted through paired, seed-matched contrasts within every `(feature_method, K)` cell:

```text
A -> B : additional real-valued trainable capacity / depth effect
A -> C : access to a complex-valued state with the same number of trainable parameters (30)
B -> D : equal trainable-parameter budget (60) comparison between deeper real RY-only and trainable complex RY+RZ
C -> D : effect of optimizing RZ instead of keeping the same initial RZ phase frozen
```

The two strongest mechanistic checks are A-versus-C and B-versus-D, interpreted jointly with the Y-odd activation diagnostics. Because B has greater variational depth than D, B-versus-D is not described as a perfectly architecture-matched causal contrast.

The primary publication claim remains architecture-level unless the combined ablation evidence supports a narrower phase-specific interpretation.

#### 11.11.11 Statistical treatment

The same five confirmatory seeds form paired replications for the ablation. For each planned contrast and metric, report raw five paired differences, mean difference, standard deviation, median [IQR], 95% Student-t confidence interval for the mean difference, and paired Hedges g. The exact two-sided sign-flip permutation test is retained as a sensitivity test.

If inferential p-values for all four ablation contrasts are reported within the same `(feature_method, K, metric)` family, Holm correction is applied across the four planned contrasts. Effect magnitude and the raw five paired deltas remain more important than a binary significance threshold because n=5.

#### 11.11.12 Claim boundary

Without this ablation, the direct A-versus-D result supports only a whole-architecture statement because trainable parameter count changes from 30 to 60 together with the addition of RZ. After the ablation, a phase-specific statement is permitted only when the relevant paired comparisons and structural diagnostics are mutually consistent.

In particular, activation of the previously structural-zero Y-odd observables demonstrates removal of the real-state restriction; it does not by itself prove that any accuracy increase is caused by complex phase. Classification metrics, separation metrics, and structural diagnostics must be interpreted together.

---

## 12. BASELINE KLASIK REFERENSI — G1-10 (DIKUNCI 2026-09-27)

### 12.1 Posisi dan batasan

Baseline klasik adalah **referensi konteks**, bukan pesaing dan bukan bagian dari estimand
konfirmatori A-vs-D. Tujuannya menunjukkan berapa akurasi yang dicapai model klasik sederhana
pada **8 fitur yang persis sama** dengan input sirkuit. Hasilnya dilaporkan sekunder/deskriptif
(`MORE_HD_STATISTICAL_ANALYSIS_PLAN.md` §10.2).

| Aspek | Keputusan |
|---|---|
| Model | chance `1/K` (dihitung), Nearest Centroid (NC), Logistic Regression multinomial (LR) |
| Program | `main_classical_baseline.py` — **terpisah** dari `main_train.py`, `main_selected_clustering.py`, dan `main_ablation.py` |
| Sumber input | `X_*_scaled.npy` / `y_*.npy` hasil `DATA_PIPELINE` §2.7 pada run primer MORE-HD (A) dengan `(seed, feature_method, K)` sama |
| Preprocessing ulang | **Tidak ada** — baseline tidak memanggil `DATA_PIPELINE` |
| Cakupan | 5 seed konfirmatori × 3 feature × 8 K = 120 cell; NC + LR = 240 fit |
| Seed 42 | Tidak dijalankan untuk baseline konfirmatori |
| Loss adjuster R MORE | Tidak diimplementasikan di track mana pun (`loss_adjuster_policy="NONE"`) |
| Ditunda (penelitian lanjutan) | kNN / SVM-RBF dan baseline piksel mentah |

Alasan program terpisah: (1) siklus hidupnya berbeda — seluruh 240 fit selesai dalam hitungan
detik, sedangkan run kuantum per `run_uid` memakan waktu lama; baseline dapat diulang tanpa
menyentuh folder run kuantum; (2) pipeline kuantum yang sudah dikunci tidak ditambah cabang
scikit-learn; (3) konsisten dengan pemisahan Jalur B dan ablation.

Alasan tanpa preprocessing ulang: satu-satunya jaminan bahwa input baseline identik dengan
input sirkuit adalah membaca array yang sama. Array bersifat spesifik per cell — PCA dan scaler
di-fit pada train K-kelas milik cell tersebut — sehingga K=3..9 **tidak** boleh diturunkan
dengan memfilter array K=10.

### 12.2 Konstanta

```
BASELINE_SEEDS          = CONFIRMATORY_SEEDS           # [101, 202, 303, 404, 505]
BASELINE_FEATURES       = ["PCA", "HU", "ZERNIKE"]
BASELINE_K_VALUES       = [3, 4, 5, 6, 7, 8, 9, 10]
BASELINE_MODELS         = ["NC", "LR"]                 # chance dihitung, bukan di-fit

LR_C_GRID               = [0.01, 0.1, 1.0, 10.0, 100.0]
LR_PENALTY              = "l2"
LR_SOLVER               = "lbfgs"                      # multinomial untuk K > 2
LR_MAX_ITER             = 5000
LR_FIT_INTERCEPT        = TRUE
LR_CLASS_WEIGHT         = NONE                         # data seimbang per kelas
LR_TIE_RULE             = "SMALLEST_C"                 # tie val accuracy -> C terkecil

NC_METRIC               = "euclidean"
NC_SHRINK_THRESHOLD     = NONE

LOSS_ADJUSTER_POLICY    = "NONE"
BASELINE_ROOT           = "runs/baselines/"
```

### 12.3 Resolusi sumber array dan pemeriksaan identitas input

```
FUNCTION RESOLVE_BASELINE_SOURCE(seed, feature_method, K):
    classes = [0 .. K-1]
    run_name_A = AUTO_GENERATE(classes, 1000, 100, 200, "MORE-HD",   feature_method, seed)
    run_name_D = AUTO_GENERATE(classes, 1000, 100, 200, "MORE-HD-C", feature_method, seed)
    dir_A = "runs/" + run_name_A
    dir_D = "runs/" + run_name_D

    ASSERT EXISTS(dir_A + "/config.json")
    manifest_A = LOAD_JSON(dir_A + "/config.json")
    ASSERT manifest_A.run_mode == "CONFIRMATORY"
    ASSERT manifest_A.seed == seed AND manifest_A.feature_method == feature_method
    ASSERT manifest_A.classes == classes

    names = ["X_train_scaled", "y_train", "X_val_scaled", "y_val", "X_test_scaled", "y_test"]
    hashes_A = {n: SHA256_FILE(dir_A + "/artifacts/" + n + ".npy") FOR n IN names}

    IF EXISTS(dir_D + "/config.json"):
        manifest_D = LOAD_JSON(dir_D + "/config.json")
        hashes_D = {n: SHA256_FILE(dir_D + "/artifacts/" + n + ".npy") FOR n IN names}
        IF hashes_D != hashes_A:
            RAISE_ERROR("Array input A dan D berbeda untuk cell ini -- baseline dibatalkan: " + run_name_A)
        _, check_run_uid_D = BUILD_RUN_IDENTITY(manifest_D)
    ELSE:
        check_run_uid_D = NULL              # dicatat; pengecekan diulang setelah D selesai

    _, run_uid_A = BUILD_RUN_IDENTITY(manifest_A)
    split_hash   = SHA256_FILE(manifest_A.split_manifest)      # splits/seed<SEED>.json
    RETURN dir_A, run_uid_A, check_run_uid_D, hashes_A, split_hash
```

Baseline sebuah cell hanya membutuhkan tahap `DATA_PIPELINE` run A sudah selesai
(artefak §2.7 ditulis sebelum clustering dimulai); tidak perlu menunggu run kuantum selesai.

### 12.4 Pelatihan dan evaluasi satu cell

```
FUNCTION RUN_BASELINE_CELL(seed, feature_method, K):
    dir_A, run_uid_A, run_uid_D, input_hashes, split_hash = RESOLVE_BASELINE_SOURCE(seed, feature_method, K)
    cell_dir = BASELINE_ROOT + "S" + seed + "_cls-" + JOIN([0..K-1], "-") + "_" + feature_method
    CREATE_DIRECTORY_EXCLUSIVE(cell_dir)

    # --- tahap 1: HANYA train + validation yang dimuat ---
    X_train = LOAD(dir_A + "/artifacts/X_train_scaled.npy")
    y_train = LOAD(dir_A + "/artifacts/y_train.npy")
    X_val   = LOAD(dir_A + "/artifacts/X_val_scaled.npy")
    y_val   = LOAD(dir_A + "/artifacts/y_val.npy")
    ASSERT SHAPE(X_train)[1] == 8 AND SHAPE(X_val)[1] == 8

    # Nearest Centroid: tanpa hyperparameter
    nc = FIT_NEAREST_CENTROID(X_train, y_train, metric=NC_METRIC, shrink_threshold=NC_SHRINK_THRESHOLD)

    # Logistic Regression: pilih C di validation
    cv_log = []
    FOR C IN LR_C_GRID:                                # urutan menaik
        model = FIT_LOGISTIC_REGRESSION(
            X_train, y_train, C=C, penalty=LR_PENALTY, solver=LR_SOLVER,
            max_iter=LR_MAX_ITER, fit_intercept=LR_FIT_INTERCEPT, class_weight=LR_CLASS_WEIGHT
        )
        acc_val = ACCURACY(y_val, model.PREDICT(X_val))
        APPEND(cv_log, {"C": C, "val_accuracy": acc_val,
                        "n_iter": model.n_iter, "converged": model.n_iter < LR_MAX_ITER})

    best_acc  = MAX(entry.val_accuracy FOR entry IN cv_log)
    C_selected = MIN(entry.C FOR entry IN cv_log IF entry.val_accuracy == best_acc)   # LR_TIE_RULE
    lr = FIT_LOGISTIC_REGRESSION(X_train, y_train, C=C_selected, ...)                 # refit pada TRAIN saja
    SAVE_ATOMIC_JSON({"grid": cv_log, "C_selected": C_selected}, cell_dir + "/LR_cv_log.json")

    # --- tahap 2: model beku; baru sekarang official test dimuat ---
    X_test = LOAD(dir_A + "/artifacts/X_test_scaled.npy")
    y_test = LOAD(dir_A + "/artifacts/y_test.npy")

    rows = []
    FOR name, model IN [("NC", nc), ("LR", lr)]:
        y_pred = model.PREDICT(X_test)
        SAVE(y_pred, cell_dir + "/" + name + "_test_predictions.npy")
        cm = CONFUSION_MATRIX(y_test, y_pred, labels=[0..K-1])
        SAVE(cm, cell_dir + "/" + name + "_confusion_matrix.npy")
        APPEND(rows, BASELINE_ROW(seed, feature_method, K, name, y_test, y_pred,
                                  C_selected IF name == "LR" ELSE NULL))

    APPEND(rows, CHANCE_ROW(seed, feature_method, K))     # accuracy = macro_F1_ref = 1/K

    SAVE_ATOMIC_JSON({
        "seed": seed, "feature_method": feature_method, "K": K,
        "source_run_uid_A": run_uid_A, "check_run_uid_D": run_uid_D,
        "input_array_sha256": input_hashes, "split_manifest_hash": split_hash,
        "loss_adjuster_policy": LOSS_ADJUSTER_POLICY,
        "sklearn_version": VERSION("scikit-learn"), "numpy_version": VERSION("numpy"),
        "git_commit": GIT_COMMIT()
    }, cell_dir + "/baseline_cell_manifest.json")

    RETURN rows


FUNCTION BASELINE_ROW(seed, feature_method, K, name, y_test, y_pred, C_selected):
    RETURN {
        "seed": seed, "feature_method": feature_method, "K": K, "baseline": name,
        "accuracy": ACCURACY(y_test, y_pred),
        "macro_f1": F1(y_test, y_pred, average="macro"),
        "per_class_f1": F1(y_test, y_pred, average=NONE),
        "C_selected": C_selected,
        "n_test": LENGTH(y_test)
    }
```

Aturan akses data: tidak ada `LOAD` terhadap `X_test_scaled`/`y_test` sebelum seluruh model
(NC dan LR dengan `C_selected`) dibekukan. Unit test wajib memverifikasi urutan ini, sama
seperti aturan no-test-access G0-01 pada pipeline kuantum.

### 12.5 Program utama

```
FUNCTION MAIN_CLASSICAL_BASELINE():
    SAVE_ATOMIC_JSON({
        "models": BASELINE_MODELS, "lr_c_grid": LR_C_GRID, "lr_penalty": LR_PENALTY,
        "lr_solver": LR_SOLVER, "lr_max_iter": LR_MAX_ITER, "lr_tie_rule": LR_TIE_RULE,
        "nc_metric": NC_METRIC, "loss_adjuster_policy": LOSS_ADJUSTER_POLICY,
        "seeds": BASELINE_SEEDS, "features": BASELINE_FEATURES, "K_values": BASELINE_K_VALUES
    }, BASELINE_ROOT + "baseline_config.json")

    all_rows = []
    FOR seed IN BASELINE_SEEDS:
        FOR K IN BASELINE_K_VALUES:
            FOR feature_method IN BASELINE_FEATURES:
                all_rows += RUN_BASELINE_CELL(seed, feature_method, K)

    ASSERT COUNT(all_rows WHERE baseline IN ["NC", "LR"]) == 240
    SAVE_CSV(all_rows, BASELINE_ROOT + "baseline_results.csv")
    # Konsolidasi ke workbook: sheet 15_Classical_Baselines (lihat 12.6)
```

Deterministik: NC dan LR-lbfgs tidak memakai randomness; tidak ada seed tambahan yang perlu
diturunkan. Bila sebuah cell gagal (misalnya run A belum ada), cell tersebut dilewati dengan
error tercatat dan dijalankan ulang kemudian dalam folder baru; cell yang sudah ada tidak ditimpa.

### 12.6 Pelaporan dan workbook

| Artefak | Isi |
|---|---|
| `runs/baselines/baseline_results.csv` | 1 baris per `(seed, feature_method, K, baseline)`; sumber data mentah |
| Workbook sheet `15_Classical_Baselines` | Template 360 baris (120 cell × NC/LR/CHANCE) dengan `source_run_uid_A`, `check_run_uid_D`, `C_selected`, accuracy, macro-F1, dan selisih seed-matched terhadap A/D |
| Workbook sheet `16_MORE_Reference` | Nilai MORE\R dan MORE (+R) dari Tabel I Wu et al. (2023); hanya accuracy; ditandai "reported, not reproduced" |

Evaluasi pembanding:

1. **A vs D (primer):** tidak berubah — mengikuti SAP §6, §8, §9.
2. **Kuantum vs NC/LR (sekunder, protokol identik):** accuracy dan macro-F1 pada official test yang
   sama; mean ± SD lintas 5 seed; selisih seed-matched `Metric(quantum) - Metric(baseline)`
   dilaporkan deskriptif; McNemar opsional dan berlabel sekunder dengan koreksi Holm per seed.
   Metrik struktural (`min_separation_ratio`, dimensi aktif) tidak berlaku untuk baseline klasik.
3. **Kuantum vs MORE (referensi literatur, protokol berbeda):** hanya accuracy, deskriptif,
   tanpa uji statistik; kolom MORE\R sebagai acuan utama dan MORE (+R) sebagai konteks.

Batas klaim: akurasi LR adalah batas bawah linear, bukan ukuran total informasi dalam sebuah
representasi fitur. Kalimat seperti "HU mengandung informasi lebih sedikit daripada PCA" tidak
boleh ditarik dari baseline ini.

---

## Hal yang Sengaja Belum Ditentukan (Perlu Keputusan Anda)
 
Seluruh lima keputusan yang sebelumnya "sengaja belum ditentukan" pada draf awal dokumen ini sudah ditutup (format penyimpanan parameter, kebijakan crash recovery, `cobyla_tol`, `active_dim_threshold` awal, `n_cluster_pair_samples`). Keputusan Hu Moments dan Zernike Moments (2.8.2, 2.8.3) juga sudah dikunci pada sesi 2026-09-22, begitu juga skema train/validation/official test (G0-01, bagian 2, 5, 7, 9, 10), dan sejak sesi 2026-09-22 angka final `n_train_per_class`/`n_val_per_class`/`n_test_per_class` untuk protokol publikasi (G0-03) juga sudah dikunci (lihat poin 7 di bawah, kini berstatus selesai). Item yang masih terbuka untuk pilot saat ini:
 
1. **Overhead langkah (b) dan (c)** di `objective_clustering` — menghitung ulang output SEMUA data train (untuk centroid sementara) di **setiap** panggilan objective bisa lumayan berat sekarang `n_train_per_class` sudah dikunci ke 1000 (naik 10× dari pilot 100). Ini diukur pada pilot konvergensi §1.1 (waktu per objective evaluation dicatat per run) sebelum `max_nfev` final (G1-01) dikunci, supaya total waktu 330 unique confirmatory executions (240 primary + 90 ablation) bisa diproyeksikan realistis.
2. **Validasi `active_dim_threshold = 1e-6`** — akan ditinjau ulang setelah prototipe MORE-HD-C benar-benar dijalankan dan dilihat skala nilai aktualnya.
3. **Perilaku Jalur B saat `clustering_params.bin` sendiri korup/tidak lengkap** (bukan sekadar `selected_eval_id` di luar rentang, tapi filenya sendiri rusak) — belum dirancang penanganannya secara eksplisit; untuk pilot ini diasumsikan tidak terjadi karena skala data kecil.
4. **Jumlah parallel run maksimum** sengaja tidak dikunci di kode. Pengguna akan menentukan sendiri jumlah proses aktif berdasarkan observasi CPU dan RAM saat pilot serta saat eksperimen berlangsung.
5. **Konsolidasi spreadsheet lokal ke master** belum diotomatisasi pada pseudocode ini. Primary track menghasilkan hingga 240 `run_result.xlsx`, sedangkan targeted ablation menambah 90 B/C `run_result.xlsx` dalam namespace terpisah. Konsolidasi akhir harus mempertahankan workbook primer dan workbook ablation sebagai dua sumber yang dapat di-join tanpa menduplikasi A/D.
6. **Implementasi Python nyata** untuk `HU_MOMENTS`, `SIGNED_LOG_TRANSFORM`, `MAP_IMAGE_TO_UNIT_DISK`, dan `EXTRACT_ZERNIKE_TERMS` (2.8.4) — pseudocode-nya sudah dikunci, tapi pemilihan library persis (`cv2`, `mahotas`, atau lainnya) dan unit test terhadap kriteria penerimaan Gate G2-01/G2-02 belum dikerjakan.
7. **[SELESAI, sesi 2026-09-22] Angka final `n_train_per_class`/`n_val_per_class`/`n_test_per_class` untuk protokol publikasi** — dikunci ke `n_train_per_class=1000`, `n_val_per_class=100`, `n_test_per_class=200`. `n_train`/`n_test` mengikuti skala FRD-09 (komparabilitas dengan thesis lama + presisi statistik confidence interval yang memadai untuk klaim G1-04 pada K=3..10). `n_val_per_class` DIREVISI dari aturan proporsi G0-01 sebelumnya (`= n_test_per_class`) menjadi angka independen lebih kecil (separuh dari test), karena validation hanya berperan untuk monitoring/checkpoint selection (bukan klaim akhir publikasi) sehingga tidak memerlukan presisi setara test set; ketersediaan pool MNIST per digit (train ~5.400–6.700, test resmi ~980–1.135) dicek dan mencukupi untuk kombinasi `1000 (train) + 100 (val) = 1100` dan `200 (test)` di semua digit. Waktu komputasi sengaja TIDAK menjadi pertimbangan pada keputusan ini (akan diuji lewat pilot timing terpisah, lihat poin 1); keputusan murni berbasis presisi statistik dan komparabilitas metodologis.
8. **Implementasi kode nyata + unit test untuk G0-01** — skema split train/validation/test dan penghapusan akses test dari `CLUSTERING_LOOP`/`SUPERVISED_LOOP`/Jalur B sudah dikunci di level pseudocode (bagian 2, 5, 7, 9, 10), tapi unit test yang memverifikasi tidak ada pemanggilan `circuit_fn` terhadap `X_test`/`y_test` sebelum `FINAL_EVALUATION` belum ditulis.
9. **Implementasi + unit test G0-02** — konflik manual-vs-otomatis sudah diselesaikan pada sesi 2026-09-27. Jalur B sekarang memakai selector lexicographic deterministik berbasis train/validation; `selected_eval_id` dan `max_nfev_supervised` tidak lagi menjadi input manual. Status tetap `IN PROGRESS` sampai implementasi Python membuktikan selector reproducible, tidak mengakses official test, menghasilkan selection log lengkap, dan mengikuti domain kandidat yang nanti dikunci oleh G1-06.