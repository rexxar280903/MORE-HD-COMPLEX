# Pseudocode — Pilot Training Pipeline MORE-HD-C
 
Dokumen ini adalah **alur logika (pseudocode)**, bukan kode Python yang bisa dijalankan.
Tujuannya untuk direview dulu sebelum implementasi asli ditulis.
 
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
baik di `CLUSTERING_LOOP`, `SUPERVISED_LOOP`, maupun pemilihan manual Jalur B), dan
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
`max_nfev_clustering` dan `max_nfev_supervised`. Jalur B memilih
`selected_eval_id`, bukan "iteration". Spreadsheet per-run wajib memakai header
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
├── main_selected_clustering.py    # Jalur B: lanjut dari objective evaluation clustering pilihan manual
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
│   └── MORE_HD_master_confirmatory_240runs.xlsx
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
    └── cls-0-1-2-3-4-5-6-7-8-9_ntrain1000_nval100_ntest200_ZERNIKE_MORE-HD-C_seed42/
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

data_seed  = DERIVE_SUBSEED(seed, "data")
init_seed  = DERIVE_SUBSEED(seed, "init")
pair_seed  = DERIVE_SUBSEED(seed, "cluster_pairs")
label_seed = DERIVE_SUBSEED(seed, "quantum_labels")
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
                                                # serta untuk pemilihan manual di Jalur B.
    n_test_per_class       = 200               # DIKUNCI (G0-03, sesi 2026-09-22): official test -- TIDAK diakses
                                                # sebelum FINAL_EVALUATION (G0-01)
    max_nfev_clustering      = 10               # smoke test saja; budget final dikunci lewat pilot G1-01
    max_nfev_supervised      = 10               # jumlah maksimum objective-function evaluations, BUKAN iterasi

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
- **Validation** (`X_val_scaled`, `y_val`) diambil dari pool `mnist_train_raw`, disjoint dari train, BUKAN dipotong dari pool test. Validation dipakai untuk semua monitoring/checkpoint selection selama `CLUSTERING_LOOP`, `SUPERVISED_LOOP`, dan pemilihan manual Jalur B (G0-01, sesi 2026-09-22).
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
FUNCTION MODEL_SETUP(config):
    SET_RANDOM_SEED(config.seed)
 
    IF config.architecture == "MORE-HD":
        circuit_fn, n_params = BUILD_CIRCUIT_MORE_HD(n_data_qubits=config.n_data_qubits, n_readout_qubits=config.n_readout_qubits)
    ELSE IF config.architecture == "MORE-HD-C":
        circuit_fn, n_params = BUILD_CIRCUIT_MORE_HD_C(n_data_qubits=config.n_data_qubits, n_readout_qubits=config.n_readout_qubits)
    ELSE:
        RAISE_ERROR("architecture harus 'MORE-HD' atau 'MORE-HD-C'")
 
    initial_params = RANDOM_UNIFORM(low=0, high=2*PI, size=n_params)
 
    SAVE(initial_params, run_dir + "/artifacts/initial_params.npy")
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
Menambahkan RZ di layer variational adalah cara paling langsung menyuntik
fase kompleks **tanpa mengubah** cara data masuk (layer 1), cara qubit
saling terhubung (layer 3), atau cara hasil diukur (layer 4) — sehingga
kalau nanti akurasi MORE-HD-C berbeda dari MORE-HD, perbedaan itu bisa
diatribusikan langsung ke penambahan RZ ini, bukan ke faktor lain.
 
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
        FOR each (x, true_class) IN ZIP(X_val, y_val):
            v = circuit_fn(x, theta)
            distances = { c: COSINE_DISTANCE(v, temp_centroids[c]) FOR c IN config.classes }
            predicted_class = ARGMIN(distances)
            IF predicted_class == true_class:
                correct += 1
            dist_true  = distances[true_class]
            dist_other = MIN(distances[c] FOR c IN config.classes IF c != true_class)
            margins.APPEND(dist_other - dist_true)
 
        pseudo_accuracy_val = correct / LENGTH(X_val)
        avg_margin_val = MEAN(margins)
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
            "train_loss": train_loss,
            "pseudo_accuracy_val": pseudo_accuracy_val,
            "avg_margin_val": avg_margin_val,
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
            "tol": config.cobyla_tol
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
 
Enam nilai pada `clustering_log.jsonl` dicatat **per objective evaluation
(`eval_id`)**, bukan per iterasi optimizer:
 
| Metrik | Fungsi (untuk apa dipakai) | Satuan / Rentang Nilai |
|---|---|---|
| `train_loss` | Objective utama yang dibaca COBYLA. | Skalar tak berdimensi dari `-S_ij × cosine_distance`. |
| `pseudo_accuracy_val` | Proxy generalisasi pada validation terhadap centroid sementara. | Proporsi 0–1. |
| `avg_margin_val` | Margin validation antara kelas benar dan pengganggu terdekat. | Selisih cosine distance; teoretis -2 sampai +2. |
| `min_separation` | Jarak cosine minimum antar centroid kelas. | 0–2. |
| `correlation_consistency` | Konsistensi urutan jarak centroid terhadap matriks korelasi S. | Spearman -1 sampai +1. |
| `active_dimensions` | Jumlah dimensi observable aktif pada centroid. | Integer 0–15. |

Kelima metrik selain `train_loss` bersifat monitoring pasif. Semuanya memakai
train/validation saja; official test tetap hanya digunakan di
`FINAL_EVALUATION`. Grafik loss, separation, margin, dan active dimensions
yang berasal dari log ini harus memakai sumbu-X **Objective Evaluation
(`eval_id`)**, bukan "Iteration".
 
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
            "tol": config.cobyla_tol
        }
    )

    trained_params_final = result.x
    ASSERT result.nfev == n_objective_evals

    SAVE(result.x, run_dir + "/artifacts/supervised_params_final.npy")
    SAVE(best_observed_theta, run_dir + "/artifacts/supervised_params_best_observed.npy")

    optimizer_summary = {
        "optimizer": "COBYLA",
        "max_nfev": config.max_nfev_supervised,
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
        "seed": config.seed,
        "master_seed": config.seed,
        "run_mode": config.run_mode,
        "data_seed": DERIVE_SUBSEED(config.seed, "data"),
        "init_seed": DERIVE_SUBSEED(config.seed, "init"),
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
    max_nfev_clustering=10,   # smoke-test budget only; final budget ditentukan G1-01
    max_nfev_supervised=10,
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

## 10. JALUR B — Pemilihan Manual Objective Evaluation Clustering
 
### 10.1 Latar Belakang
 
`result.x` yang dipakai Jalur A adalah **final point yang dikembalikan COBYLA**.
Ia tidak boleh disebut otomatis sebagai titik dengan `train_loss` terbaik.
Riwayat `clustering_log.jsonl` berisi setiap parameter yang benar-benar
dievaluasi oleh objective dan diidentifikasi dengan `eval_id`. Secara terpisah,
`clustering_optimizer_result.json` mencatat final point, `nfev`, status
terminasi, dan `best_observed_eval_id`.

Jalur B tetap mengizinkan inspeksi train/validation untuk memilih satu titik
objective evaluation tertentu. Nama inputnya adalah `selected_eval_id`, bukan
`selected_eval_id`. Official test tidak boleh digunakan pada pemilihan ini.

**Catatan G0-02:** pemilihan manual tetap merupakan isu metodologis terpisah dan
belum dianggap terselesaikan hanya karena terminologi G1-02 diperbaiki.
 
**Input manual dari pengguna:**
 
1. `selected_eval_id` — ID objective evaluation clustering pada run sumber.
2. `max_nfev_supervised` — budget maksimum objective evaluations supervised
   untuk run Jalur B.
 
### 10.2 Pseudocode
 
```
FUNCTION MAIN_FROM_SELECTED_CLUSTERING(
    source_run_dir,
    selected_eval_id,
    max_nfev_supervised
):
 
    source_manifest = LOAD_JSON(source_run_dir + "/config.json")
    source_clustering_result = LOAD_JSON(
        source_run_dir + "/logs/clustering_optimizer_result.json"
    )
 
    config = Config(
        classes                = source_manifest.classes,
        n_train_per_class      = source_manifest.n_train_per_class,
        n_val_per_class        = source_manifest.n_val_per_class,
        n_test_per_class       = source_manifest.n_test_per_class,
        max_nfev_clustering    = source_manifest.max_nfev_clustering,  # provenance saja
        max_nfev_supervised    = max_nfev_supervised,
        architecture           = source_manifest.architecture,
        feature_method         = source_manifest.feature_method,
        n_data_qubits          = source_manifest.n_data_qubits,
        n_readout_qubits       = source_manifest.n_readout_qubits,
        n_input_channels       = source_manifest.n_input_channels,
        cobyla_tol             = source_manifest.cobyla_tol,
        seed                   = source_manifest.seed,
        run_mode               = source_manifest.run_mode,
        n_cluster_pair_samples = source_manifest.n_cluster_pair_samples
    )
 
    config.run_name = (
        source_manifest.run_name +
        "_jalurB_eval" +
        ZERO_PAD(selected_eval_id, 4)
    )
    config.run_dir = "runs/" + config.run_name
 
    CREATE_DIRECTORY(config.run_dir + "/artifacts")
    CREATE_DIRECTORY(config.run_dir + "/logs")
 
    # Validasi terhadap ACTUAL nfev, bukan budget maksimum.
    actual_nfev = source_clustering_result.nfev
    IF selected_eval_id < 0 OR selected_eval_id >= actual_nfev:
        RAISE_ERROR(
            "selected_eval_id di luar objective evaluations yang benar-benar " +
            "tersimpan. Rentang valid: 0.." + STRING(actual_nfev - 1)
        )
 
    X_train_scaled = LOAD(source_run_dir + "/artifacts/X_train_scaled.npy")
    y_train        = LOAD(source_run_dir + "/artifacts/y_train.npy")
    X_val_scaled   = LOAD(source_run_dir + "/artifacts/X_val_scaled.npy")
    y_val          = LOAD(source_run_dir + "/artifacts/y_val.npy")
    X_test_scaled  = LOAD(source_run_dir + "/artifacts/X_test_scaled.npy")
    y_test         = LOAD(source_run_dir + "/artifacts/y_test.npy")
 
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
 
    metrics, confusion_matrix = FINAL_EVALUATION(
        circuit_fn, trained_params_final, quantum_labels,
        X_test_scaled, y_test, config
    )
 
    SAVE_ARTIFACT_BUNDLE_JALUR_B(
        config, metrics, source_run_dir, selected_eval_id
    )
 
    PRINT("Jalur B selesai. Hasil ada di: " + config.run_dir)
    PRINT(
        "theta diambil dari objective evaluation #" +
        selected_eval_id +
        " pada run sumber: " +
        source_run_dir
    )
 
    RETURN metrics, confusion_matrix
 
 
FUNCTION SAVE_ARTIFACT_BUNDLE_JALUR_B(
    config, metrics, source_run_dir, selected_eval_id
):
 
    supervised_optimizer_result = LOAD_JSON(
        config.run_dir + "/logs/supervised_optimizer_result.json"
    )

    manifest = {
        "run_name": config.run_name,
        "path_type": "jalur_b_manual_selection",
        "timestamp": NOW(),
        "source_run_dir": source_run_dir,
        "selected_eval_id": selected_eval_id,
        "selection_basis": "manual_visual_inspection_train_val_metrics",
        "classes": config.classes,
        "n_train_per_class": config.n_train_per_class,
        "n_val_per_class": config.n_val_per_class,
        "n_test_per_class": config.n_test_per_class,
        "max_nfev_supervised": config.max_nfev_supervised,
        "architecture": config.architecture,
        "feature_method": config.feature_method,
        "n_data_qubits": config.n_data_qubits,
        "n_readout_qubits": config.n_readout_qubits,
        "n_input_channels": config.n_input_channels,
        "cobyla_tol": config.cobyla_tol,
        "seed": config.seed,
        "master_seed": config.seed,
        "run_mode": config.run_mode,
        "data_seed": DERIVE_SUBSEED(config.seed, "data"),
        "init_seed": DERIVE_SUBSEED(config.seed, "init"),
        "pair_seed": DERIVE_SUBSEED(config.seed, "cluster_pairs"),
        "label_seed": DERIVE_SUBSEED(config.seed, "quantum_labels"),
        "split_manifest": config.split_manifest_path,
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
            "confusion_matrix": "logs/confusion_matrix.npy"
        }
    }
 
    SAVE(manifest, config.run_dir + "/config.json")
 
 
# Contoh:
MAIN_FROM_SELECTED_CLUSTERING(
    source_run_dir      = "runs/cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed42",
    selected_eval_id    = 7,
    max_nfev_supervised = 15
)
```
 
### 10.3 Ringkasan Perbedaan Jalur A vs Jalur B
 
| Aspek | Jalur A (otomatis) | Jalur B (manual) |
|---|---|---|
| Sumber `theta` clustering | `result.x` = final point resmi COBYLA | `READ_PARAM_RECORD` pada `selected_eval_id` pilihan manusia |
| Riwayat objective | `clustering_log.jsonl` dengan `eval_id` | Membaca log yang sama dari run sumber |
| Total objective calls | `result.nfev` | Validasi `selected_eval_id` memakai `source_clustering_result.nfev` |
| Best observed | Disimpan terpisah dari final point | Dapat dilihat melalui `best_observed_eval_id`, tetapi tidak wajib dipilih |
| Budget supervised | `max_nfev_supervised` dari config awal | Input manual `max_nfev_supervised` |
| Akses official test | Hanya `FINAL_EVALUATION` | Hanya `FINAL_EVALUATION` |
| Nama run | normal | suffix `_jalurB_evalXXXX` |
| Provenance | manifest Jalur A | `source_run_dir` + `selected_eval_id` + `selection_basis` |
 
---
 
## Hal yang Sengaja Belum Ditentukan (Perlu Keputusan Anda)
 
Seluruh lima keputusan yang sebelumnya "sengaja belum ditentukan" pada draf awal dokumen ini sudah ditutup (format penyimpanan parameter, kebijakan crash recovery, `cobyla_tol`, `active_dim_threshold` awal, `n_cluster_pair_samples`). Keputusan Hu Moments dan Zernike Moments (2.8.2, 2.8.3) juga sudah dikunci pada sesi 2026-09-22, begitu juga skema train/validation/official test (G0-01, bagian 2, 5, 7, 9, 10), dan sejak sesi 2026-09-22 angka final `n_train_per_class`/`n_val_per_class`/`n_test_per_class` untuk protokol publikasi (G0-03) juga sudah dikunci (lihat poin 7 di bawah, kini berstatus selesai). Item yang masih terbuka untuk pilot saat ini:
 
1. **Overhead langkah (b) dan (c)** di `objective_clustering` — menghitung ulang output SEMUA data train (untuk centroid sementara) di **setiap** panggilan objective bisa lumayan berat sekarang `n_train_per_class` sudah dikunci ke 1000 (naik 10× dari pilot 100). Ini perlu diukur lewat pilot timing sebelum `maxiter` final (G1-01) dikunci, supaya total waktu 240 run konfirmatori bisa diproyeksikan realistis.
2. **Validasi `active_dim_threshold = 1e-6`** — akan ditinjau ulang setelah prototipe MORE-HD-C benar-benar dijalankan dan dilihat skala nilai aktualnya.
3. **Perilaku Jalur B saat `clustering_params.bin` sendiri korup/tidak lengkap** (bukan sekadar `selected_eval_id` di luar rentang, tapi filenya sendiri rusak) — belum dirancang penanganannya secara eksplisit; untuk pilot ini diasumsikan tidak terjadi karena skala data kecil.
4. **Jumlah parallel run maksimum** sengaja tidak dikunci di kode. Pengguna akan menentukan sendiri jumlah proses aktif berdasarkan observasi CPU dan RAM saat pilot serta saat eksperimen berlangsung.
5. **Konsolidasi 48 spreadsheet lokal ke master** belum diotomatisasi pada pseudocode ini. Training hanya menghasilkan `run_result.xlsx` per-run; penggabungan akhir dilakukan setelah seluruh run yang diperlukan selesai.
6. **Implementasi Python nyata** untuk `HU_MOMENTS`, `SIGNED_LOG_TRANSFORM`, `MAP_IMAGE_TO_UNIT_DISK`, dan `EXTRACT_ZERNIKE_TERMS` (2.8.4) — pseudocode-nya sudah dikunci, tapi pemilihan library persis (`cv2`, `mahotas`, atau lainnya) dan unit test terhadap kriteria penerimaan Gate G2-01/G2-02 belum dikerjakan.
7. **[SELESAI, sesi 2026-09-22] Angka final `n_train_per_class`/`n_val_per_class`/`n_test_per_class` untuk protokol publikasi** — dikunci ke `n_train_per_class=1000`, `n_val_per_class=100`, `n_test_per_class=200`. `n_train`/`n_test` mengikuti skala FRD-09 (komparabilitas dengan thesis lama + presisi statistik confidence interval yang memadai untuk klaim G1-04 pada K=3..10). `n_val_per_class` DIREVISI dari aturan proporsi G0-01 sebelumnya (`= n_test_per_class`) menjadi angka independen lebih kecil (separuh dari test), karena validation hanya berperan untuk monitoring/checkpoint selection (bukan klaim akhir publikasi) sehingga tidak memerlukan presisi setara test set; ketersediaan pool MNIST per digit (train ~5.400–6.700, test resmi ~980–1.135) dicek dan mencukupi untuk kombinasi `1000 (train) + 100 (val) = 1100` dan `200 (test)` di semua digit. Waktu komputasi sengaja TIDAK menjadi pertimbangan pada keputusan ini (akan diuji lewat pilot timing terpisah, lihat poin 1); keputusan murni berbasis presisi statistik dan komparabilitas metodologis.
8. **Implementasi kode nyata + unit test untuk G0-01** — skema split train/validation/test dan penghapusan akses test dari `CLUSTERING_LOOP`/`SUPERVISED_LOOP`/Jalur B sudah dikunci di level pseudocode (bagian 2, 5, 7, 9, 10), tapi unit test yang memverifikasi tidak ada pemanggilan `circuit_fn` terhadap `X_test`/`y_test` sebelum `FINAL_EVALUATION` belum ditulis.
9. **Konflik G0-02** — kriteria penerimaan G0-02 meminta fungsi pemilihan checkpoint Jalur B yang otomatis dan deterministik, sementara keputusan yang dikunci di bagian 10.1 adalah pemilihan manual oleh Ken. Ini belum diselesaikan; Ken akan menentukan sendiri resolusinya (skor otomatis / dua jalur manual+auto / revisi kriteria) berdasarkan analisisnya sendiri. Lihat catatan di 10.1 dan log keputusan `MORE_HD_RESEARCH_READINESS_GATES`.