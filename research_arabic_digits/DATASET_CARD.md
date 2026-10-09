# Dataset Card — Digit Arab

**Status:** `LOCAL-RESEARCH READY — RAW REDISTRIBUTION NOT AUTHORIZED`
**Versi:** 1.0
**Tanggal:** 2026-10-09

Dokumen ini adalah sumber kebenaran untuk data mentah dan turunannya. Tidak ada
lisensi redistribusi eksplisit pada halaman resmi yang diperiksa; karena itu
proyek memakai kebijakan konservatif `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`.

## 1. Identitas wajib

| Bidang | Nilai |
|---|---|
| Nama resmi dataset | Modified Arabic Digits Database (MADBase; situs juga menyebut “Modified ADBase”) |
| Singkatan yang dipakai | `MADBASE-AUC-V1` |
| Versi/rilis/tanggal snapshot | Tidak ada nomor versi upstream; snapshot lokal 2026-10-08 dikunci dengan hash arsip |
| Pemilik/penerbit | Sherif Abdelazeem dan Ezzat El-Sherif, American University in Cairo |
| URL resmi atau DOI | `https://datacenter.aucegypt.edu/shazeem/` |
| Halaman unduhan yang dipakai | Tautan resmi `MAHDBase_TrainingSet.rar` dan `MAHDBase_TestingSet.rar` |
| Tanggal unduh | 2026-10-08 (Asia/Jakarta) |
| Lisensi | **Belum dinyatakan dalam lisensi standar pada halaman resmi**; unduhan publik tidak dianggap otomatis mengizinkan redistribusi |
| Citation | El-Sherif & Abdelazeem, *Arabic handwritten digit recognition*, IJDAR 11, 127–141 (2008), DOI `10.1007/s10032-008-0073-5`; konfirmasi citation dataset tetap diperlukan |
| Ketentuan redistribusi | Belum dikonfirmasi; arsip/citra dikecualikan dari Git |
| Kontak sumber | `shazeem@aucegypt.edu`, `ezzatali@aucegypt.edu` sebagaimana halaman resmi |

**Keputusan identitas:** `ARABIC_INDIC` (`٠١٢٣٤٥٦٧٨٩`), berdasarkan nama label
dan inspeksi visual seluruh sepuluh kelas. Bukti mesin-terbaca ada di
`research_data/manifests/label_mapping.json`.

Istilah berikut tidak boleh dicampur:

- Western Arabic digits: `0 1 2 3 4 5 6 7 8 9`;
- Arabic-Indic digits: `٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩`;
- Eastern Arabic/Persian digits: `۰ ۱ ۲ ۳ ۴ ۵ ۶ ۷ ۸ ۹`.

Nama dataset dan dokumentasi resminya menentukan kategori yang benar.

## 2. Provenance dan integritas file

Simpan data mentah sebagai read-only di lokasi yang diizinkan lisensi. Data
besar atau terbatas lisensinya tidak boleh di-commit tanpa keputusan eksplisit.

| Artefak sumber | Ukuran byte | SHA-256 | Asal | Status verifikasi |
|---|---:|---|---|---|
| `MAHDBase_TrainingSet.rar` | 31,184,713 | `ae2d5bcd658320eaf054de1ba46aac187a0470830d685258a0a2b0659c8d2eb8` | Unduhan resmi AUC | Cocok dengan snapshot pin |
| `MAHDBase_TestingSet.rar` | 5,487,352 | `a8294a266f1843b4c659291dfda6e71620abbfcaec47e4313024c691eb494b1d` | Unduhan resmi AUC | Cocok dengan snapshot pin |

Manifest checksum kanonik berada di:
`research_arabic_digits/research_data/manifests/raw_files.sha256`.

Kriteria penerimaan:

- setiap arsip dan file mentah memiliki SHA-256;
- proses ekstraksi dicatat tanpa menimpa arsip sumber;
- jumlah file/baris sebelum dan sesudah ekstraksi cocok;
- satu snapshot dataset dipakai untuk seluruh run;
- perubahan upstream tidak mengganti snapshot secara diam-diam.

## 3. Struktur data

| Properti | Nilai terverifikasi |
|---|---|
| Modalitas | Citra tulisan tangan digit tunggal |
| Format file | BMP 8-bit indexed grayscale; loader menghasilkan array `uint8` |
| Dimensi citra | 28×28 |
| Kanal warna | Satu kanal grayscale |
| Tipe data/rentang piksel | `uint8`, 0–255 |
| Latar/foreground dan polaritas | Mentah: putih/gelap; kanonik: hitam/terang dengan transformasi `255 - raw` |
| Jumlah kelas | 10 |
| Label asli | Integer dari nama file `digit0`…`digit9` |
| Jumlah sampel total | 70.000 |
| Unit penulis/subjek | 700 penulis; 100 citra per penulis |
| Official train/test split | Ada: 60.000 train dan 10.000 test |
| Metadata penulis tersedia | Ada pada nama file; train writer001–600, test writer601–700 |

### Inventaris kandidat per kelas dan split

Tabel 0–9 berikut hanya berlaku bila sumber terverifikasi sebagai dataset digit
tunggal sepuluh kelas. Bila domain tugas berbeda, ganti tabel berdasarkan
`label_mapping.json`; jangan memaksa sampel ke sepuluh label.

| Label numerik | Glif terverifikasi | Train tersedia | Test tersedia | Total | Catatan |
|---:|---|---:|---:|---:|---|
| 0 | `٠` | 6.000 | 1.000 | 7.000 | |
| 1 | `١` | 6.000 | 1.000 | 7.000 | |
| 2 | `٢` | 6.000 | 1.000 | 7.000 | |
| 3 | `٣` | 6.000 | 1.000 | 7.000 | |
| 4 | `٤` | 6.000 | 1.000 | 7.000 | |
| 5 | `٥` | 6.000 | 1.000 | 7.000 | |
| 6 | `٦` | 6.000 | 1.000 | 7.000 | |
| 7 | `٧` | 6.000 | 1.000 | 7.000 | |
| 8 | `٨` | 6.000 | 1.000 | 7.000 | |
| 9 | `٩` | 6.000 | 1.000 | 7.000 | |

## 4. Label mapping

Domain label internal ditentukan oleh mapping sumber yang sudah diverifikasi dan
harus disimpan eksplisit. Untuk task digit tunggal sepuluh kelas yang
terkonfirmasi, mapping yang diharapkan adalah integer `0..9`; untuk task lain,
domain wajib dicatat apa adanya. Jangan menyimpulkan label dari urutan folder
tanpa memeriksa dokumentasi dan contoh citra.

| Label sumber | Label internal | Glif | Bukti/verifikasi |
|---|---:|---|---|
| `digit0`…`digit9` | 0…9 | `٠١٢٣٤٥٦٧٨٩` | Nama file, distribusi kelas, dan inspeksi visual writer001/pass01 |

Verifikasi minimum:

1. tampilkan contoh acak dari setiap label sebelum preprocessing;
2. minta satu pemeriksaan manusia atas seluruh kelas;
3. uji bahwa label source-to-internal bersifat satu-ke-satu;
4. simpan mapping sebagai artefak mesin-terbaca;
5. uji mapping setelah decoding/ekstraksi, bukan hanya pada nama arsip.

## 5. Kebijakan split

Keputusan sumber: official train/test split dipertahankan. Train dan validation
dibentuk secara writer-disjoint hanya dari official training pool. Pilot test
memakai kelompok writer ketiga dari pool tersebut. Final test konfirmatori
berasal dari 20 writer official test yang dibekukan, diberi asal
`OFFICIAL_SOURCE`, dan identik lintas semua seed.

Ukuran per kelas adalah 1000 train, 100 validation, dan 200 final test. Seed
pilot adalah 42; seed konfirmatori 101, 202, 303, 404, dan 505. Manifest dan
hashnya berada di `research_arabic_digits/splits/`.

Urutan preferensi:

1. pertahankan official train/test split jika disediakan dan dapat diverifikasi;
2. ambil validation hanya dari official training pool;
3. jika ada ID penulis, cegah penulis yang sama tersebar antar split apabila
   tujuan studi menuntut generalisasi antarpenulis;
4. jika dataset tidak memiliki official split, kunci strategi group-stratified
   atau stratified split sebelum melihat hasil model;
5. beri final held-out test asal `OFFICIAL_SOURCE` bila berasal dari official
   split, atau `CONSTRUCTED_FROZEN` bila dibentuk oleh protokol;
6. simpan identitas sampel, label, seed, dan hash manifest untuk setiap split.

Class regime memakai label kumulatif/nested `0..K-1` untuk K=3..10. Budget
optimizer MNIST tidak dipakai untuk run konfirmatori MADBase.

## 6. Audit kualitas dan kebocoran

| Pemeriksaan | Metode/bukti yang direncanakan | Hasil |
|---|---|---|
| File rusak/tidak dapat dibaca | decode seluruh citra | 70.000/70.000 berhasil |
| Label di luar domain | validasi himpunan label | Tidak ada; domain tepat 0–9 |
| Dimensi/kanal tidak konsisten | ringkasan bentuk dan dtype | Semua 28×28; decoder kanonik satu kanal `uint8` |
| Duplikat piksel-identik | SHA-256 per array citra | 39 grup hanya dalam satu split; 12 grup melintasi train/test |
| Near-duplicate lintas split | dHash Hamming ≤1 lalu pixel MAE ≤2 | 3.879 pasangan kandidat; belum menjadi aturan eksklusi |
| Sampel identik dengan label berbeda | join hash terhadap label | 0 grup konflik label |
| Kebocoran penulis | audit ID penulis | 0 writer overlap; train 1–600, test 601–700 |
| Ketidakseimbangan kelas | tabel jumlah per kelas | Tidak ada; 6.000/kelas train, 1.000/kelas test |
| Orientasi/polaritas salah | inspeksi sepuluh label + statistik | Glif benar; raw putih/gelap, dibalik menjadi hitam/terang |
| Blank/crop ekstrem | foreground dan border | 0 blank; 1.068 train dan 155 test menyentuh border, perlu keputusan protokol |

Duplikat tidak boleh dihapus setelah melihat performa model. Aturan deduplikasi
atau eksklusi harus dibekukan dari audit data dan seluruh ID terdampak dicatat.

## 7. Preprocessing yang perlu diputuskan

| Keputusan | Nilai | Status |
|---|---|---|
| Decoding dan dtype kanonik | Pillow `L` → array `uint8` 28×28 | `FROZEN` |
| Normalisasi intensitas | `uint8 / 255.0` setelah loader | `FROZEN` |
| Inversi polaritas | `255 - raw` agar foreground terang | `FROZEN` |
| Resize/pad/crop | Tidak ada; pertahankan 28×28 official | `FROZEN` |
| Interpolasi resize | Tidak berlaku karena tidak ada resize | `N/A` |
| Centring | Tidak ada re-centring | `FROZEN` |
| Augmentasi | `NONE` | `FROZEN` |
| PCA/Hu/Zernike geometry | Delapan kanal sesuai `core/constants.py`; fit/scaler train-only | `FROZEN` |

Transformasi tidak boleh merotasi, mencerminkan, atau menukar makna glif.
Statistik yang dipelajari dari data—PCA, scaler, ambang, atau pilihan parameter—
harus di-fit pada train saja.

## 8. Lisensi, privasi, dan etika

- Simpan citra hanya secara lokal untuk penelitian; jangan redistribusikan raw
  atau contoh citra sampai izin tertulis/lisensi eksplisit tersedia.
- Catat apakah data membawa ID penulis atau metadata pribadi.
- Jangan menampilkan sampel mentah pada paper/dasbor sebelum izin jelas.
- Hindari klaim tentang bahasa, kewarganegaraan, atau populasi penulis yang tidak
  dinyatakan oleh sumber dataset.
- Laporkan keterbatasan representativitas penulis, perangkat akuisisi, gaya
  tulisan, dan domain penggunaan.
- Model ini untuk penelitian klasifikasi digit, bukan verifikasi identitas atau
  inferensi karakteristik penulis.

## 9. Checklist penutupan dataset card

- [x] Identitas/sumber/snapshot hash dataset terisi.
- [x] Citation dicatat; ketiadaan lisensi eksplisit ditangani dengan kebijakan
  local-only/no-redistribution.
- [x] Semua arsip mentah memiliki SHA-256.
- [x] Jumlah sampel per kelas cocok dengan sumber.
- [x] Mapping label dan glif diverifikasi.
- [x] Official split dipertahankan dan seluruh manifest writer-disjoint dibuat.
- [x] Duplikat/near-duplicate diaudit; snapshot dipertahankan tanpa eksklusi diam-diam.
- [x] Preprocessing dibekukan tanpa memakai outcome model official test.
- [x] Contoh tiap kelas diperiksa setelah decoding.
- [x] Bukti audit ditautkan pada `READINESS_GATES.md`.
