# Register Referensi — Studi Digit Arab

**Status:** `PRIMARY SOURCES REGISTERED / LICENSE RESTRICTED BY POLICY`
**Tanggal:** 2026-10-09

File ini adalah register verifikasi, bukan bibliografi final. MADBase telah
dipilih dan sumber primernya dicatat. Karena halaman resmi tidak menampilkan
lisensi redistribusi eksplisit, raw data diperlakukan local-only.

## 1. Referensi dataset

| ID | Kebutuhan | Status | Bukti yang harus disimpan |
|---|---|---|---|
| DATASET-01 | Paper/dokumentasi resmi dataset | `CLOSED` | Halaman resmi AUC dan DOI paper MADBase |
| DATASET-02 | Lisensi/terms dataset | `CLOSED WITH RESTRICTION` | tidak ada grant eksplisit; local-only/no redistribution |
| DATASET-03 | Deskripsi split, penulis, dan proses akuisisi | `CLOSED` | Halaman resmi + `dataset_inventory.json` |
| DATASET-04 | Mapping label/glif | `CLOSED` | `label_mapping.json` + inspeksi sepuluh kelas |

### Sumber dataset yang dipakai

| ID | Referensi | DOI/URL | Peran |
|---|---|---|---|
| MADBASE-WEB | Abdelazeem, S. & El-Sherif, E. *The Arabic Handwritten Digits Databases: ADBase & MADBase*. American University in Cairo. Diakses 2026-10-08. | `https://datacenter.aucegypt.edu/shazeem/` | Sumber unduhan, ukuran, penulis, split, dan konstruksi MADBase |
| MADBASE-PAPER | Abdleazeem, S. & El-Sherif, E. (2008). *Arabic handwritten digit recognition*. International Journal on Document Analysis and Recognition, 11, 127–141. | `10.1007/s10032-008-0073-5` | Deskripsi/benchmark primer; ejaan nama mengikuti metadata paper |

## 2. Referensi metode yang sudah ada di repositori

Metadata berikut berasal dari bibliografi studi MNIST saat ini dan tetap perlu
diverifikasi kembali pada penyusunan paper digit Arab.

| ID | Referensi | DOI/status |
|---|---|---|
| MORE-01 | Wu, J., Hu, T., & Li, Q. (2023). *MORE: Measurement and Correlation Based Variational Quantum Circuit for Multi-classification*. IEEE QCE, 208–218. | `10.1109/QCE57702.2023.00031` |
| HU-01 | Hu, M.-K. (1962). *Visual pattern recognition by moment invariants*. IRE Transactions on Information Theory, 8, 179–187. | `10.1109/TIT.1962.1057692` |
| ZERNIKE-01 | Teague, M. R. (1980). *Image analysis via the general theory of moments*. Journal of the Optical Society of America, 70, 920. | `10.1364/JOSA.70.000920` |
| ZERNIKE-02 | Khotanzad, A., & Hong, Y. H. (1990). *Invariant image recognition by Zernike moments*. IEEE TPAMI, 12, 489–497. | `10.1109/34.55109` |
| QML-01 | Biamonte, J. et al. (2017). *Quantum machine learning*. Nature, 549, 195–202. | `10.1038/nature23474` |
| VQA-01 | Cerezo, M. et al. (2021). *Variational quantum algorithms*. Nature Reviews Physics, 3, 625–644. | `10.1038/s42254-021-00348-9` |

Sumber BibTeX awal: `../paper/referensi_paper.bib`. Verifikasi referensi existing:
`../paper/VERIFIKASI_REFERENSI.md`.

## 3. Aturan penggunaan sumber

- Klaim jumlah sampel, resolusi, jumlah penulis, atau split harus menunjuk ke
  sumber dataset yang sama dengan snapshot yang digunakan.
- Halaman agregator/marketplace tidak menggantikan sumber resmi bila paper atau
  repositori pemilik tersedia.
- Simpan tanggal akses untuk halaman yang dapat berubah.
- Jangan menyalin angka dataset kandidat ke metode sebelum checksum snapshot
  yang benar sudah dicatat.
- Gunakan kutipan sekunder hanya bila sumber primer tidak tersedia, dan tandai.

## 4. Checklist sebelum bibliografi final

- [x] DATASET-01 sampai DATASET-04 memiliki keputusan dan bukti.
- [x] DOI/URL cocok dengan metadata yang dicatat.
- [x] Citation key stabil dan unik.
- [ ] Setiap klaim metode/pembatasan memiliki sumber yang tepat.
- [x] Tidak ada referensi MNIST yang dipakai untuk mendeskripsikan digit Arab.
- [ ] Semua referensi yang dikutip muncul di bibliografi dan sebaliknya.
