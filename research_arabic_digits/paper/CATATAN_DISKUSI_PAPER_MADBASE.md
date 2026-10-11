# Catatan Diskusi Penyusunan Paper — MORE-HD / MORE-HD-C pada MADBase (digit Arab)

> File ini merekam keputusan, temuan, dan langkah lanjutan naskah studi digit Arab. Baca bagian **Keputusan** dan
> **Langkah berikutnya** terlebih dahulu. Studi MNIST memiliki catatannya sendiri (`paper/CATATAN_DISKUSI_PAPER.md`)
> dan **tidak diubah** oleh pekerjaan ini.

- Mulai: 2026-10-11
- Target jurnal: Vietnam Journal of Computer Science (template disalin ke `template/ws-vjcs template.docx`)
- Bahasa: Indonesia dulu (`MORE-HD-C_MADBase_paper_ID.docx`); versi Inggris menyusul.
- Sumber data: `../research_data/consolidated/`, `../research_data/analysis/`, diekstrak ke `build/data/`.
- Dokumen acuan: `../RESEARCH_PROTOCOL.md`, `../STATISTICAL_ANALYSIS_PLAN.md` (v1.0 + amendemen administratif 1.0.1),
  `../RESULTS.md`, `../DEVIATION_LOG.md`.

---

## 1. Keputusan

| # | Keputusan | Alasan |
|---|---|---|
| M-1 | Paper dibingkai sebagai **replikasi konfirmatori** temuan MNIST ("kapasitas, bukan fase kompleks") pada dataset kedua yang independen. | Hasil primer dan keempat kontras ablasi mengulang arah dan besaran MNIST. |
| M-2 | Perbandingan dengan MORE memakai **MORE-REPRO di dalam protokol** (Tabel 4), bukan nilai literatur. | Wu dkk. (2023) tidak melaporkan MORE pada MADBase; Tabel I mereka hanya MNIST. |
| M-3 | Perbandingan dengan MNIST hanya **deskriptif** (Subbab 4.9, Tabel 7, Gambar 9). | SAP §8 dan AD-D005; protokol split/budget berbeda. |
| M-4 | Jalur B masuk **Lampiran A** sebagai analisis sensitivitas sekunder. | Sudah dipra-spesifikasi di protokol MADBase (berbeda dari keputusan K-3 MNIST). |
| M-5 | Analisis mekanistik dan pasangan kelas diberi label **eksploratif**. | Dirancang setelah hasil diketahui (DEV-05). |
| M-6 | Tidak ada citra MADBase di naskah; hanya glif Unicode. | Kebijakan `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`. |
| M-7 | Angka di teks dibaca otomatis dari `build/numbers.json` (f-string di `content_id.py`). | Mencegah salah salin; naskah selalu sinkron dengan analisis. |

## 2. Status data (dicek 2026-10-11)

| Track | Run | Status |
|---|---:|---|
| Primer A/D | 240 | Semua `COMPLETED`, attempt 1, label kuantum tidak degenerate, selisih backend ≤ 1,8e-15 |
| Ablasi B/C | 90 | Semua `COMPLETED`, `paired_input_check = MATCH` |
| MORE-REPRO (M) | 120 | Semua `COMPLETED`, MATCH |
| Jalur B | 90 | Semua `COMPLETED`, test tidak dipakai untuk seleksi |
| Baseline klasik | 240 fit (+120 chance analitik) | Semua `COMPLETED`, LR konvergen |

Admission final: `research_data/reports/final_admission.json` (10 Okt 2026 19:02 WIB, `accepted = true`).

## 3. Temuan utama (angka dari `build/numbers.json`)

- **Primer D − A:** +5,9 pp akurasi (22/24 sel positif; SK>0 pada PCA K=6, PCA K=7, Hu K=7; tidak ada SK<0);
  PCA +9,1, Hu +4,0, Zernike +4,5. Macro-F1 +6,5 pp (21/24). Separasi +0,020 (17/24, tanpa SK yang mengecualikan nol).
  McNemar 86/120 signifikan setelah Holm (65 menguntungkan D, 21 menguntungkan A).
- **Ablasi (K ∈ {3, 6, 10}):** B − A +12,3 (9/9 positif), C − A +2,5 (6/9), D − B −9,0 (0/9 positif), D − C +0,7 (5/9).
  B terbaik di 8/9 sel. p Holm terkecil 0,25 (keterbatasan lima seed).
- **MORE-REPRO:** M 37,9%; A − M +11,9 (22/24), D − M +17,8 (22/24). Selisih negatif hanya pada Hu K kecil, SK memuat nol.
- **Baseline:** LR 85,2%, NC 73,3%, D 55,7%, A 49,8%, M 37,9%, peluang 17,9%. D tidak pernah menyamai NC/LR.
- **6↔9 (pra-spesifikasi):** 4,6% galat A–D; merged +2,4 pp. Pada glif Arab-Indik ٦/٩ bukan rotasi.
- **Pasangan kelas (eksploratif, K = 10):** Hu: ٠↔٥ 6,3%, ٧↔٨ 5,4%; Zernike: ٧↔٨ 6,3%; PCA: 6↔9 8,5%, 7↔8 hanya 0,6%.
- **Mekanistik (eksploratif, 180 run):** reproduksi akurasi selisih 0,0. Probe D 79,4% ≈ B 78,5% ≈ C 78,9% > A 72,0%
  (D − B +0,9 [−0,4; 2,3]). Tanpa Y-ganjil, probe D −4,9 pp, akurasi label −3,2 pp. Effective rank A 14,8; C 18,4;
  D 26,6; B 44,0 (D − B −17,4). Rank D 50–52 (acak: 52); 8 RZ lapisan terakhir inert (norma kolom ≤ 1,8e-15).
  ρ(eff. rank, akurasi) dalam sel 0,50. Masih membaik di akhir anggaran: B 67% vs D 58% (Fisher p = 0,51) —
  berbeda dengan MNIST, tidak signifikan.
- **Lintas dataset (deskriptif):** D − A MNIST +5,9 vs MADBase +5,9; ablasi sejajar; korelasi pola per sel ρ = 0,04
  (tanda sama 18/24). Semua model lebih akurat di MADBase dalam protokol masing-masing.
- **Jalur B:** seleksi checkpoint clustering dari validasi menaikkan A +5,9 pp dan D +5,3 pp; D − A rata-rata +2,7 pp
  (vs +3,3 pp di titik akhir; 5/9 positif; tanda sama 6/9).

**Batas klaim (wajib dipatuhi):** tidak ada klaim manfaat fase kompleks; kata "signifikan" hanya untuk McNemar;
perbandingan MNIST dan analisis pasca-hasil dilabeli deskriptif/eksploratif; tidak ada klaim quantum advantage;
tidak ada klaim kausal efek dataset; generalisasi penulis terbatas (20 penulis uji tetap).

## 4. Risiko reviewer dan jawabannya

| Risiko | Jawaban |
|---|---|
| "Hanya mengulang paper MNIST" | Replikasi pada aksara, penulis, split writer-disjoint, dan budget berbeda adalah kontribusi; tambahan temuan khas aksara (٧↔٨, ٠↔٥). Deklarasikan studi pendamping ke editor; tulis ulang bagian metode bila perlu untuk menghindari kemiripan teks. |
| Tidak ada p < 0,05 per sel | Resolusi sign-flip 5 seed = 0,0625; estimation-first sesuai SAP §6. |
| Lisensi MADBase | Data tidak diredistribusi; minta konfirmasi tertulis penggunaan ke pemilik dataset sebelum submit. |
| Set uji hanya 20 penulis | Diakui di keterbatasan; dibekukan sebelum hasil, identik lintas seed. |
| Duplikat lintas split (12 grup) | Dilaporkan, tidak dieksklusi sesuai protokol beku. |
| Jalur B mengubah besar efek | Lampiran A; arah rata-rata bertahan, per sel kurang konsisten. |

## 5. Struktur naskah (25 halaman render LibreOffice)

1. Pendahuluan (RQ i–iv, kontribusi) · 2. Penelitian terkait (VQC multi-kelas; ekspresibilitas; riil vs kompleks;
digit Arab & fitur) · 3. Metode (data & split; fitur; sirkuit; Proposisi 1; pelatihan dua fase; ukuran hasil; desain &
statistik; analisis eksploratif) · 4. Hasil (kelengkapan; D vs A; ablasi; diagnostik readout; MORE-REPRO; baseline;
per kelas; mekanistik + Proposisi 2; lintas dataset; biaya) · 5. Pembahasan · 6. Kesimpulan · Ketersediaan data &
kode · Lampiran A (Jalur B) · 40 referensi.
Gambar 1–9 (`figures/id/`), Tabel 1–7 + A.1, 13 persamaan.

## 6. Langkah berikutnya

- [ ] Isi data penulis, afiliasi, email, header halaman genap (placeholder `[...]`).
- [ ] Isi URL repositori dan ucapan terima kasih/pendanaan.
- [ ] Minta izin/konfirmasi penggunaan MADBase untuk publikasi (kontak di `../DATASET_CARD.md`).
- [ ] Putuskan cara mengutip studi MNIST pendamping (preprint/naskah dalam review) — saat ini disebut "studi pendamping kami".
- [ ] Baca ulang naskah, terutama Pembahasan dan tafsiran pasangan glif (eksploratif).
- [ ] Commit skrip orkestrasi dan dokumen G6 ke git (DEV-02) lalu jalankan ulang `../tools/finalize_g6.py`.
- [ ] Buat versi bahasa Inggris (`content_en.py`) bila diperlukan untuk pengiriman.
- [ ] Render final di Microsoft Word (persamaan memakai MML2OMML Word saat build di Windows).

## 7. Peta file

| File | Isi |
|---|---|
| `MORE-HD-C_MADBase_paper_ID.docx` | Naskah bahasa Indonesia |
| `figures/id/*.png` | Gambar 300 dpi |
| `build/content_id.py` | **Teks naskah** — ubah di sini lalu build ulang |
| `build/build_paper.py` | Renderer ke template VJCS (Word XSL bila ada, selain itu pandoc) |
| `build/tables_data.py`, `build/tables.json`, `build/numbers.json` | Angka tabel dan teks |
| `build/analyze_madbase.py` | Analisis lengkap + cek silang terhadap `research_data/analysis` |
| `build/make_figures.py` | Semua gambar |
| `build/write_results.py` | Mengisi `../RESULTS.md` |
| `build/extract_workbooks.py`, `build/collect_jalurb.py`, `build/data/` | Ekstraksi sheet workbook dan Jalur B |
| `build/mechanism/` | Analisis mekanistik eksploratif (input, skrip, hasil) |
| `build/references.py`, `build/crossref_refs.json` | Referensi (MADBase diverifikasi ke Crossref 2026-10-11) |
| `build/render/MORE-HD-C_MADBase_paper_ID.pdf` | Pratinjau PDF (LibreOffice) |

Cara build ulang (dari root repo):

```bash
python research_arabic_digits/paper/build/extract_workbooks.py
python research_arabic_digits/paper/build/collect_jalurb.py
python research_arabic_digits/paper/build/analyze_madbase.py
python research_arabic_digits/paper/build/write_results.py
python research_arabic_digits/paper/build/make_figures.py
python research_arabic_digits/paper/build/build_paper.py
```

## 8. Log

- **2026-10-11:** Training konfirmatori dinyatakan selesai; G6 ditutup (RESULTS, DEVIATION_LOG, manifest
  reproduksi). Analisis lengkap + mekanistik eksploratif dijalankan ulang dari artefak (180 sirkuit, reproduksi
  akurasi persis). Draf paper ID (25 hlm) dan dashboard MADBase dibuat. File studi MNIST tidak diubah.
