# MORE-HD Training Observatory — MADBase (digit Arab)

Dashboard lokal untuk mengeksplorasi hasil training studi digit tulisan tangan Arab-Indik (MADBase) dari workbook di
`research_arabic_digits/research_data/consolidated` dan tabel statistik di `research_arabic_digits/research_data/analysis`.
Ini salinan terpisah dari dashboard MNIST (`research_data/data_from_laptop/dashboard`); dashboard dan data MNIST tidak
diubah.

## Membuka dashboard

1. Klik kanan `serve.ps1`, lalu pilih **Run with PowerShell**.
2. Buka `http://localhost:4174` di browser (port berbeda dari dashboard MNIST, 4173, sehingga keduanya bisa dibuka bersamaan).
3. Biarkan jendela PowerShell tetap terbuka selama dashboard digunakan.

## Memperbarui data

Jalankan `refresh_data.ps1` (atau `python build_data.py`). Script hanya membaca data sumber dan menulis ulang folder
`data`. Untuk mesin dengan waktu eksekusi terbatas, build dapat dipecah per fase:

```powershell
python build_data.py --phase summary
python build_data.py --phase history --track PRIMARY --part clustering
python build_data.py --phase history --track PRIMARY --part supervised
python build_data.py --phase details --track PRIMARY
# ulangi history/details untuk ABLATION dan MORE_REFERENCE
```

Tab **Temuan & replikasi** membaca `research_arabic_digits/paper/build/numbers.json`, `jalurB_summary.csv`,
`confusion_pairs_k10.csv`, dan `cross_dataset_cells.csv`; jalankan `paper/build/analyze_madbase.py` lebih dulu bila
hasil analisis berubah.

## Cakupan

- 240 run utama MORE-HD dan MORE-HD-C, 90 run ablasi model B dan C, 120 run reproduksi MORE (snapshot `MADBASE-AUC-V1`).
- Label kelas ditampilkan bersama glif Arab-Indik (٠–٩).
- Metrik per kelas, confusion matrix, threshold sensitivity, benchmark sirkuit, baseline klasik, serta seluruh evaluasi
  clustering dan supervised per run.
- Tab tambahan: ringkasan konfirmatori, perbandingan deskriptif dengan MNIST, Jalur B, pasangan glif yang tertukar, dan
  analisis mekanistik (eksploratif).

Citra mentah MADBase tidak dimuat atau ditampilkan (kebijakan `LOCAL_RESEARCH_ONLY_NO_RAW_REDISTRIBUTION`). Dashboard
tidak membutuhkan koneksi internet.
