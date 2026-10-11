"""Naskah bahasa Indonesia: MORE-HD/MORE-HD-C pada MADBase (digit tulisan tangan Arab-Indik).

Template Vietnam Journal of Computer Science. Angka di teks dibaca dari numbers.json (analyze_madbase.py) agar
naskah selalu sinkron dengan data; tabel dibaca dari tables.json lewat tables_data.py.
"""
import json
from pathlib import Path

from tables_data import (primary_rows, ablation_rows, more_rows, baseline_rows, mechanism_rows, jalurb_rows,
                         cross_rows)

N = json.loads((Path(__file__).resolve().parent / "numbers.json").read_text(encoding="utf-8"))


def f(x, d=1):
    """Angka gaya Indonesia (koma desimal, tanda minus tipografis)."""
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−").replace(".", ",")


def s(x, d=1):
    out = f(x, d)
    return out if out.startswith("−") else "+" + out


def ci(v, d=1):
    return f"[{f(v['lo'], d)}; {f(v['hi'], d)}]"


def pval(p):
    if p < 1e-3:
        e = int(f"{p:.0e}".split("e")[1])
        return f"*p* < $10^{{{e + 1}}}$"
    return f"*p* = {f(p, 3 if p < 0.01 else 2)}"


P, AB, MC = N["primary_acc"], N["ablation"], N["mcnemar"]
F1, SEP = N["primary_f1_macro"], N["primary_min_separation_ratio"]
MM, BM, SIX, JB = N["model_mean_acc"], N["baseline_mean"], N["six_nine"], N["jalurB"]
MECH = N.get("mechanism")
X = N["cross"]
Q = N["quality"]
RT = N["runtime_median_sec"]
BU = N["benchmark_unitary_ms"]
ABC = N["model_mean_acc_ablation_cells"]
SEPA = N["model_mean_sep_ablation_cells"]
SP = N["spearman_sep_acc"]
SYO = N["spearman_yodd_acc_D_within"]
CP = N["confusion_top3"]


def share(f_, pair):
    return f(dict(CP[f_]).get(pair, 0) * 100) if pair in dict(CP[f_]) else None


LABELS = {"fig": "Gambar", "tab": "Tabel", "ref": "Ref.", "refs": "Ref.", "where": "dengan"}

TITLE_LINES = ["Kapasitas, Bukan Fase Kompleks: Replikasi Studi Ablasi",
               "Pengklasifikasi Kuantum Variasional Multi-Readout pada",
               "Digit Tulisan Tangan Arab"]
AUTHORS = ["[Nama Penulis Pertama]", "[Nama Penulis Kedua]"]
AFFILIATIONS = [["[Departemen, Universitas, Alamat]", "[Kota, Kode Pos, Negara]", "[email_penulis_pertama@universitas.ac.id]"],
                ["[Departemen, Universitas, Alamat]", "[Kota, Kode Pos, Negara]", "[email_penulis_kedua@universitas.ac.id]"]]
HISTORY = ["Diterima (Tanggal Bulan Tahun)", "Direvisi (Tanggal Bulan Tahun)"]
HEADERS = {"Authors’ Names": "[Nama Penulis]",
           "Instructions for Typing Manuscripts (Paper’s Title)": "Kapasitas versus Fase Kompleks pada Digit Tulisan Tangan Arab"}

ABSTRACT = (
    "Pengklasifikasi kuantum variasional yang membaca 15 ekspektasi Pauli dari dua qubit readout dapat menampung "
    "banyak kelas, tetapi ansatz yang hanya berisi gerbang RY dan CNOT menghasilkan state beramplitudo riil sehingga "
    "enam observabel dengan jumlah faktor Y ganjil selalu bernilai nol. MORE-HD-C menambahkan rotasi RZ terlatih untuk "
    "menghapus batasan tersebut, sekaligus menggandakan jumlah parameter terlatih. Kami menguji apakah atribusi "
    "kenaikan akurasinya bertahan pada dataset kedua yang independen, yaitu Modified Arabic Digits Database (MADBase) "
    "berisi digit Arab-Indik dari 700 penulis. Dengan protokol yang dibekukan sebelum hasil tersedia, split latih dan "
    "validasi yang terpisah menurut penulis, serta set uji resmi dari 20 penulis yang sama untuk semua seed, kami "
    "melatih 450 sirkuit pada tugas kumulatif tiga hingga sepuluh kelas dengan tiga himpunan fitur delapan kanal dan "
    f"lima seed berpasangan. MORE-HD-C menaikkan akurasi uji MORE-HD rata-rata {f(P['mean'])} poin persentase dan "
    f"lebih tinggi pada {P['n_pos']} dari {P['n']} kondisi. Ablasi yang dirancang sebelumnya kembali mengatribusikan "
    "kenaikan itu pada kapasitas parameter terlatih, bukan pada fase kompleks: sirkuit riil 60 parameter mengungguli "
    f"MORE-HD-C pada kesembilan sel ablasi (rata-rata {f(-AB['D−B']['mean'])} poin), sedangkan melatih sudut RZ "
    f"alih-alih membekukannya hanya mengubah akurasi {s(AB['D−C']['mean'])} poin. Analisis eksploratif atas sirkuit "
    "terlatih mereplikasi penjelasannya: readout kompleks tidak membawa informasi kelas yang dapat didekode lebih banyak "
    "daripada readout riil, delapan sudut RZ tidak memengaruhi readout, dan sisanya hanya menambah sedikit arah "
    "independen. Kedua varian MORE-HD mengungguli reproduksi sirkuit MORE di dalam protokol yang sama, tetapi semua "
    "model kuantum masih jauh di bawah baseline klasik."
)
KEYWORDS = ("*Kata kunci*: Pengklasifikasi kuantum variasional; klasifikasi multi-kelas; readout kuantum; "
            "structural zeros; studi ablasi; replikasi; digit tulisan tangan Arab; MADBase.")

BLOCKS = [
    # ================================================================== 1 Pendahuluan
    ("h1", "Pendahuluan"),
    ("p", "Quantum machine learning mempelajari apakah sirkuit kuantum berparameter dapat dipakai sebagai model "
          "terlatih untuk data klasik.[@biamonte2017,@preskill2018,@bharti2022] Pada pendekatan variasional, sudut-sudut "
          "gerbang sebuah sirkuit dievaluasi pada perangkat atau simulator kuantum, kemudian optimizer klasik "
          "memperbaruinya untuk menurunkan sebuah loss.[@cerezo2021,@benedetti2019] Pengklasifikasi kuantum "
          "variasional (variational quantum classifier, VQC) yang dibangun dengan cara ini sudah diuji pada banyak "
          "benchmark kecil,[@mitarai2018,@farhi2018,@schuld2020,@havlicek2019] tetapi sebagian besar rancangannya "
          "berangkat dari klasifikasi biner, dengan keputusan yang dibaca dari satu atau beberapa nilai ekspektasi."),
    ("p", "Klasifikasi multi-kelas menuntut ruang readout yang cukup lapang untuk menempatkan *K* perwakilan kelas "
          "secara terpisah. MORE, yang diusulkan Wu *dkk.*,[@wu2023] mengukur satu qubit readout pada basis *X*, *Y*, "
          "dan *Z*, lalu memperlakukan ketiga ekspektasi sebagai vektor tiga dimensi. Fase clustering variasional yang "
          "dipandu korelasi antar-kelas menempatkan satu *label kuantum* per kelas di ruang tersebut, dan fase supervised "
          "menarik setiap sampel ke label kelasnya. Wu *dkk.* menunjukkan bahwa label yang makin berdesakan ketika *K* "
          "bertambah menurunkan akurasi."),
    ("p", "MORE-HD (Multi-Readout qubit expansion for High-Dimensional label space) mempertahankan pelatihan dua fase "
          "dan aturan label terdekat MORE, tetapi memakai ansatz lain: delapan qubit data dan dua qubit readout, enkoding "
          "sudut RY, serta tiga lapisan variasional yang masing-masing berisi RY terlatih pada kesepuluh qubit dan blok "
          "CNOT tetap. Kedua qubit readout menghasilkan 15 ekspektasi Pauli dua qubit non-identitas. Karena state awal, "
          "RY, dan CNOT semuanya riil, amplitudo state MORE-HD selalu riil dan enam observabel dengan jumlah faktor *Y* "
          "ganjil selalu nol, sehingga hanya sembilan dari 15 dimensi readout yang dapat aktif. MORE-HD-C menyisipkan RZ "
          "terlatih setelah setiap RY variasional untuk membuka keenam dimensi itu. Akan tetapi, MORE-HD-C sekaligus "
          "menggandakan parameter terlatih dari 30 menjadi 60, padahal jumlah dan susunan parameter sendiri dapat "
          "mengubah fungsi yang dipelajari sirkuit.[@sim2019,@holmes2022,@mcclean2018,@schuld2021,@abbas2021] Dua "
          "kontrol ablasi memisahkan kedua penjelasan: MORE-HD-60P, sirkuit riil dengan 60 parameter, dan "
          "MORE-HD-C-FixedRZ, sirkuit kompleks dengan sudut RZ yang dibekukan."),
    ("p", "Pada MNIST,[@lecun1998] studi pendamping kami dengan rancangan ini menemukan bahwa kenaikan akurasi "
          "MORE-HD-C berasal dari kapasitas parameter terlatih, bukan dari fase kompleks. Kesimpulan yang bertumpu pada "
          "satu benchmark tetap rentan terhadap kekhasan dataset tersebut: bentuk glif, cara penulisan, dan susunan "
          "split ikut menentukan informasi yang tersisa setelah citra dikompresi menjadi delapan kanal. Makalah ini "
          "menguji apakah temuan itu bertahan pada Modified Arabic Digits Database (MADBase),[@abdleazeem2008,@madbaseweb] "
          "yang memuat digit Arab-Indik (٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩) dari 700 penulis dalam format 28 × 28 piksel. Glif "
          "Arab-Indik berbeda bentuk dari digit Latin; misalnya, ٠ ditulis sebagai titik kecil dan ٧ serta ٨ "
          "berbentuk huruf V tegak dan terbalik. Selain itu, identitas penulis tersedia sehingga split dapat dibuat "
          "terpisah menurut penulis. Seluruh desain, termasuk anggaran optimizer yang dikalibrasi ulang pada MADBase, "
          "dibekukan sebelum metrik uji tersedia."),
    ("p", "Makalah ini menjawab empat pertanyaan: (i) apakah MORE-HD-C lebih akurat daripada MORE-HD pada MADBase; "
          "(ii) apakah ablasi kembali mengatribusikan perbedaan itu pada kapasitas parameter; (iii) bagaimana kedua "
          "model dibandingkan dengan sirkuit MORE yang dilatih dengan protokol sama dan dengan baseline klasik; dan "
          "(iv), secara eksploratif, mengapa dimensi readout tambahan kurang membantu dan pasangan glif mana yang paling "
          "sering tertukar. Kontribusi kami adalah:"),
    ("bul", [
        "replikasi konfirmatori pada dataset kedua dengan split latih dan validasi yang terpisah menurut penulis, set "
        "uji resmi tetap dari 20 penulis, lima seed berpasangan, dan rencana analisis statistik yang dibekukan sebelum "
        "hasil tersedia;",
        f"estimasi berpasangan MORE-HD-C dikurangi MORE-HD pada 24 kondisi, dengan rata-rata {s(P['mean'])} poin "
        f"persentase dan arah positif pada {P['n_pos']} kondisi;",
        "ablasi terarah yang kembali menunjukkan bahwa sirkuit riil dengan jumlah parameter terlatih yang sama lebih "
        "akurat daripada sirkuit kompleks, sementara melatih fase hampir tidak berpengaruh;",
        "perbandingan di dalam protokol dengan reproduksi sirkuit MORE dan dengan baseline klasik pada input delapan "
        "kanal yang identik;",
        "analisis eksploratif atas sirkuit terlatih (linear probe dan Jacobian readout) dan atas pasangan kelas yang "
        "tertukar, yang menunjukkan bahwa galat fitur invarian-rotasi terkonsentrasi pada glif Arab-Indik yang saling "
        "berupa rotasi, serta "
        "perbandingan deskriptif dengan studi MNIST."]),
    ("p", "Bab 2 mengulas penelitian terkait, Bab 3 menjelaskan data, model, dan protokol, Bab 4 menyajikan hasil, "
          "Bab 5 membahasnya, dan Bab 6 menyimpulkan. Lampiran A memuat analisis sensitivitas pemilihan checkpoint "
          "clustering."),

    # ================================================================== 2 Penelitian terkait
    ("h1", "Penelitian Terkait"),
    ("h2", "Pengklasifikasi kuantum variasional multi-kelas"),
    ("p", "Rancangan VQC awal, seperti jaringan saraf kuantum Farhi dan Neven, pembelajaran sirkuit kuantum hibrida "
          "Mitarai *dkk.*, dan pengklasifikasi berpusat-sirkuit Schuld *dkk.*, terutama dievaluasi pada tugas "
          "biner.[@farhi2018,@mitarai2018,@schuld2020] Untuk lebih dari dua kelas, satu hasil pengukuran biner tidak "
          "lagi memadai. Menambah qubit yang diukur atau qubit ancilla mengurangi qubit untuk pemrosesan data; pengodean "
          "*one-hot* membutuhkan banyak qubit, sedangkan pengodean biner dapat menyisakan state yang tidak mewakili kelas "
          "mana pun; lapisan klasik tambahan atau dekomposisi menjadi beberapa pengklasifikasi biner menambah parameter "
          "dan biaya inferensi.[@wu2023] Jalur lain menetapkan state label yang mudah dibedakan,[@perez2020] mempelajari "
          "embedding berbasis fidelitas atau jarak,[@lloyd2020] atau memakai arsitektur konvolusional dan hierarkis yang "
          "mengurangi qubit lapis demi lapis.[@cong2019,@grant2018]"),
    ("p", "MORE memanfaatkan satu qubit readout secara lebih penuh dengan mengukur *X*, *Y*, dan *Z* dan mempelajari "
          "label kuantum melalui clustering berbasis korelasi antar-kelas.[@wu2023] Implementasi eksperimennya memakai "
          "delapan qubit, *Z feature map* dengan dua repetisi, dan jaringan konvolusional kuantum tiga tahap dengan 91 "
          "parameter; kode publiknya menunjukkan bahwa blok konvolusi dan pooling memadukan RX, RY, RZ, RYY, RZZ, dan "
          "CNOT. MORE-HD mempertahankan prosedur latih MORE, tetapi mengganti ansatz dengan rangkaian RY-CNOT dan "
          "memperluas readout menjadi dua qubit. Karena ansatz dan readout berubah bersamaan, perbandingan dengan MORE "
          "mencerminkan pengaruh gabungan keduanya."),
    ("h2", "Ekspresibilitas, trainability, dan peran parameter"),
    ("p", "Pada sirkuit acak yang cukup ekspresif, gradien dapat menghilang secara eksponensial terhadap jumlah qubit "
          "(barren plateau),[@mcclean2018] dan besar gradien berkaitan dengan ekspresibilitas ansatz.[@holmes2022] "
          "Ekspresibilitas dan kemampuan entangling berbeda jauh antar-keluarga ansatz dengan jumlah qubit yang "
          "sama,[@sim2019] strategi enkoding membatasi fungsi yang dapat direpresentasikan,[@schuld2021] dan dimensi "
          "efektif telah diusulkan untuk membandingkan kapasitas model kuantum dan klasik.[@abbas2021] Konsekuensinya, "
          "perbandingan arsitektur VQC perlu mengendalikan jumlah parameter terlatih dan kedalaman sirkuit."),
    ("h2", "Amplitudo riil dan kompleks"),
    ("p", "Teori kuantum bernilai riil dapat dibedakan secara eksperimental dari teori kuantum standar, sehingga "
          "bilangan kompleks merupakan bagian esensial teori kuantum.[@renou2021] Meski demikian, banyak ansatz "
          "variasional hanya memakai RY dan CNOT dan karenanya tetap berada di subruang riil. Bagi pengklasifikasi, "
          "pertanyaan praktisnya adalah apakah akses ke state kompleks memperbaiki aturan keputusan, bukan apakah "
          "keluarga state-nya kompleks."),
    ("h2", "Digit tulisan tangan Arab dan reduksi fitur"),
    ("p", "MADBase disusun oleh Abdleazeem dan El-Sherif dari 70.000 citra digit Arab-Indik yang ditulis oleh 700 "
          "penulis, masing-masing menulis setiap digit sepuluh kali, dan sengaja diformat menyerupai MNIST: citra "
          "skala abu-abu 28 × 28 dengan split resmi 60.000 latih dan 10.000 uji.[@abdleazeem2008,@madbaseweb] Penulis "
          "set latih dan set uji tidak tumpang tindih, sehingga dataset ini cocok untuk mengukur generalisasi ke "
          "penulis baru. Karena sirkuit jangka dekat hanya menerima sedikit kanal input, citra perlu dikompresi. "
          "Principal component analysis (PCA) mempertahankan varians, sedangkan invarian momen Hu[@hu1962] dan magnitudo "
          "momen Zernike[@teague1980,@khotanzad1990] invarian terhadap rotasi. Sifat invarian ini menjadi relevan pada "
          "digit Arab-Indik, karena beberapa glif saling berupa rotasi. Model kuantum juga perlu dibandingkan dengan "
          "model klasik pada input terkompresi yang sama, karena keunggulan kuantum jarang teramati pada tugas seperti "
          "ini dan bukan satu-satunya tujuan yang bermakna.[@huang2021,@cerezo2022,@schuldkilloran2022]"),

    # ================================================================== 3 Metode
    ("h1", "Metode"),
    ("h2", "Data dan tugas"),
    ("p", "Kami memakai snapshot MADBase resmi yang diunduh pada 8 Oktober 2026 dan dikunci dengan SHA-256 arsip "
          "(70.000 citra, seluruhnya dapat didekode; 7.000 citra per kelas; 100 citra per penulis).[@madbaseweb] Label "
          "internal 0–9 dipetakan satu-ke-satu ke glif ٠–٩ berdasarkan nama file dan inspeksi visual seluruh kelas. "
          "Polaritas citra dibalik (255 dikurangi nilai piksel) agar latar menjadi gelap seperti MNIST; tidak ada resize, "
          "re-centring, atau augmentasi. Tugasnya memakai himpunan kelas kumulatif {0, …, *K* − 1} untuk *K* = 3, …, 10."),
    ("p", "Untuk setiap master seed, sebuah manifest menetapkan per kelas 1.000 sampel latih dan 100 sampel validasi "
          "yang diambil dari pool latih resmi dengan kelompok penulis yang terpisah (writer-disjoint). Set uji akhir "
          "berisi 200 sampel per kelas dari 20 penulis set uji resmi; set ini dibekukan dan identik untuk semua seed, "
          "sehingga variasi antar-seed mencerminkan variasi data latih, inisialisasi, dan optimisasi, bukan pergantian "
          "sampel uji. Split bersifat *berpasangan* (sampel identik untuk semua model dan metode fitur dalam satu seed) "
          "dan *bersarang* (sampel suatu kelas tidak berubah ketika *K* bertambah). Seed konfirmatori adalah 101, 202, "
          "303, 404, dan 505. Seed 42 hanya dipakai untuk smoke test dan pilot dengan kelompok penulis ketiga dari pool "
          "latih, dan dikeluarkan dari semua hasil. Set uji resmi dimuat secara lazy hanya pada tahap evaluasi akhir dan "
          "tidak tersedia bagi clustering, pelatihan, pemilihan budget, maupun pilot."),
    ("p", "Audit data menemukan 0 penulis yang tumpang tindih antara set latih dan uji resmi, 0 citra identik dengan "
          "label berbeda, 39 grup duplikat identik di dalam satu split, dan 12 grup duplikat identik yang melintasi "
          "split latih dan uji resmi. Sesuai protokol, snapshot resmi dipertahankan tanpa eksklusi, dan duplikat "
          "dilaporkan sebagai keterbatasan. Karena halaman resmi tidak menyatakan lisensi redistribusi, data mentah "
          "hanya dipakai secara lokal untuk penelitian dan tidak ada citra mentah yang ditampilkan dalam makalah ini."),
    ("h2", "Representasi fitur"),
    ("p", "Setiap metode fitur menghasilkan tepat delapan kanal. **PCA** mengambil delapan komponen dari citra yang "
          "diratakan dalam rentang [0, 1] dengan SVD penuh yang di-fit pada split latih. **Momen Hu** adalah tujuh "
          "invarian Hu[@hu1962] dari citra skala abu-abu, distabilkan dengan logaritma bertanda,"),
    ("eq", "hu", r"h^{\prime}=-\mathrm{sign}(h)\,\log_{10}\left(|h|+10^{-30}\right),"),
    ("pn", "ditambah satu kanal kedelapan bernilai konstan nol. **Momen Zernike** adalah magnitudo $|Z_{nm}|$ dari "
           "delapan orde (*n*, *m*) ∈ {(2,0), (2,2), (3,1), (3,3), (4,0), (4,2), (5,1), (5,5)} pada cakram berjari-jari "
           "14 piksel yang berpusat di tengah citra.[@teague1980,@khotanzad1990] Setiap kanal *j* lalu diskalakan dengan "
           "statistik split latih saja,"),
    ("eq", "scale", r"\tilde{x}_j=\pi\,\frac{x_j-\min_j}{\max_j-\min_j},"),
    ("pn", "dan nilai validasi serta uji ditransformasikan dengan scaler yang sama tanpa pemotongan (clipping)."),

    ("h2", "Sirkuit"),
    ("p", "Semua varian MORE-HD memakai sepuluh qubit, yaitu delapan qubit data $q_0,\\dots,q_7$ dan dua qubit readout "
          "$r_0,r_1$, tanpa qubit ancilla. Dengan $R_Y(\\theta)=e^{-i\\theta Y/2}$ dan $R_Z(\\phi)=e^{-i\\phi Z/2}$, "
          "state-nya adalah"),
    ("eq", "state", r"|\psi(\mathbf{x},\boldsymbol{\theta})\rangle=\prod_{l=1}^{L}\left[E\,U_l(\boldsymbol{\theta}_l)\right]\,\bigotimes_{i=0}^{7}R_Y(\tilde{x}_i)\,|0\rangle^{\otimes 10},"),
    ("pn", "dengan enkoding $R_Y(\\tilde{x}_i)$ pada qubit data $q_i$, $E$ blok entangling tetap (cincin CNOT "
           "$q_i\\to q_{i+1}$ dan $q_7\\to q_0$ pada qubit data, CNOT dari $q_7$ ke kedua qubit readout, dan "
           "$r_0\\to r_1$), serta $U_l$ lapisan rotasi. MORE-HD (model A) memakai $U_l=\\bigotimes_{q}R_Y(\\theta_{l,q})$ "
           "pada kesepuluh qubit dengan *L* = 3, yaitu 30 parameter. MORE-HD-C (model D) memakai "
           "$U_l=\\bigotimes_{q}R_Z(\\phi_{l,q})R_Y(\\theta_{l,q})$, yaitu 60 parameter (Gambar {fig:circuit}). Readout-nya "
           "adalah vektor ekspektasi Pauli pada kedua qubit readout,"),
    ("eq", "readout", r"\mathbf{v}(\mathbf{x})=\left(\langle\psi|P|\psi\rangle\right)_{P\in\mathcal{P}},\qquad \mathcal{P}=\{I,X,Y,Z\}^{\otimes 2}\setminus\{II\},"),
    ("pn", "dengan urutan tetap *IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ*."),
    ("fig", "circuit", "fig1_circuit.png", 5.0,
     "Sirkuit MORE-HD-C sepuluh qubit. Delapan qubit data menerima enkoding sudut $R_Y$; setiap lapisan dari tiga "
     "lapisan variasional menerapkan $R_Y(\\theta)$ dan $R_Z(\\phi)$ pada semua qubit, diikuti blok CNOT tetap. "
     "MORE-HD (A) identik tanpa gerbang $R_Z$. Ke-15 ekspektasi Pauli non-identitas dari kedua qubit readout membentuk "
     "vektor output."),
    ("p", "Dua model ablasi memisahkan jumlah parameter dari himpunan gerbang (Tabel {tab:models}). MORE-HD-60P "
          "(model B) menumpuk enam lapisan RY saja, sehingga menyamai MORE-HD-C dalam jumlah parameter terlatih (60) "
          "tetapi tetap menghasilkan state riil; model ini memiliki enam blok entangling, bukan tiga. MORE-HD-C-FixedRZ "
          "(model C) menyisipkan RZ seperti MORE-HD-C tetapi membekukan sudutnya pada nilai tak-nol yang ditentukan "
          "seed, sehingga memberi akses ke state kompleks dengan 30 parameter terlatih. Semua varian berbagi enkoding, "
          "blok entangling, qubit, readout, dan urutan observabel yang sama."),
    ("tab", "models", "Model yang dibandingkan. *Aktif* adalah jumlah dimensi readout yang tidak selalu bernilai nol.",
     [["Kode", "Model", "Lapisan variasional", "Terlatih", "Beku", "Aktif", "State"]],
     [["A", "MORE-HD", "3 × (RY + blok CNOT)", "30", "0", "9", "riil"],
      ["D", "MORE-HD-C", "3 × (RY, RZ + blok CNOT)", "60", "0", "15", "kompleks"],
      ["B", "MORE-HD-60P", "6 × (RY + blok CNOT)", "60", "0", "9", "riil"],
      ["C", "MORE-HD-C-FixedRZ", "3 × (RY, RZ beku + CNOT)", "30", "30", "15", "kompleks"],
      ["M", "MORE-REPRO", "QCNN 3 tahap (Subbab 4.5)", "91", "0", "3", "kompleks"]],
     [480, 1700, 2100, 760, 560, 640, 760]),

    ("h2", "Structural zeros pada sirkuit beramplitudo riil"),
    ("p", "**Proposisi 1.** *Misalkan* $|\\psi\\rangle$ *memiliki amplitudo riil pada basis komputasi dan P adalah "
          "hasil kali tensor matriks Pauli dengan jumlah faktor Y ganjil* $n_Y(P)$. *Maka* "
          "$\\langle\\psi|P|\\psi\\rangle=0$."),
    ("p", "*Bukti.* Tulis $Y=iJ$ dengan $J=XZ$, yang riil dan antisimetris, sedangkan *I*, *X*, dan *Z* riil dan "
          "simetris. Maka"),
    ("eq", "prop", r"P=i^{\,n_Y(P)}Q,\qquad Q^{\top}=(-1)^{n_Y(P)}Q,\qquad \langle\psi|P|\psi\rangle=i^{\,n_Y(P)}\,\psi^{\top}Q\,\psi,"),
    ("pn", "dengan *Q* riil. Untuk $n_Y(P)$ ganjil, *Q* antisimetris sehingga $\\psi^{\\top}Q\\psi=0$ untuk setiap "
           "vektor riil ψ. □"),
    ("p", "Karena $|0\\rangle$, $R_Y$, dan CNOT bernilai riil, model A dan B selalu menghasilkan state riil, dan "
          "observabel *IY, XY, YI, YX, YZ*, dan *ZY* bernilai nol untuk setiap input dan parameter. $R_Z$ bernilai "
          "kompleks, sehingga observabel tersebut *dapat* menjadi tak-nol pada model C dan D; apakah hal itu terjadi "
          "diukur secara empiris. Sembilan dimensi aktif tidak dengan sendirinya kurang untuk sepuluh kelas, karena *K* "
          "vektor satuan dapat membentuk simpleks beraturan dalam *K* − 1 dimensi; yang membatasi adalah wilayah ruang "
          "readout yang dapat dicapai sirkuit."),

    ("h2", "Pelatihan dua fase"),
    ("p", "Pelatihan mengikuti MORE[@wu2023] (Gambar {fig:pipeline}). Matriks korelasi antar-kelas dihitung dari vektor "
          "fitur rata-rata 100 sampel latih pertama setiap kelas,"),
    ("eq", "corr", r"S_{ab}=\frac{\mathrm{MSE}_{ab}}{\max_{a^{\prime}\neq b^{\prime}}\mathrm{MSE}_{a^{\prime}b^{\prime}}}\quad(a\neq b),\qquad S_{aa}=-1,"),
    ("pn", "dengan $\\mathrm{MSE}_{ab}$ selisih kuadrat rata-rata antara rata-rata kelas. Pada fase *clustering*, lima "
           "sampel latih per kelas diambil acak dan semua pasangan $\\Pi$ dari 5*K* sampel dipakai tanpa "
           "penyeimbangan. Dengan jarak kosinus $d_{\\cos}(\\mathbf{u},\\mathbf{w})=1-\\mathbf{u}\\cdot\\mathbf{w}/(\\lVert\\mathbf{u}\\rVert\\lVert\\mathbf{w}\\rVert)$, "
           "loss-nya adalah"),
    ("eq", "lclu", r"\mathcal{L}_{\mathrm{clu}}(\boldsymbol{\theta})=\frac{1}{|\Pi|}\sum_{(i,j)\in\Pi}-S_{c_i c_j}\,d_{\cos}\left(\mathbf{v}(\mathbf{x}_i),\mathbf{v}(\mathbf{x}_j)\right),"),
    ("pn", "yang menjauhkan sampel dari kelas yang tidak mirip dan mendekatkan sampel dari kelas yang sama. Label "
           "kuantum kelas *c* adalah median per komponen yang dinormalisasi dari vektor readout ternormalisasi kelima "
           "sampel clustering-nya,"),
    ("eq", "label", r"\boldsymbol{\ell}_c=n\left(\mathrm{med}\left\{n\left(\mathbf{v}(\mathbf{x}_i)\right):c_i=c\right\}\right),\qquad n(\mathbf{u})=\mathbf{u}/\lVert\mathbf{u}\rVert ."),
    ("p", "Fase *supervised* dimulai dari parameter hasil clustering dan meminimalkan rata-rata jarak kosinus seluruh "
          "*N* = 1.000*K* sampel latih ke label kelasnya,"),
    ("eq", "lsup", r"\mathcal{L}_{\mathrm{sup}}(\boldsymbol{\theta})=\frac{1}{N}\sum_{n=1}^{N}d_{\cos}\left(\mathbf{v}(\mathbf{x}_n),\boldsymbol{\ell}_{y_n}\right),"),
    ("pn", "dan sebuah sampel diprediksi sebagai kelas dengan label terdekat,"),
    ("eq", "pred", r"\hat{y}(\mathbf{x})=\underset{c}{\mathrm{arg\,min}}\;\left[d_{\cos}\left(\mathbf{v}(\mathbf{x}),\boldsymbol{\ell}_c\right)\right]."),
    ("p", "Kedua fase memakai COBYLA[@powell1994] (SciPy,[@virtanen2020] langkah awal 1,0, toleransi $10^{-4}$) dengan "
          "anggaran evaluasi fungsi objektif yang sama untuk setiap model: 900 untuk clustering dan 940 untuk "
          "pembelajaran supervised. Anggaran ini tidak diwarisi dari studi MNIST. Anggaran ditetapkan sebelum run "
          "konfirmatori melalui aturan plateau yang dibekukan sebelumnya, diterapkan pada 14 pilot konvergensi MADBase "
          "(seed 42; *K* = 3 dan 10; fitur PCA dan Zernike; batas 1.000 evaluasi) yang hanya memakai loss latih. "
          "Seluruh 14 kurva memenuhi aturan plateau pada kedua fase. Data validasi hanya dipantau, dan split uji dibaca "
          "satu kali per run setelah pelatihan selesai."),
    ("fig", "pipeline", "fig2_pipeline.png", 4.6,
     "Alur dua fase yang dipakai semua model. Langkah (3) dan (5) dioptimasi dengan COBYLA pada anggaran evaluasi "
     "yang sama; set uji resmi dari 20 penulis yang dibekukan hanya dipakai pada langkah (6)."),

    ("h2", "Ukuran hasil"),
    ("p", "Ukuran klasifikasi primer adalah akurasi uji dan macro-F1. Ukuran representasi primer adalah rasio separasi "
          "minimum antar-label kuantum,"),
    ("eq", "sep", r"\rho_{\min}=\frac{\min_{a<b}d_{\cos}\left(\boldsymbol{\ell}_a,\boldsymbol{\ell}_b\right)}{1+1/(K-1)},"),
    ("pn", "dengan penyebut berupa jarak kosinus berpasangan simpleks beraturan, yang menormalkan antar-*K* tetapi "
           "bukan nilai optimum yang dapat dicapai. Aktivasi enam observabel *Y*-ganjil $\\mathcal{Y}$ diringkas pada "
           "split validasi dengan"),
    ("eq", "yodd", r"f_{Y}=\frac{1}{N_{\mathrm{val}}}\sum_{n}\frac{\lVert\mathbf{v}_{\mathcal{Y}}(\mathbf{x}_n)\rVert^{2}}{\lVert\mathbf{v}(\mathbf{x}_n)\rVert^{2}} ."),

    ("h2", "Desain eksperimen dan analisis statistik"),
    ("p", "Track primer membandingkan A dan D pada seluruh 2 × 3 × 8 kombinasi model, metode fitur, dan *K* "
          "(240 run). Track ablasi melatih B dan C untuk *K* ∈ {3, 6, 10} dan semua metode fitur (90 run). Track "
          "referensi melatih sirkuit MORE (model M) dengan protokol kami (120 run). Bersama lima seed, total terdapat "
          "450 eksekusi konfirmatori. Sebuah analisis sensitivitas sekunder (Jalur B, 90 run) melatih ulang fase "
          "supervised A dan D pada sel ablasi, dimulai dari checkpoint clustering yang dipilih secara deterministik dari "
          "metrik validasi (Lampiran A). Untuk "
          "setiap metode fitur, *K*, dan seed *s*, estimand primernya adalah selisih berpasangan"),
    ("eq", "delta", r"\Delta_s=\mathrm{Metric}(\mathrm{D},s)-\mathrm{Metric}(\mathrm{A},s),\qquad \bar{\Delta}\pm t_{0.975,4}\,\frac{s_{\Delta}}{\sqrt{5}},"),
    ("pn", "yang diringkas dengan rata-rata, selang kepercayaan (SK) Student-*t* 95%, dan Hedges *g* "
           "berpasangan[@hedges1981] ($g=(1-3/(4n-5))\\,\\bar{\\Delta}/s_{\\Delta}$ dengan *n* = 5). Uji tingkat seed "
           "adalah uji permutasi sign-flip eksak dua sisi; dengan lima pasangan, nilai *p* terkecil yang mungkin adalah "
           "0,0625, sehingga kesimpulan bertumpu pada besar efek, SK, dan konsistensi kelima selisih. Karena sampel uji "
           "sama untuk semua seed, citra uji bukan replikasi independen lintas seed. Sebagai pemeriksaan sekunder "
           "tingkat sampel, uji eksak McNemar[@mcnemar1947] membandingkan prediksi A dan D pada sampel uji yang identik "
           "dalam setiap seed, dengan koreksi Holm[@holm1979] atas 24 kondisi dalam satu seed. Empat kontras ablasi "
           "B − A, C − A, D − B, dan D − C memakai ringkasan yang sama dengan koreksi Holm antar-kontras di setiap sel. "
           "Rencana analisis melarang klaim bahwa MORE-HD-C memperoleh manfaat dari fase kompleks kecuali kontras "
           "ablasi, separasi, dan diagnostik *Y*-ganjil saling konsisten. Perbandingan dengan studi MNIST hanya "
           "deskriptif, dan analisis yang tidak tercantum dalam rencana beku diberi label *eksploratif*."),
    ("p", "Baseline klasik (peluang 1/*K*, pengklasifikasi nearest centroid Euclidean, dan regresi logistik "
          "multinomial dengan kekuatan regularisasi dipilih melalui akurasi validasi[@pedregosa2011]) dilatih pada "
          "array delapan kanal yang persis sama dengan input sirkuit. Semua sirkuit disimulasikan secara eksak sebagai "
          "statevector complex128 tanpa shot noise menggunakan NumPy;[@harris2020] simulator diverifikasi terhadap "
          "PennyLane[@bergholm2018] pada setiap run. Sebuah run hanya masuk analisis bila dataset, snapshot, split, "
          "input, pasangan, konfigurasi, dan status evaluasi akhirnya cocok, dan tepat satu percobaan berstatus "
          "selesai."),

    ("h2", "Analisis eksploratif"),
    ("p", "Setelah hasil konfirmatori diketahui, kami menjalankan tiga analisis tambahan dengan prosedur yang sama "
          "seperti pada studi MNIST. Pertama, parameter akhir dari 180 run di sel ablasi (model A–D, *K* ∈ {3, 6, 10}) "
          "dievaluasi ulang. (i) *Linear probe*:[@alain2016] regresi logistik multinomial dilatih pada vektor readout "
          "ternormalisasi sampel latih dan diuji pada sampel uji, sekali dengan ke-15 observabel dan sekali hanya dengan "
          "sembilan observabel *Y*-genap; akurasinya mengestimasi informasi kelas yang dapat didekode secara linear, "
          "terlepas dari aturan label terdekat. (ii) *Jacobian readout*: untuk 200 input uji yang seimbang, turunan ke-15 "
          "readout terhadap setiap sudut terlatih dihitung eksak dengan aturan parameter-shift;[@mitarai2018,@schuld2019] "
          "rank-nya menghitung arah output independen, dan effective rank-nya[@roy2007] (eksponensial entropi nilai "
          "singular ternormalisasi) mengukur seberapa merata arah itu dipakai. (iii) *Konvergensi*: sebuah run dianggap "
          "masih membaik bila loss terbaiknya turun lebih dari 2% total perbaikannya pada 100 evaluasi terakhir suatu "
          "fase. Kedua, kami menghitung porsi galat dari setiap pasangan kelas pada *K* = 10 untuk menemukan glif yang "
          "paling sering tertukar. Ketiga, ringkasan utama dibandingkan secara deskriptif dengan studi MNIST."),

    # ================================================================== 4 Hasil
    ("h1", "Hasil"),
    ("h2", "Kelengkapan run dan kualitas data"),
    ("p", f"Seluruh {N['admission']['main_runs']} run konfirmatori, {N['jalurB_n']} run Jalur B, dan "
          f"{N['admission']['classical_fits']} fit baseline klasik lolos pemeriksaan admission tanpa galat. Setiap run "
          "selesai pada percobaan pertama, tidak ada run ganda, tidak ada label kuantum yang degenerate, tidak ada output "
          "uji dengan norma nol, dan semua run memakai tepat 900 dan 940 evaluasi objektif. Selisih absolut maksimum "
          f"antara simulator dan PennyLane adalah ${f(Q['backend_max_abs_diff'] * 1e15, 1)}\\times10^{{-15}}$. Pemeriksaan "
          "input berpasangan cocok untuk seluruh run ablasi dan referensi. Sesuai Proposisi 1, model A dan B mengaktifkan "
          "tepat 9 dimensi readout pada setiap run, sedangkan model C dan D mengaktifkan ke-15 dimensi."),

    ("h2", "MORE-HD-C dibandingkan MORE-HD"),
    ("p", f"Gambar {{fig:acc}} menampilkan akurasi uji terhadap *K*, dan Tabel {{tab:primary}} memuat ke-24 "
          f"perbandingan berpasangan. MORE-HD-C memiliki akurasi rata-rata lebih tinggi daripada MORE-HD pada "
          f"{P['n_pos']} dari {P['n']} kondisi, dengan rata-rata selisih berpasangan {s(P['mean'])} poin persentase (pp); "
          f"rata-rata akurasi seluruh kondisi naik dari {f(MM['A'])}% menjadi {f(MM['D'])}%. Kenaikan terbesar terjadi pada "
          f"fitur PCA ({s(N['primary_acc_by_feature']['PCA'])} pp), lebih kecil pada momen Zernike "
          f"({s(N['primary_acc_by_feature']['ZERNIKE'])} pp) dan momen Hu ({s(N['primary_acc_by_feature']['HU'])} pp). "
          f"Dua kondisi bernilai negatif (PCA *K* = 3 dan Hu *K* = 4), dan SK keduanya memuat nol. SK 95% berada di atas "
          f"nol pada {P['ci_pos']} kondisi (PCA *K* = 6 dan 7; Hu *K* = 7) dan tidak ada yang berada di bawah nol. "
          f"Hedges *g* melebihi 0,8 pada {P['n_g_gt_08']} kondisi. Macro-F1 mengikuti pola yang sama (rata-rata "
          f"{s(F1['mean'])} pp, positif pada {F1['n_pos']} dari 24 kondisi), sedangkan rasio separasi minimum label "
          f"kuantum hanya sedikit meningkat (rata-rata {s(SEP['mean'], 3)}, positif pada {SEP['n_pos']} kondisi, tanpa "
          "SK yang tidak memuat nol)."),
    ("fig", "acc", "fig3_accuracy_vs_K.png", 5.0,
     "Akurasi uji MORE-HD (A), MORE-HD-C (D), dan reproduksi MORE di dalam protokol (M) terhadap jumlah kelas pada "
     "MADBase (rata-rata lima seed; pita menunjukkan ±1 SD). Garis titik-titik adalah tingkat peluang."),
    ("tab", "primary", "Akurasi uji (%) MORE-HD (A) dan MORE-HD-C (D) pada MADBase, rata-rata ± SD lima seed, serta "
     "selisih berpasangan Δ = D − A dalam poin persentase dengan SK 95% dan Hedges *g*. Tanda bintang menandai SK yang "
     "tidak memuat nol.",
     [["Fitur", "*K*", "A", "D", "Δ", "SK 95%", "*g*"]],
     primary_rows(),
     [900, 400, 1250, 1250, 800, 1500, 600]),
    ("p", f"Gambar {{fig:forest}} memperlihatkan selisih per seed. Variasi antar-seed di dalam satu kondisi besar, "
          f"sehingga tidak ada uji sign-flip per kondisi yang dapat mencapai *p* < 0,05 dengan lima seed; setelah "
          f"koreksi Holm semua nilai *p* bernilai 1. Dari 120 selisih tingkat seed, {N['primary_seed_deltas_pos']} "
          f"positif. Pada tingkat sampel, uji McNemar menolak kesamaan tingkat galat pada {MC['sig']} dari {MC['n']} "
          f"perbandingan seed-kondisi setelah koreksi Holm; {MC['sig_d_better']} di antaranya menguntungkan MORE-HD-C "
          f"dan {MC['sig_a_better']} menguntungkan MORE-HD. Secara eksploratif, {P['n_pos']} dari 24 selisih rata-rata "
          f"positif kecil kemungkinannya terjadi bila arah efek acak (uji binomial dua sisi "
          f"{pval(N['primary_acc_sign_test_p'])}); karena kondisi-kondisi saling bergantung melalui desain bersarang, "
          "angka ini bersifat deskriptif."),
    ("fig", "forest", "fig4_forest_D_minus_A.png", 5.0,
     "Selisih akurasi berpasangan D − A per kondisi pada MADBase. Belah ketupat menunjukkan rata-rata lima seed "
     "(terisi bila SK 95% tidak memuat nol), garis horizontal SK 95%, dan titik abu-abu selisih tiap seed."),

    ("h2", "Ablasi: kapasitas versus fase kompleks"),
    ("p", f"Gambar {{fig:abl}} dan Tabel {{tab:abl}} menyajikan ablasi pada *K* ∈ {{3, 6, 10}}. Sirkuit riil 60 "
          f"parameter B adalah model paling akurat pada {N['best_model_count_ablation_cells'].get('B', 0)} dari sembilan "
          f"sel, dengan rata-rata akurasi {f(ABC['B'])}%, dibandingkan {f(ABC['D'])}% untuk D, {f(ABC['C'])}% untuk C, dan "
          f"{f(ABC['A'])}% untuk A. Menambah 30 parameter RY riil pada MORE-HD (B − A) menaikkan akurasi rata-rata "
          f"{f(AB['B−A']['mean'])} pp, positif pada kesembilan sel dan dengan SK di atas nol pada {AB['B−A']['ci_pos']} "
          f"sel. Menambah akses kompleks dengan 30 parameter terlatih yang sama (C − A) hanya menaikkan akurasi "
          f"{f(AB['C−A']['mean'])} pp, positif pada {AB['C−A']['n_pos']} sel dan dengan SK di atas nol pada "
          f"{AB['C−A']['ci_pos']} sel. Melatih sudut RZ alih-alih membekukannya (D − C) mengubah akurasi rata-rata "
          f"{s(AB['D−C']['mean'])} pp tanpa SK yang tidak memuat nol. Pada jumlah parameter terlatih yang sama, MORE-HD-C "
          f"kurang akurat daripada sirkuit riil pada kesembilan sel (D − B = {f(AB['D−B']['mean'])} pp; SK di bawah nol "
          f"pada {AB['D−B']['ci_neg']} sel). Setelah koreksi Holm, tidak ada uji sign-flip tingkat seed yang signifikan "
          f"(*p* terkoreksi terkecil = {f(N['ablation_holm_min'], 2)}), sesuai keterbatasan lima pasangan."),
    ("fig", "abl", "fig5_ablation.png", 5.0,
     "Ablasi pada *K* ∈ {3, 6, 10} pada MADBase. Batang menunjukkan rata-rata akurasi uji dan titik menunjukkan kelima "
     "seed. B menambahkan 30 parameter riil pada A; C menambahkan fase RZ beku pada A; D menambahkan fase RZ terlatih "
     "pada A."),
    ("tab", "abl", "Kontras ablasi untuk akurasi uji (pp) pada MADBase: rata-rata selisih berpasangan lima seed dan "
     "SK 95%. Tanda bintang menandai SK yang tidak memuat nol.",
     [["Fitur", "*K*", "B − A", "C − A", "D − B", "D − C"]],
     ablation_rows(),
     [900, 400, 1475, 1475, 1475, 1475]),
    ("p", "Dengan demikian, syarat rencana analisis untuk klaim yang spesifik pada fase kembali tidak terpenuhi. "
          "Observabel *Y*-ganjil aktif pada C dan D, tetapi kontras akses kompleks dan kontras fase terlatih tidak "
          "menunjukkan manfaat akurasi yang konsisten, sedangkan sirkuit riil dengan jumlah parameter yang sama lebih "
          "akurat. Peningkatan MORE-HD-C atas MORE-HD pada MADBase, seperti pada MNIST, terutama dijelaskan oleh tambahan "
          "kapasitas parameter terlatih."),

    ("h2", "Diagnostik readout"),
    ("p", f"Fraksi norma *Y*-ganjil rata-rata {f(N['yodd_mean']['D'], 2)} untuk MORE-HD-C dan "
          f"{f(N['yodd_mean']['C'], 2)} untuk sirkuit fase beku, dan tepat nol untuk A dan B (Gambar {{fig:readout}}a). "
          "Jadi, sebagian besar sinyal readout C dan D memang berpindah ke dimensi yang baru tersedia. Aktivasi ini "
          "tidak berkaitan kuat dengan akurasi: di dalam kondisi, korelasi peringkat antara $f_Y$ dan akurasi MORE-HD-C "
          f"adalah ρ Spearman = {f(SYO['rho'], 2)} ({pval(SYO['p'])}; eksploratif). Sebaliknya, separasi label sejalan "
          f"dengan akurasi (Gambar {{fig:readout}}b): lintas run, korelasi Spearman antara $\\rho_{{\\min}}$ dan akurasi "
          f"adalah {f(SP['A']['rho'], 2)} untuk MORE-HD, {f(SP['D']['rho'], 2)} untuk MORE-HD-C, {f(SP['B']['rho'], 2)} "
          f"untuk MORE-HD-60P, dan {f(SP['C']['rho'], 2)} untuk sirkuit fase beku, dan {f(SP['within_all']['rho'], 2)} "
          f"bila digabung di dalam kondisi ({pval(SP['within_all']['p'])}; eksploratif). Pada sel ablasi, rata-rata rasio "
          f"separasi adalah A {f(SEPA['A'], 3)}, D {f(SEPA['D'], 3)}, C {f(SEPA['C'], 3)}, dan B {f(SEPA['B'], 3)}, dengan "
          "B kembali tertinggi. Pola ini mendukung pandangan Wu *dkk.*[@wu2023] bahwa label yang berdesakan membatasi "
          "akurasi, dan menunjukkan bahwa tambahan kapasitas terlatih memisahkan label lebih efektif daripada akses ke "
          "dimensi imajiner."),
    ("fig", "readout", "fig6_readout.png", 5.0,
     "(a) Fraksi norma *Y*-ganjil per run pada MADBase: selalu nol untuk model riil A dan B, dan besar untuk C dan D. "
     "(b) Akurasi uji terhadap rasio separasi minimum label kuantum untuk A dan D (seluruh 240 run primer)."),

    ("h2", "Perbandingan dengan MORE di dalam protokol"),
    ("p", "Wu *dkk.* tidak melaporkan akurasi MORE pada MADBase,[@wu2023] sehingga tidak ada nilai acuan literatur "
          "untuk dataset ini. Sebagai gantinya, sirkuit MORE (model M) dibangun ulang gerbang demi gerbang dari kode "
          "publiknya (delapan qubit, feature map *Z* dengan dua repetisi, jaringan konvolusional kuantum[@cong2019] tiga "
          "tahap dengan 91 parameter, dan satu qubit readout yang diukur pada *X*, *Y*, dan *Z*) dan dilatih dengan data, "
          "split, fitur, pasangan clustering, matriks korelasi, aturan label, anggaran optimizer, dan set uji yang persis "
          f"sama, tanpa loss adjuster *R*. Tabel {{tab:more}} menunjukkan bahwa M mencapai akurasi rata-rata "
          f"{f(N['more_M_mean_acc'])}%. MORE-HD lebih tinggi daripada M pada {N['more_A_minus_M']['n_pos']} dari 24 "
          f"kondisi (rata-rata {s(N['more_A_minus_M']['mean'])} pp; SK di atas nol pada {N['more_A_minus_M']['ci_pos']}), "
          f"dan MORE-HD-C pada {N['more_D_minus_M']['n_pos']} dari 24 kondisi (rata-rata "
          f"{s(N['more_D_minus_M']['mean'])} pp; SK di atas nol pada {N['more_D_minus_M']['ci_pos']}). Selisih negatif "
          "hanya muncul pada momen Hu (A − M pada *K* = 3 dan 8; D − M pada *K* = 3 dan 4), dan SK-nya memuat nol. "
          "Karena M memakai 91 "
          "parameter dengan anggaran evaluasi yang sama, perbandingan ini menguji desain sirkuit dan readout secara "
          "gabungan, bukan dimensi readout semata."),
    ("tab", "more", "Perbandingan di dalam protokol dengan reproduksi MORE (M) pada MADBase. Akurasi M dalam % "
     "(rata-rata ± SD) dan selisih berpasangan dalam pp dengan SK 95%. Tanda bintang menandai SK yang tidak memuat nol.",
     [["Fitur", "*K*", "M", "A − M", "D − M"]],
     more_rows(),
     [900, 400, 1300, 2300, 2300]),

    ("h2", "Baseline klasik"),
    ("p", f"Tabel {{tab:base}} menempatkan model kuantum di samping pengklasifikasi klasik yang dilatih pada input "
          f"delapan kanal yang identik. Dirata-ratakan atas *K* dan metode fitur, regresi logistik mencapai "
          f"{f(BM['LR'])}% dan nearest centroid {f(BM['NC'])}%, dibandingkan {f(BM['D'])}% untuk MORE-HD-C, "
          f"{f(BM['A'])}% untuk MORE-HD, dan {f(BM['M'])}% untuk M (peluang {f(BM['chance'])}%). Tidak ada satu kondisi "
          "pun di mana MORE-HD-C menyamai nearest centroid atau regresi logistik. Model kuantum jelas mempelajari "
          "struktur kelas, tetapi tidak mendekati baseline linear pada tugas ini."),
    ("tab", "base", "Akurasi uji (%) baseline klasik dan model kuantum pada input identik di MADBase (rata-rata lima "
     "seed). NC, nearest centroid; LR, regresi logistik. “Rata-rata” merata-ratakan *K* = 3, …, 10.",
     [["Fitur", "", "Peluang", "NC", "LR", "M", "A", "D"]],
     baseline_rows(),
     [900, 900, 900, 850, 850, 850, 850, 850]),

    ("h2", "Analisis per kelas"),
    ("p", "Analisis 6↔9 yang ditetapkan sebelumnya diwarisi dari studi MNIST, tempat kedua digit saling berupa "
          f"rotasi. Pada *K* = 10, kekeliruan antara kelas 6 dan 9 menyumbang rata-rata "
          f"{f(SIX['share_of_errors_mean_ABCD'] * 100)}% galat model A–D (PCA "
          f"{f(SIX['share_by_feature']['PCA'] * 100)}%, Hu {f(SIX['share_by_feature']['HU'] * 100)}%, Zernike "
          f"{f(SIX['share_by_feature']['ZERNIKE'] * 100)}%), dan menggabungkan kedua kelas hanya menaikkan akurasi "
          f"{f(SIX['merged_gain_ABCD_pp'])} pp; kesimpulan sebelumnya tidak berubah. Pada aksara Arab-Indik, ٦ dan ٩ "
          "bukan rotasi satu sama lain, sehingga kami menelusuri seluruh pasangan kelas secara eksploratif "
          f"(Gambar {{fig:conf}}). Dengan fitur invarian-rotasi, pasangan yang paling sering tertukar adalah 7↔8 (٧↔٨; "
          f"Zernike {share('ZERNIKE', '7↔8')}%, Hu {share('HU', '7↔8')}% dari galat) dan 0↔5 (٠↔٥; Hu "
          f"{share('HU', '0↔5')}%), jauh di atas porsi seragam 2,2%. Glif ٧ dan ٨ secara kasar saling berupa rotasi "
          f"180°, sedangkan dengan PCA, yang tidak invarian terhadap rotasi, pasangan itu hanya menyumbang "
          f"{f(N['confusion_pair_78']['PCA'] * 100)}% galat. Glif ٠ dan ٥ sama-sama berupa goresan kecil yang membulat. "
          "PCA paling sering menukar 6↔9 dan 8↔9. Pola ini konsisten dengan sifat invarian fitur, tetapi tidak "
          "membuktikan mekanismenya."),
    ("fig", "conf", "fig8_confusion_pairs.png", 5.0,
     "Enam pasangan kelas dengan porsi galat terbesar pada *K* = 10 untuk model A–D (gabungan lima seed; eksploratif). "
     "Batang oranye menandai pasangan ٧↔٨ dan ٠↔٥; garis titik-titik adalah porsi seragam 1/45."),

    ("h2", "Mengapa dimensi tambahan kurang membantu: analisis eksploratif"),
]

if MECH:
    T6 = MECH["table"]
    PP = MECH["probe_pairs"]
    BLOCKS += [
        ("p", f"Evaluasi ulang {MECH['n_runs']} sirkuit terlatih di sel ablasi mereproduksi setiap akurasi uji yang "
              f"tercatat secara persis (selisih maksimum {f(MECH['max_abs_acc_repro_diff'], 1)}). Tabel {{tab:mech}} dan "
              f"Gambar {{fig:cap}} merangkum hasilnya."),
        ("p", f"**Informasi dalam readout.** Dimensi *Y*-ganjil membawa informasi kelas: menghapus keenam observabel itu "
              f"menurunkan akurasi linear probe MORE-HD-C sebesar {f(MECH['probe_drop_without_yodd_D']['mean'])} pp (SK 95% "
              f"{ci(MECH['probe_drop_without_yodd_D'])}, 45 run) dan akurasi label terdekatnya sebesar "
              f"{f(MECH['label_acc_drop_without_yodd_D']['mean'])} pp {ci(MECH['label_acc_drop_without_yodd_D'])}. Namun, "
              f"total informasi yang dapat didekode secara linear hampir sama pada ketiga model yang lebih besar: akurasi "
              f"probe {f(T6['D']['probe'])}% untuk MORE-HD-C, {f(T6['C']['probe'])}% untuk sirkuit fase beku, dan "
              f"{f(T6['B']['probe'])}% untuk MORE-HD-60P (D − B berpasangan = {s(PP['D-B']['mean'])} pp {ci(PP['D-B'])}), "
              f"dibandingkan {f(T6['A']['probe'])}% untuk MORE-HD. Akses kompleks pada jumlah parameter tetap (C − A = "
              f"{s(PP['C-A']['mean'])} pp {ci(PP['C-A'])}) menaikkan informasi yang dapat didekode sebesar 30 parameter riil "
              f"tambahan (B − A = {s(PP['B-A']['mean'])} pp {ci(PP['B-A'])})."),
        ("p", f"**Konversi menjadi akurasi.** Perbedaannya terletak pada seberapa banyak informasi itu diubah menjadi "
              f"akurasi oleh aturan label terdekat. Selisih antara akurasi probe dan akurasi label terdekat adalah "
              f"{f(T6['B']['gap'])} pp untuk MORE-HD-60P, tetapi {f(T6['D']['gap'])} pp untuk MORE-HD-C, "
              f"{f(T6['C']['gap'])} pp untuk sirkuit fase beku, dan {f(T6['A']['gap'])} pp untuk MORE-HD. Mengganti label "
              f"kuantum lima sampel dengan median atas seluruh sampel latih menaikkan akurasi setiap model tetapi tetap "
              f"mempertahankan urutan B ({f(T6['B']['oracle'])}%) > C ({f(T6['C']['oracle'])}%) ≈ D "
              f"({f(T6['D']['oracle'])}%) > A ({f(T6['A']['oracle'])}%), sehingga kehilangan ini tidak hanya disebabkan "
              "oleh estimasi label yang bising."),
        ("p", f"**Kapasitas terlatih.** Jacobian peta readout menunjukkan bahwa dua cara menambah 30 parameter tidak "
              f"setara. Sudut RY MORE-HD-60P semuanya independen (rank 60), sedangkan rank MORE-HD-C pada parameter "
              f"terlatih adalah {MECH['rank_D_values'][0]}–{MECH['rank_D_values'][-1]} (rata-rata "
              f"{f(T6['D']['rank'])}) dan tepat {MECH['random_rank']['D']:.0f} pada parameter acak. Delapan arah yang "
              "selalu hilang berasal dari gerbang RZ pada qubit data di lapisan terakhir, yang tidak dapat memengaruhi "
              "readout; norma kolom Jacobian-nya tidak melebihi "
              f"${f(MECH['last_layer_data_rz_colnorm_max'] * 1e15, 1)}\\times10^{{-15}}$ pada semua run:"),
        ("p", "**Proposisi 2.** *Pada MORE-HD-C (dan MORE-HD-C-FixedRZ), state tereduksi qubit readout tidak bergantung "
              "pada sudut RZ yang diterapkan pada delapan qubit data di lapisan variasional terakhir.*"),
        ("p", "*Bukti.* Misalkan $D=\\bigotimes_{q=0}^{7}R_Z(\\phi_q)\\otimes I_r$ diterapkan tepat sebelum blok "
              "entangling terakhir $E$. Semua gerbang dalam $E$ adalah CNOT, sehingga $E$ merupakan permutasi $\\Pi$ "
              "pada basis komputasi dan $ED=\\tilde{D}E$ dengan matriks diagonal $\\tilde{D}=\\Pi D\\Pi^{\\top}$. Tidak "
              "ada CNOT dalam $E$ yang memiliki kontrol di qubit readout dan target di qubit data, sehingga bit data dari "
              "$\\Pi^{\\top}|d,r\\rangle$ hanya bergantung pada *d* dan $\\tilde{D}=D^{\\prime}\\otimes I_r$ dengan "
              "$D^{\\prime}$ uniter pada register data. Uniter pada register yang di-trace out tidak mengubah state "
              "readout tereduksi. □"),
        ("p", f"Sudut RZ yang tersisa menambah lebih sedikit arah independen daripada sudut RY. Effective rank "
              f"Jacobian[@roy2007] naik dari {f(T6['A']['eff'])} untuk MORE-HD menjadi {f(T6['B']['eff'])} untuk MORE-HD-60P, "
              f"tetapi hanya {f(T6['D']['eff'])} untuk MORE-HD-C (D − B berpasangan = {f(MECH['eff_rank_D_minus_B']['mean'])} "
              f"{ci(MECH['eff_rank_D_minus_B'])}), dengan urutan yang sama pada parameter acak; kolom RZ menyumbang "
              f"{f(MECH['rz_colnorm_share_D'] * 100, 0)}% kuadrat norma Jacobian MORE-HD-C. Di dalam kondisi, effective "
              f"rank yang lebih besar sejalan dengan akurasi yang lebih tinggi (ρ Spearman = "
              f"{f(MECH['spearman_effrank_acc_within']['rho'], 2)}, {pval(MECH['spearman_effrank_acc_within']['p'])}, "
              f"180 run; Gambar {{fig:cap}}c) dan dengan selisih probe-ke-aturan yang lebih kecil (ρ = "
              f"{f(MECH['spearman_effrank_gap_within']['rho'], 2)}, {pval(MECH['spearman_effrank_gap_within']['p'])})."),
        ("p", f"**Anggaran optimasi.** Pada akhir anggaran, MORE-HD-60P masih membaik pada "
              f"{f(MECH['fisher_sup_B_vs_D']['B'], 0)}% run fase supervised dibandingkan "
              f"{f(MECH['fisher_sup_B_vs_D']['D'], 0)}% untuk MORE-HD-C (uji eksak Fisher "
              f"{pval(MECH['fisher_sup_B_vs_D']['p'])}), dan pada fase clustering {f(MECH['fisher_clu_B_vs_D']['B'], 0)}% "
              f"berbanding {f(MECH['fisher_clu_B_vs_D']['D'], 0)}% ({pval(MECH['fisher_clu_B_vs_D']['p'])}). Tidak ada "
              "indikasi bahwa MORE-HD-C lebih terhambat anggaran daripada sirkuit riil yang mengunggulinya; berbeda dengan "
              "MNIST, selisih proporsi ini tidak signifikan."),
        ("tab", "mech", "Ringkasan mekanistik eksploratif atas 45 run per model di sel ablasi MADBase (rata-rata). Rank "
         "dan effective rank Jacobian readout pada parameter terlatih; probe = regresi logistik multinomial pada readout "
         "15 dimensi yang dinormalisasi (atau sembilan komponen *Y*-genapnya); selisih = probe dikurangi akurasi label "
         "terdekat; kolom terakhir = proporsi run yang masih membaik pada 100 evaluasi supervised terakhir.",
         [["Model", "Terlatih", "Rank", "Eff. rank", "Probe 15", "Probe 9", "Akurasi", "Selisih", "Membaik"]],
         mechanism_rows(),
         [560, 760, 600, 760, 820, 760, 820, 600, 920]),
        ("fig", "cap", "fig7_capacity.png", 5.0,
         "Analisis mekanistik eksploratif pada sel ablasi MADBase. (a) Akurasi linear probe dari readout 15 dimensi "
         "(arsir), akurasi probe yang hanya memakai sembilan dimensi *Y*-genap (garis hitam, C dan D), dan akurasi label "
         "kuantum terdekat (padat); rata-rata ± SE. (b) Effective rank Jacobian readout pada parameter terlatih; angka di "
         "bawah kode menunjukkan jumlah parameter terlatih dan, untuk D, jumlah parameter efektif. (c) Effective rank "
         "terhadap akurasi relatif terhadap rata-rata sel."),
    ]

BLOCKS += [
    ("h2", "Perbandingan deskriptif dengan studi MNIST"),
    ("p", f"Tabel {{tab:cross}} dan Gambar {{fig:cross}} membandingkan ringkasan utama dengan studi MNIST kami. "
          "Perbandingan ini deskriptif karena domain, penulis, cara akuisisi, susunan split (pada MADBase set uji tetap "
          "dan writer-disjoint), dan anggaran optimizer berbeda. Besar efek primer hampir identik "
          f"({s(X['mnist']['DA']['mean'])} pp pada MNIST dan {s(X['madbase']['DA']['mean'])} pp pada MADBase), dan "
          "keempat kontras ablasi memiliki arah dan besaran yang serupa. Sebaliknya, kondisi mana yang menunjukkan "
          f"kenaikan besar tidak berulang: korelasi Spearman selisih D − A per kondisi antara kedua dataset adalah "
          f"{f(X['cell_DA_spearman']['rho'], 2)}, dan tanda selisih sama pada {X['cell_DA_same_sign']} dari 24 kondisi. "
          "Semua model, termasuk baseline klasik, lebih akurat pada MADBase daripada pada MNIST dalam protokol "
          "masing-masing, tetapi urutan B > C ≈ D > A > M dan kesenjangan terhadap baseline klasik tetap sama."),
    ("tab", "cross", "Ringkasan deskriptif studi MNIST dan MADBase dengan protokol yang sejajar (rata-rata atas sel; "
     "selisih dalam pp). Bukan perbandingan inferensial; lihat teks untuk perbedaan protokol.",
     [["Ukuran", "MNIST", "MADBase"]],
     cross_rows(),
     [3700, 2100, 2100]),
    ("fig", "cross", "fig9_cross_dataset.png", 5.0,
     "Perbandingan deskriptif dengan studi MNIST. (a) Selisih D − A per kondisi pada kedua dataset; garis putus-putus "
     "adalah identitas. (b) Rata-rata kontras primer dan ablasi pada kedua dataset."),

    ("h2", "Biaya komputasi"),
    ("p", f"Penambahan RZ hampir tidak menambah biaya. Pada mesin konfirmatori (Intel Core i3-12100F, satu thread per "
          f"run), pembentukan uniter sirkuit memerlukan median {f(BU['MORE-HD'])} ms untuk MORE-HD dan "
          f"{f(BU['MORE-HD-C'])} ms untuk MORE-HD-C, dibandingkan {f(BU['MORE-HD-60P'])} ms untuk MORE-HD-60P. Median "
          f"waktu per run adalah {f(RT['A'], 0)} s untuk MORE-HD, {f(RT['D'], 0)} s untuk MORE-HD-C, {f(RT['B'], 0)} s "
          f"untuk MORE-HD-60P, dan {f(RT['M'], 0)} s untuk M. Waktu ini hanya deskriptif, karena run dijalankan enam "
          "hingga delapan proses paralel pada mesin yang sama."),

    # ================================================================== 5 Pembahasan
    ("h1", "Pembahasan"),
    ("p", "Hasil pada MADBase mereplikasi ketiga kesimpulan studi MNIST. Pertama, penghapusan batasan state riil "
          "bekerja sesuai rancangan pada tingkat readout: keenam structural zeros MORE-HD menjadi aktif dan sebagian "
          "besar sinyal output memakainya. Kedua, MORE-HD-C lebih akurat daripada MORE-HD pada hampir semua kondisi, "
          "dengan besar efek rata-rata yang praktis sama dengan MNIST. Ketiga, ablasi menunjukkan bahwa kenaikan ini "
          "tidak spesifik pada fase kompleks: sirkuit riil dengan 60 parameter terlatih yang sama lebih akurat, "
          "membekukan fase hampir tidak mengurangi apa pun, dan menambah fase pada jumlah parameter tetap hanya memberi "
          "kenaikan kecil yang tidak konsisten. Replikasi ini memperkuat kesimpulan studi MNIST karena terjadi pada "
          "aksara, penulis, dan split yang berbeda, dengan anggaran optimizer yang dikalibrasi ulang, dan dengan "
          "rencana analisis yang dibekukan sebelum hasil tersedia."),
    ("p", "Analisis eksploratif juga menghasilkan penjelasan yang sama. Dimensi *Y*-ganjil membawa informasi kelas, "
          "tetapi sirkuit riil 60 parameter menyimpan jumlah informasi yang dapat didekode secara linear yang setara "
          "dalam sembilan dimensi. Yang membedakan model adalah konversi informasi itu menjadi keputusan label "
          "terdekat, dan konversi ini sejalan dengan kapasitas terlatih: delapan dari 30 sudut RZ MORE-HD-C terbukti "
          "inert (Proposisi 2), dan sisanya menambah jauh lebih sedikit arah output independen dibandingkan RY dan blok "
          "entangling tambahan pada MORE-HD-60P. Anggaran evaluasi tidak menjelaskan keunggulan sirkuit riil. Dua "
          "rekomendasi praktis dari studi MNIST tetap berlaku: perbandingan arsitektur VQC perlu ablasi yang menjaga "
          "jumlah parameter terlatih, dan jumlah parameter perlu diperiksa dari gerbang yang inert secara struktural."),
    ("p", "Ada dua perbedaan dari MNIST. Pertama, kenaikan pada tingkat kondisi tidak berulang: kondisi dengan selisih "
          "D − A terbesar berbeda antar-dataset, sehingga besar efek per kondisi sebaiknya tidak ditafsirkan berlebihan "
          "dengan hanya lima seed. Kedua, galat fitur invarian-rotasi pada MADBase terkonsentrasi pada glif yang saling "
          "berupa rotasi atau sama-sama membulat, yaitu ٧↔٨ dan ٠↔٥, bukan pada 6↔9. Temuan ini menunjukkan bahwa "
          "pilihan fitur delapan kanal perlu mempertimbangkan geometri aksara; pada aksara Arab-Indik, fitur yang "
          "mempertahankan orientasi, seperti PCA, tampak lebih sesuai daripada momen invarian. Dengan PCA, semua model "
          "mencapai akurasi tertinggi dan MORE-HD-C memperoleh kenaikan terbesar."),
    ("p", "Secara absolut, kedua varian MORE-HD mengungguli sirkuit MORE yang dilatih dengan protokol sama, tetapi "
          "regresi logistik dan bahkan nearest centroid pada delapan kanal yang sama jauh lebih akurat. Hal ini sejalan "
          "dengan pandangan bahwa keunggulan kuantum tidak semestinya menjadi harapan bawaan untuk data "
          "klasik.[@huang2021,@schuldkilloran2022,@cerezo2022] Kontribusi studi ini terletak pada pemahaman desain "
          "readout dan kapasitas VQC multi-kelas, bukan pada klaim keunggulan atas model klasik."),
    ("p", "Penelitian ini memiliki keterbatasan. Setiap kondisi hanya memiliki lima seed, sehingga inferensi per kondisi "
          "bersifat kasar dan tidak ada uji tingkat seed yang dapat mencapai *p* < 0,05. Set uji berasal dari 20 penulis "
          "yang sama untuk semua seed, sehingga generalisasi ke populasi penulis lain belum diukur, dan 12 grup duplikat "
          "identik melintasi split latih dan uji resmi. Model B disamakan dengan D dalam jumlah parameter terlatih "
          "tetapi tidak dalam kedalaman. Urutan kelas kumulatif mencampurkan jumlah kelas dengan identitas digit yang "
          "ditambahkan, sehingga tren lintas *K* hanya berlaku untuk urutan {0, …, *K* − 1}. Hasil supervised dilaporkan "
          "pada anggaran evaluasi yang sama, bukan pada konvergensi; analisis Jalur B (Lampiran A) menunjukkan bahwa "
          "pemilihan checkpoint clustering dari data validasi dapat menaikkan akurasi kedua model. Semua hasil berasal dari "
          "simulasi statevector tanpa noise. Analisis mekanistik dan analisis pasangan kelas dirancang setelah hasil "
          "konfirmatori diketahui, dan perbandingan dengan MNIST bersifat deskriptif. Lisensi redistribusi MADBase belum "
          "dinyatakan, sehingga data mentah tidak dapat dibagikan bersama kode."),

    # ================================================================== 6 Kesimpulan
    ("h1", "Kesimpulan"),
    ("p", f"Kami mereplikasi studi ablasi MORE-HD dan MORE-HD-C pada digit tulisan tangan Arab-Indik MADBase dengan "
          f"protokol yang dibekukan sebelumnya, split yang terpisah menurut penulis, dan anggaran optimizer yang "
          f"dikalibrasi ulang. Pada 24 kondisi dan lima seed berpasangan, MORE-HD-C menaikkan akurasi uji rata-rata "
          f"{f(P['mean'])} pp dan mengaktifkan ke-15 dimensi readout. Seperti pada MNIST, ablasi menunjukkan bahwa kenaikan "
          f"itu berasal dari tambahan kapasitas parameter terlatih, bukan dari fase kompleks: sirkuit riil dengan "
          f"jumlah parameter yang sama lebih akurat pada kesembilan sel ablasi, dan melatih fase hampir tidak menambah "
          "apa pun dibanding membekukannya. Analisis eksploratif mereplikasi penjelasannya: readout kompleks tidak "
          "membawa informasi kelas lebih banyak daripada readout riil, delapan sudut RZ tidak dapat memengaruhi readout, "
          "dan sisanya hanya menambah sedikit arah independen. Penelitian lanjutan sebaiknya menghapus gerbang yang "
          "inert, menguji sirkuit riil dan kompleks dengan kedalaman yang disamakan, memakai fitur yang mempertahankan "
          "orientasi untuk aksara yang glifnya saling berupa rotasi, menguji populasi penulis yang lebih luas dan "
          "perangkat keras bernoise, serta selalu menyertakan baseline klasik."),

    ("h1n", "Ketersediaan Data dan Kode"),
    ("pn", "Kode, protokol beku, rencana analisis statistik, manifest split, dan workbook hasil terkonsolidasi tersedia "
           "di repositori penulis [URL repositori ditambahkan sebelum pengiriman naskah]. Citra MADBase tidak "
           "didistribusikan ulang karena halaman resmi tidak menyatakan lisensi redistribusi; dataset dapat diunduh dari "
           "situs resminya[@madbaseweb] dan diverifikasi terhadap hash SHA-256 snapshot yang dicatat di repositori."),
    ("h1n", "Ucapan Terima Kasih"),
    ("pn", "[Ucapan terima kasih dan informasi pendanaan.]"),

    # ================================================================== Lampiran A Jalur B
    ("app", "Sensitivitas terhadap Pemilihan Checkpoint Clustering (Jalur B)"),
    ("p", "Pada analisis utama, fase supervised dimulai dari parameter clustering pada titik akhir COBYLA. Sebagai "
          "analisis sensitivitas sekunder yang ditetapkan sebelumnya, Jalur B memilih satu evaluasi clustering dari run "
          "A dan D di sel ablasi dengan aturan leksikografis beku yang hanya memakai metrik latih dan validasi (akurasi "
          "semu validasi tertinggi, lalu separasi minimum validasi, jarak rata-rata ke label benar, loss latih, dan indeks "
          "evaluasi). Dari checkpoint itu, label kuantum diekstrak ulang, fase supervised dilatih ulang dengan anggaran "
          "yang sama (940 evaluasi), dan model dievaluasi pada set uji yang sama; data uji tidak dipakai untuk pemilihan. "
          f"Seluruh {N['jalurB_n']} run selesai. Pemilihan checkpoint clustering menaikkan akurasi rata-rata A sebesar "
          f"{f(JB['A_change_mean'])} pp dan D sebesar {f(JB['D_change_mean'])} pp. Rata-rata selisih D − A pada sembilan "
          f"sel menjadi {s(JB['DA_mean_jalurB'])} pp, dibandingkan {s(JB['DA_mean_jalurA'])} pp pada titik akhir; "
          f"selisih positif pada {JB['DA_pos_jalurB']} sel (pada titik akhir {JB['DA_pos_jalurA']} sel), tanda selisih sama pada {JB['same_sign']} dari 9 sel, dan SK berada di atas nol pada {JB['ci_pos_jalurB']} sel "
          f"(Tabel {{tab:jalurb}}). Arah efek primer bertahan secara rata-rata tetapi kurang konsisten per sel, sehingga "
          "besar efek per sel bergantung pada titik henti optimizer."),
    ("tab", "jalurb", "Jalur B: akurasi uji (%) A dan D dengan checkpoint clustering terpilih dari validasi, serta selisih D − A "
     "(pp) pada titik akhir (Jalur A) dan pada checkpoint terpilih (Jalur B) dengan SK 95%. Tanda bintang menandai SK "
     "yang tidak memuat nol.",
     [["Fitur", "*K*", "A (B)", "D (B)", "D − A (A)", "D − A (B)"]],
     jalurb_rows(),
     [900, 400, 1000, 1000, 1300, 2400]),

    ("h1n", "Daftar Pustaka"),
    ("refs",),
]


# ------------------------------------------------------------------ bidi: Arabic-Indic digits
# Digit Arab-Indik berkelas bidi AN; tanpa penanda arah, urutan seperti "٧↔٨" atau "٠–٩" ditampilkan terbalik di
# Word/LibreOffice. Sisipkan LEFT-TO-RIGHT MARK (U+200E) setelah setiap digit Arab-Indik.
_AI = set("٠١٢٣٤٥٦٧٨٩")


def _lrm(text):
    return "".join(ch + "\u200e" if ch in _AI else ch for ch in text) if isinstance(text, str) else text


def _fix(block):
    kind = block[0]
    if kind in ("p", "pn", "h1", "h2", "h3"):
        return (kind, _lrm(block[1]))
    if kind == "bul":
        return (kind, [_lrm(x) for x in block[1]])
    if kind == "fig":
        return block[:4] + (_lrm(block[4]),)
    if kind == "tab":
        return (kind, block[1], _lrm(block[2]), [[_lrm(c) for c in r] for r in block[3]],
                [[_lrm(c) for c in r] for r in block[4]], block[5])
    return block


BLOCKS = [_fix(b) for b in BLOCKS]
ABSTRACT = _lrm(ABSTRACT)
