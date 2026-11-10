## INFORMASI PROYEK

**Judul Proyek:**  
"Prediksi Risiko Alih Fungsi Lahan Pasca-Karhutla Menjadi Perkebunan Kelapa Sawit"
---
**Nama Mahasiswa:** Fathur Roshin Haddinanta & Prima Afda Mukhlisan  
**NIM:** 244311014 & 244311024  
**Program Studi:** Teknologi Rekayasa Perangkat Lunak  
**Mata Kuliah:** Data Science  
**Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.  
**Tahun Akademik:** 2026, Semester 5  
**Link GitHub Repository:** (https://github.com/fathursn21/UAS-Data-Science)  
**Link Video Pembahasan:** [URL Video (YouTube/Drive) – Diisi pada Minggu 6]  
**SDG yang Dipilih:** SDG 15 – Ekosistem Daratan (*Life on Land*)  
**Target SDG yang Relevan:** Target 15.1 (Pelestarian dan restorasi ekosistem daratan, hutan, dan lahan gambut) & Target 15.2 (Pengelolaan hutan berkelanjutan, penghentian deforestasi, dan restorasi hutan terdegradasi)

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti **ML Lifecycle** yang terdiri dari empat tahap berulang (iteratif):

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|-------|-----------|-----------------|----------------|
| **Learning Problem** | **L** | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization** | **O** | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P** | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```text
   L (Learning Problem) --> M (Model Design) --> O (Optimization) --> P (Predict & Evaluate)
          ^                                                                   |
          +--------------- iterasi / perbaikan berdasarkan hasil evaluasi ----+
```

**Catatan untuk mahasiswa:** Setiap bagian diberi label **[L]**, **[M]**, **[O]**, atau **[P]**. Pastikan Anda dapat menjelaskan keterkaitan antar tahap, misalnya bagaimana hasil **P** mendorong perbaikan pada **L/M/O** (Bab 7.5).

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. **[L]** Mengidentifikasi masalah nyata dari **ide sendiri** yang selaras dengan salah satu tujuan **SDGs** ([Agenda 2030 – sdgs.un.org/2030agenda](https://sdgs.un.org/2030agenda)), lalu merumuskan problem statement (target, objective, data) secara jelas
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. **[L]** Melakukan data preparation yang sesuai dengan karakteristik dataset
4. **[M]** Merancang tiga model machine learning dengan justifikasi hypothesis space dan inductive bias (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. **[O]** Mengoptimasi model secara sistematis (loss function, optimizer, hyperparameter tuning, regularisasi)
6. **[P]** Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML dan menganalisis hasilnya
7. Melaporkan hasil eksperimen secara ilmiah dan sistematis
8. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
9. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW  **[L]**

### 2.1 Latar Belakang

Kebakaran hutan dan lahan (karhutla) merupakan salah satu krisis ekologis tahunan yang paling krusial di Indonesia, terutama di provinsi-provinsi dengan luasan hutan tropis dan lahan gambut yang besar seperti Riau, Sumatera Selatan, Kalimantan Tengah, dan Kalimantan Barat. Selain melepaskan emisi gas rumah kaca dalam jumlah masif dan memicu bencana kabut asap lintas batas, karhutla sering kali menjadi titik awal dari perubahan penggunaan lahan (*land-use change*) yang permanen. Dalam banyak kasus, area bekas terbakar (*burn scars*) tidak dibiarkan mengalami suksesi alami atau restorasi ekosistem, melainkan beralih fungsi menjadi perkebunan monokultur, khususnya kelapa sawit (*Elaeis guineensis*), dalam kurun waktu 1 hingga 3 tahun pasca-kebakaran (Descals et al., 2024; Gaveau et al., 2022).

Dinamika konversi lahan pasca-karhutla dipengaruhi oleh interaksi kompleks antara faktor biofisik, iklim, dan aksesibilitas geografis. Studi menunjukkan bahwa kedekatan suatu lahan terbakar dengan perkebunan kelapa sawit eksisting, jaringan jalan, akses sungai, serta anomali cuaca kering ekstrem (seperti fenomena El Niño) berperan besar dalam menentukan apakah suatu area bekas kebakaran akan ditanami kelapa sawit atau tetap menjadi semak belukar dan hutan sekunder (Ramdani et al., 2026). Sayangnya, pemantauan konversi lahan saat ini umumnya bersifat reaktif, yaitu baru terdeteksi setelah kanopi kelapa sawit tumbuh dan tertangkap oleh citra satelit beberapa tahun kemudian.

Proyek ini menawarkan pendekatan proaktif berbasis *Machine Learning* dan *Deep Learning* untuk memprediksi probabilitas risiko alih fungsi lahan pasca-karhutla menjadi perkebunan kelapa sawit dalam jendela waktu 1–3 tahun setelah kebakaran terjadi. Dengan mengintegrasikan data historis titik panas (*hotspot*) dari NASA FIRMS, peta tahun tanam kelapa sawit resolusi tinggi (Descals et al., 2024), faktor jarak geografis dari OpenStreetMap/Global Forest Watch, serta data iklim dari BMKG/CHIRPS, model prediktif yang dibangun dapat membantu instansi kehutanan (seperti KLHK dan BRGM) serta lembaga konservasi untuk mengidentifikasi area bekas terbakar yang berisiko tinggi mengalami konversi ilegal, sehingga intervensi restorasi lahan dan patroli pengawasan dapat diprioritaskan secara efektif.

**Referensi Ilmiah:**
> 1. Descals, A., Gaveau, D. L. A., Wich, S., Szantoi, Z., & Meijaard, E. (2024). Global mapping of oil palm planting year from 1990 to 2021. *Earth System Science Data*, 16(11), 5111–5129. https://doi.org/10.5194/essd-16-5111-2024
> 2. Gaveau, D. L. A., Locatelli, B., Salim, M. A., Husnayaen, Manurung, T., Descals, A., Angelsen, A., Meijaard, E., & Sheil, D. (2022). Slowing deforestation in Indonesia follows declining oil palm expansion and lower oil prices. *PLOS ONE*, 17(3), e0266178. https://doi.org/10.1371/journal.pone.0266178
> 3. Ramdani, R., Fatchurrochman, A., Purnomo, E. P., Amin, A. M., Al Hifdhi, H., Ariesca, R., & Kuncoro, O. B. (2026). Beyond the slash-and-burn narrative: evidence on forest and land fire drivers from oil palm enclaves in West Kalimantan, Indonesia. *Fire Ecology*, 22(1), 168. https://doi.org/10.1186/s42408-026-00600-6

### 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? | **SDG 15:** Ekosistem Daratan (*Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss*). |
| **Target SDG** | Target (misal 3.4, 13.1) mana yang paling relevan? | **Target 15.1** (Pelestarian, restorasi, dan pemanfaatan berkelanjutan ekosistem daratan, khususnya hutan dan lahan gambut) dan **Target 15.2** (Menghentikan deforestasi dan merestorasi hutan yang terdegradasi). |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? | Konversi lahan bekas karhutla menjadi perkebunan kelapa sawit dalam 1–3 tahun pasca-kebakaran di Riau, Sumatera Selatan, Kalimantan Tengah, dan Kalimantan Barat. Pihak yang terdampak meliputi masyarakat lokal sekitar hutan (dampak kabut asap, banjir, hilangnya mata pencaharian ekosistem), keanekaragaman hayati endemik, serta lembaga pemerintah (KLHK/BRGM) yang kesulitan memprioritaskan ribuan hektar lahan terbakar untuk direstorasi. |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan atau proses yang dibantu? | Hubungan antara intensitas kebakaran, jarak ke kebun sawit eksisting, aksesibilitas jalan/sungai, dan anomali iklim bersifat non-linear dan berdimensi tinggi. ML/DL mampu mempelajari pola spasio-temporal historis tersebut untuk menghasilkan **skor probabilitas risiko konversi lahan** segera setelah kebakaran padam, sehingga membantu pengambil kebijakan menentukan lokasi prioritas restorasi dan pengawasan lapangan. |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi pada target SDG (langsung/tidak langsung)? | Kontribusi **tidak langsung** terhadap Indikator SDG 15.1.1 (*Forest area as a proportion of total land area*) dan 15.2.1 (*Progress towards sustainable forest management*) dengan menyediakan sistem pendukung keputusan (*Decision Support System*) berbasis *early-warning* untuk mencegah hilangnya tutupan hutan permanen pasca-kebakaran. |
| **Pertimbangan etika** | Risiko bias, privasi, atau dampak salah prediksi? | **1. Bias Sensor:** Satelit optik dapat mengalami *under-detection* pada area berawan tebal atau kebakaran gambut bawah permukaan.<br>**2. Dampak Salah Prediksi (*False Positive*):** Prediksi model adalah estimasi risiko ekologis, bukan bukti pidana pembakaran disengaja oleh petani kecil (*smallholders*) maupun korporasi. Output model harus digunakan untuk alokasi program restorasi dan pendampingan, bukan kriminalisasi tanpa verifikasi lapangan.<br>**3. Privasi:** Data bersumber dari citra satelit publik (10m–500m) dan stasiun cuaca tanpa memuat data pribadi individu. |

**Sumber ide:** Ide ini berangkat dari pengamatan terhadap isu nasional bencana kebakaran hutan dan lahan (karhutla) yang berulang di Sumatera dan Kalimantan, serta diskursus publik dan temuan ilmiah mengenai fenomena beralihnya lahan bekas terbakar menjadi perkebunan kelapa sawit beberapa tahun setelah api padam.

### 2.3 Ringkasan Alur Lifecycle Proyek

Proyek ini dijalankan secara iteratif mengikuti kerangka **L-M-O-P**. Pada tahap **Learning Problem (L)**, masalah alih fungsi lahan pasca-karhutla dirumuskan sebagai tugas klasifikasi biner pada data tabular spasio-temporal dengan menggabungkan data titik panas NASA FIRMS, data tahun tanam sawit (Zenodo/GFW), jarak infrastruktur (OSM), dan data cuaca (BMKG/CHIRPS) pada empat provinsi prioritas. Selanjutnya, pada tahap **Model Design (M)**, dirancang tiga arsitektur model dengan tingkat kompleksitas berbeda: *Logistic Regression* sebagai *baseline* linear, *XGBoost Classifier* sebagai model *ensemble tree-based*, dan *Multilayer Perceptron* (MLP) sebagai model *Deep Learning*. Pada tahap **Optimization (O)**, dilakukan pelatihan dengan fungsi *Weighted Binary Cross-Entropy Loss*, *hyperparameter tuning* pada validasi silang, serta penerapan regularisasi (*Dropout*, *Batch Normalization*, dan *Early Stopping*) untuk mencegah *overfitting*. Akhirnya, pada tahap **Predict & Evaluate (P)**, performa ketiga model diuji pada *test set* menggunakan metrik PR-AUC, ROC-AUC, dan F1-Score; temuan dari analisis kesalahan (*error analysis*), seperti rendahnya *recall* pada kelas positif akibat ketimpangan kelas (*class imbalance*), digunakan sebagai umpan balik iteratif untuk memperbaiki teknik penyeimbangan data pada tahap **L** dan penyesuaian bobot *loss* pada tahap **O**.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements

1. **Identifikasi Risiko Pasca-Karhutla:** Bagaimana memprediksi apakah suatu titik atau area bekas kebakaran hutan dan lahan (*burn scar*) di provinsi prioritas (Riau, Sumatera Selatan, Kalimantan Tengah, Kalimantan Barat) akan beralih fungsi menjadi perkebunan kelapa sawit dalam rentang waktu 1–3 tahun pasca-kebakaran?
2. **Interaksi Fitur Spasio-Temporal dan Iklim:** Bagaimana pengaruh faktor pendorong ekspansi geografis ( khususnya jarak dari titik terbakar ke perkebunan sawit eksisting, jalan, dan sungai) yang dikombinasikan dengan karakteristik intensitas kebakaran (*Fire Radiative Power*) serta anomali iklim terhadap probabilitas konversi lahan?
3. **Penanganan Ketimpangan Kelas (*Imbalanced Data*):** Mengingat tidak semua lahan bekas terbakar berubah menjadi perkebunan sawit (kelas positif merupakan kelas minoritas), bagaimana merancang *pipeline* pemrosesan data dan fungsi objektif yang mampu meminimalkan *False Negative* tanpa mengorbankan presisi secara berlebihan?
4. **Komparasi Kapasitas Representasi Model:** Sejauh mana peningkatan performa yang dihasilkan oleh model *ensemble non-linear* (XGBoost) dan model *Deep Learning* (MLP) dibandingkan dengan model linear sederhana (*Logistic Regression*) dalam menangkap pola konversi lahan pasca-karhutla?

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **Target** | Apa yang diprediksi? Jenis tugas (klasifikasi biner/multikelas, regresi, forecasting, clustering, dll.)? | **Jenis Tugas:** Klasifikasi Biner (*Binary Classification*).<br><br>**Target (Y):** Variabel biner `is_converted_to_palm` ∈ {0, 1}, di mana Y = 1 jika titik/grid lahan yang terbakar pada tahun ke-t terdeteksi berubah menjadi perkebunan kelapa sawit pada tahun t+1, t+2, atau t+3; dan Y = 0 jika lahan tersebut tidak berubah menjadi perkebunan kelapa sawit dalam kurun waktu 1–3 tahun pasca-karhutla. |
| **Objective** | Apa yang dioptimalkan/diminimalkan secara matematis (misal: cross-entropy, MSE)? Metrik keberhasilan utama? | **Objective Matematis:** Meminimalkan *Weighted Binary Cross-Entropy Loss*:<br>ℒ(θ) = - (1/N) ∑ [ w₁ yᵢ log(ŷᵢ) + w₀ (1 - yᵢ) log(1 - ŷᵢ) ]<br><br>**Metrik Keberhasilan Utama:** **PR-AUC (*Precision-Recall AUC*)**, **ROC-AUC**, **F1-Score**, dan **Recall** pada kelas positif (Y=1). |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data (i.i.d.? distribusi train vs test?) | **Sumber & Jenis:** Data tabular spasio-temporal berukuran ~50.000–150.000 baris dan 18 fitur (dari NASA FIRMS, GFW, BMKG, dll.).<br><br>**Asumsi Data:** Observasi diasumsikan independen secara kondisional terhadap fitur spasial yang diekstraksi, namun dijaga dengan *stratified split* dan pemeriksaan kebocoran spasial (*spatial leakage*). |

### 3.3 Goals

1. **Membangun Pipeline Dataset Spasio-Temporal:** Mengintegrasikan data historis titik panas karhutla, data tahun tanam kelapa sawit (rentang 1–3 tahun pasca-kebakaran), jarak infrastruktur geografis, dan parameter cuaca di empat provinsi prioritas menjadi dataset tabular yang bersih dan bebas dari *data leakage*.
2. **Mencapai Performa Prediksi yang Andal pada Data Timpang:** Menghasilkan model prediktif yang mampu mencapai nilai **ROC-AUC ≥ 0.80**, **PR-AUC ≥ 0.65**, dan **F1-Score ≥ 0.70** pada data uji (*test set*).
3. **Mengevaluasi Tiga Pendekatan Model (Baseline, Advanced, Deep Learning):** Membandingkan kinerja, waktu komputasi, serta kesesuaian *inductive bias* antara *Logistic Regression*, *XGBoost Classifier*, dan *Multilayer Perceptron* (MLP) yang dilatih minimal 20 *epochs*.
4. **Mendukung Pencapaian SDG 15 (Target 15.1 & 15.2):** Mengidentifikasi fitur-fitur pendorong utama konversi lahan pasca-karhutla (melalui analisis *Feature Importance*) dan menghasilkan peta/peringkat probabilitas risiko untuk membantu pemangku kepentingan memprioritaskan lokasi restorasi ekosistem daratan.

### 3.4 Solution Approach  **[M]**

Proyek ini mengimplementasikan dan membandingkan **tiga model** dengan karakteristik sebagai berikut:

#### **Model 1 – Baseline Model: Logistic Regression**
- **Pilihan Model:** *Logistic Regression* dengan regularisasi L2 (*Ridge*) dan penyesuaian bobot kelas (`class_weight='balanced'`).
- **Alasan Pemilihan:** *Logistic Regression* merupakan tolok ukur (*baseline*) standar untuk klasifikasi biner tabular. Model ini memiliki kompleksitas rendah, waktu pelatihan sangat cepat, menghasilkan output berupa probabilitas yang terkalibrasi melalui fungsi sigmoid, serta memungkinkan kita melihat arah pengaruh linear setiap fitur (misalnya apakah semakin dekat jarak ke kebun sawit eksisting akan meningkatkan log-odds konversi lahan secara monotonik).

#### **Model 2 – Advanced / ML Model: XGBoost Classifier**
- **Pilihan Model:** *Extreme Gradient Boosting* (`XGBClassifier`).
- **Alasan Pemilihan:** Data geospasial dan iklim memiliki interaksi non-linear dan ambang batas (*threshold*) yang kuat. misalnya, risiko alih fungsi lahan mungkin melonjak drastis hanya jika jarak ke kebun sawit eksisting kurang dari 2 km **dan** terjadi pada musim kemarau ekstrem dengan intensitas kebakaran berulang. Algoritma *gradient boosted decision trees* seperti XGBoost sangat unggul pada data tabular heterogen, tahan terhadap multikolinearitas, menyediakan fitur `scale_pos_weight` untuk menangani ketimpangan kelas, dan menghasilkan *feature importance* yang informatif.

#### **Model 3 – Deep Learning Model (WAJIB): Multilayer Perceptron (MLP / Deep Tabular Network)**
- **Pilihan Implementasi:** **A. Tabular Data – Multilayer Perceptron (MLP)** dengan arsitektur *Deep Neural Network* yang terdiri dari 3 *hidden layers* (`128 -> 64 -> 32` *units*), dilengkapi *Batch Normalization*, fungsi aktivasi *ReLU* (*Rectified Linear Unit*), dan regularisasi *Dropout* (`0.3`).
- **Alasan Pemilihan:** Berbeda dengan model pohon keputusan yang mempartisi ruang fitur secara ortogonal (*axis-aligned splits*), MLP mampu mempelajari kombinasi representasi fitur laten yang mulus (*smooth non-linear decision boundaries*) di ruang berdimensi tinggi. Model ini dilatih selama minimal 30 *epochs* dengan *optimizer* Adam, *learning rate scheduler*, serta pencatatan kurva *loss* dan *metrics* pada setiap *epoch*.
---

## 4. DATA UNDERSTANDING  **[L]**
### 4.1 Informasi Dataset
**Sumber Dataset:**  
[Sebutkan sumber: Kaggle, UCI ML Repository, atau sumber lain dengan URL]

**Deskripsi Dataset:**
- Jumlah baris (rows): [angka]
- Jumlah kolom (columns/features): [angka]
- Tipe data: [Tabular / Image / Text / Time Series / Audio / Video]
- Ukuran dataset: [MB/GB]
- Format file: [CSV / JSON / Images / TXT / etc.]

### 4.2 Deskripsi Fitur
Jelaskan setiap fitur/kolom yang ada dalam dataset.
**Contoh tabel:**
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| id | Integer | ID unik data | 1, 2, 3 |
| age | Integer | Usia (tahun) | 25, 30, 45 |
| income | Float | Pendapatan (juta) | 5.5, 10.2 |
| category | Categorical | Kategori produk | A, B, C |
| text | String | Teks ulasan | "Produk bagus..." |
| image | Image | Citra 224x224 RGB | Array 224x224x3 |
| label | Categorical | Label target | 0, 1 atau "positif", "negatif" |

**[Buat tabel deskripsi fitur Anda di sini]**

### 4.3 Kondisi Data

Jelaskan kondisi dan permasalahan data:

- **Missing Values:** [Ada/Tidak, berapa persen?]
- **Duplicate Data:** [Ada/Tidak, berapa banyak?]
- **Outliers:** [Ada/Tidak, pada fitur apa?]
- **Imbalanced Data:** [Ada/Tidak, rasio kelas?]
- **Noise:** [Jelaskan jika ada]
- **Data Quality Issues:** [Jelaskan jika ada masalah lain]
- **Asumsi Data (i.i.d., distribusi train-test, potensi data leakage):** [Jelaskan]

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

**Requirement:** Minimal 3 visualisasi yang bermakna dan insight-nya.
**Contoh jenis visualisasi yang dapat digunakan:**
- Histogram (distribusi data)
- Boxplot (deteksi outliers)
- Heatmap korelasi (hubungan antar fitur)
- Bar plot (distribusi kategori)
- Scatter plot (hubungan 2 variabel)
- Wordcloud (untuk text data)
- Sample images (untuk image data)
- Time series plot (untuk temporal data)
- Confusion matrix heatmap
- Class distribution plot


#### Visualisasi 1: [Judul Visualisasi]
[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 2: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 3: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

---

## 5. DATA PREPARATION  **[L]**

Bagian ini menjelaskan **semua** proses transformasi dan preprocessing data yang dilakukan.
### 5.1 Data Cleaning
**Aktivitas:**
- Handling missing values
- Removing duplicates
- Handling outliers
- Data type conversion
**Contoh:**
```
Missing Values:
- Fitur 'age' memiliki 50 missing values (5% dari data)
- Strategi: Imputasi dengan median karena distribusi skewed
- Alasan: Median lebih robust terhadap outliers dibanding mean
```

**[Jelaskan langkah-langkah data cleaning yang Anda lakukan]**

### 5.2 Feature Engineering
**Aktivitas:**
- Creating new features
- Feature extraction
- Feature selection
- Dimensionality reduction

**[Jelaskan feature engineering yang Anda lakukan]**

### 5.3 Data Transformation

**Untuk Data Tabular:**
- Encoding (Label Encoding, One-Hot Encoding, Ordinal Encoding)
- Scaling (Standardization, Normalization, MinMaxScaler)

**Untuk Data Text:**
- Tokenization
- Lowercasing
- Removing punctuation/stopwords
- Stemming/Lemmatization
- Padding sequences
- Word embedding (Word2Vec, GloVe, fastText)

**Untuk Data Image:**
- Resizing
- Normalization (pixel values 0-1 atau -1 to 1)
- Data augmentation (rotation, flip, zoom, brightness, etc.)
- Color space conversion

**Untuk Time Series:**
- Creating time windows
- Lag features
- Rolling statistics
- Differencing

**[Jelaskan transformasi yang Anda lakukan]**

### 5.4 Data Splitting

**Strategi pembagian data:**
```
- Training set: [X]% ([jumlah] samples)
- Validation set: [X]% ([jumlah] samples) - jika ada
- Test set: [X]% ([jumlah] samples)
```
**Contoh:**
```
Menggunakan stratified split untuk mempertahankan distribusi kelas:
- Training: 80% (8000 samples)
- Test: 20% (2000 samples)
- Random state: 42 untuk reproducibility
```

> **Penting:** Test set hanya digunakan **sekali** pada tahap P (evaluasi akhir). Tuning pada tahap O hanya boleh memakai validation set / cross-validation. Fit scaler/encoder hanya pada data training untuk menghindari data leakage.

**[Jelaskan strategi splitting Anda dan alasannya]**

### 5.5 Data Balancing (jika diperlukan)
**Teknik yang digunakan:**
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Undersampling
- Class weights
- Ensemble sampling

**[Jelaskan jika Anda melakukan data balancing]**

### 5.6 Ringkasan Data Preparation

**Per langkah, jelaskan:**
1. **Apa** yang dilakukan
**[Jelaskan ]**
2. **Mengapa** penting
**[Jelaskan Mengapa ?]**
3. **Bagaimana** implementasinya
**[Jelaskan Bagaimana]**

---

## 6. MODELING  **[M + O]**

> Setiap model (6.1, 6.2, 6.3) dijelaskan dalam dua fase: **Model Design [M]** (sub-bagian .1–.3) dan **Optimization [O]** (sub-bagian .4–.6).

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama model, misal: Logistic Regression]  
**Teori Singkat:**  
[Jelaskan secara singkat bagaimana model ini bekerja]  
**Alasan Pemilihan:**  
[Mengapa memilih model ini sebagai baseline?]

#### 6.1.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [misal: model linear]
- **Hypothesis space:** [himpunan fungsi yang dapat direpresentasikan, misal: batas keputusan linear]
- **Inductive bias / asumsi:** [misal: hubungan fitur-target linear, fitur independen]
- **Implikasi terhadap dataset Anda:** [apakah asumsi cocok dengan data?]

#### 6.1.3 Implementasi (Ringkas)  **[M]**
```python
# Contoh kode (opsional, bisa dipindah ke GitHub)
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi  **[O]**

**Objective/loss yang dioptimalkan:** [misal: log-loss + L2]  
**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- C (regularization): 1.0
- solver: 'lbfgs'
- max_iter: 100
```

#### 6.1.5 Hasil Awal  **[O → P]**

**[Tuliskan hasil awal pada validation set; evaluasi akhir pada test set dijelaskan di Bab 7]**

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama model, misal: Random Forest / XGBoost]  
**Teori Singkat:**  
[Jelaskan bagaimana algoritma ini bekerja]

**Alasan Pemilihan:**  
[Mengapa memilih model ini?]

**Keunggulan:**
- [Sebutkan keunggulan]

**Kelemahan:**
- [Sebutkan kelemahan]

#### 6.2.2 Hypothesis Space & Inductive Bias  **[M]**

- **Model family / arsitektur:** [misal: ensemble pohon keputusan]
- **Hypothesis space:** [fungsi non-linear piecewise, interaksi fitur otomatis, dll.]
- **Inductive bias / asumsi:** [misal: data tabular dengan struktur ambang/threshold]
- **Perbandingan dengan baseline:** [apa yang bisa dipelajari model ini yang tidak bisa baseline?]

#### 6.2.3 Implementasi (Ringkas)  **[M]**
```python
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Optimization: Hyperparameter Tuning  **[O]**

**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (WAJIB minimal 1 metode sederhana):**
- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Search space: [rentang parameter yang dicoba]
- Skema validasi: [k-fold CV / validation set]
- Best parameters: [...]
- Skor validasi sebelum vs sesudah tuning: [...]

#### 6.2.5 Kontrol Overfitting  **[O]**
[Jelaskan regularisasi/pembatasan kompleksitas yang dipakai, misal: max_depth, min_samples_leaf, early stopping pada boosting]

#### 6.2.6 Hasil Model  **[O → P]**

**[Tuliskan hasil pada validation set; evaluasi akhir dijelaskan di Bab 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model  **[M]**

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Mengapa arsitektur ini cocok untuk dataset Anda?]

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias  **[M]**

**Inductive bias arsitektur:** [misal: CNN → lokalitas & translation invariance; LSTM → ketergantungan sekuensial; MLP → tanpa asumsi struktur]  
**Hypothesis space:** [kapasitas model: jumlah layer, unit, parameter]

**Deskripsi Layer:**

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus  **[M]**

**Input shape:** [Sebutkan dimensi input]  
**Preprocessing khusus untuk DL:**
- [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]

#### 6.3.4 Optimization: Konfigurasi Training  **[O]**

**Training Configuration:**
```
- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
- Regularisasi: [Dropout / L2 / BatchNorm / augmentasi]
```

**Eksperimen optimasi (minimal 2 konfigurasi dibandingkan):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Metric |
|------------|---------------|------------|--------------|----------|------------|
| Exp-1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| Exp-2 | [ ] | [ ] | [ ] | [ ] | [ ] |

#### 6.3.5 Implementasi (Ringkas)  **[M + O]**

**Framework:** TensorFlow/Keras / PyTorch
```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process & Diagnosis Optimasi  **[O]**

**Training Time:**  
[Sebutkan waktu training total, misal: 15 menit]

**Computational Resource:**  
[CPU / GPU, platform: Local / Google Colab / Kaggle]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]
- Apakah learning rate sesuai (loss stabil/osilasi/lambat)? [Jelaskan]

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION  **[P]**

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**Justifikasi pemilihan metrik:** [Mengapa metrik ini selaras dengan Objective di Bab 3.2? Mengapa accuracy saja cukup/tidak cukup?]

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Semua prediksi dievaluasi pada **test set** yang tidak pernah dipakai saat training maupun tuning.

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Tampilkan minimal 5 contoh prediksi benar dan salah]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting & Generalization Gap:**  
   [Bandingkan performa train vs validation vs test; analisis apakah model overfitting atau underfitting]

6. **Kesesuaian Inductive Bias:**  
   [Apakah asumsi tiap model (Bab 6.x.2) terbukti sesuai dengan data? Jelaskan dengan bukti hasil]

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

Jelaskan minimal **satu** iterasi perbaikan berdasarkan hasil evaluasi:

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
|--------------------------|-----------------------|----------|--------------------|
| [misal: recall kelas minoritas rendah] | [L] data | [misal: class weight / SMOTE] | [ ... ] |
| [misal: gap train-val besar] | [O] optimasi | [misal: tambah dropout, early stopping] | [ ... ] |
| [misal: model terlalu sederhana] | [M] desain | [misal: tambah layer / ganti arsitektur] | [ ... ] |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.3 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
|-------|-----------------|-----------|
| **L** – Learning Problem | [ ... ] | [ ... ] |
| **M** – Model Design | [ ... ] | [ ... ] |
| **O** – Optimization | [ ... ] | [ ... ] |
| **P** – Predict & Evaluate | [ ... ] | [ ... ] |

### 8.4 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Kontribusi terhadap SDGs:**  
[Jelaskan secara jujur dan terukur bagaimana hasil proyek mendukung target SDG yang dipilih (Bab 2.2), serta keterbatasan kontribusinya]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

**Saran struktur folder (mengikuti lifecycle):**
```
project/
├── data/                 # [L] dataset (jangan upload jika besar)
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb      # [L]
│   ├── 02_data_preparation.ipynb          # [L]
│   ├── 03_model_design_baseline_ml.ipynb  # [M]
│   ├── 04_deep_learning_optimization.ipynb# [M+O]
│   └── 05_evaluation_comparison.ipynb     # [P]
├── src/                  # script modular
├── models/               # model tersimpan
├── reports/figures/
├── requirements.txt
└── README.md
```

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll
```

### 10.3 Reproducibility Checklist
- [ ] Random seed ditetapkan (misal 42) di semua library
- [ ] Notebook dapat dijalankan ulang dari awal sampai akhir
- [ ] Test set tidak dipakai saat training/tuning
- [ ] Versi library tercatat

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

Pengerjaan mengikuti urutan lifecycle L → M → O → P dengan satu iterasi perbaikan. Rincian tugas per minggu ada di file panduan terpisah: **Panduan_Tahapan_6_Minggu.md**.

| Minggu | Fokus Lifecycle | Aktivitas | Deliverable / Checkpoint |
|--------|-----------------|-----------|--------------------------|
| **1** | **L** – Learning Problem | Mahasiswa mengajukan ide sendiri selaras dengan satu SDG, pilih dataset, studi literatur, rumuskan problem statement, target, objective, data (Bab 1–3) | Formulir ide + proposal singkat 1–2 halaman; ide & dataset disetujui dosen. **Checkpoint 1** |
| **2** | **L** – Data | Data understanding, EDA (opsional), cleaning, transformasi, splitting (Bab 4–5) | Notebook data preparation + repo GitHub dibuat |
| **3** | **M** – Model Design | Rancang & latih Model 1 (baseline) dan Model 2 (advanced); jelaskan hypothesis space & inductive bias (Bab 6.1–6.2) | Hasil baseline & advanced pada validation set. **Checkpoint 2** |
| **4** | **M → O** | Tuning Model 2; desain arsitektur Model 3 (deep learning) dan training pertama (Bab 6.2.4, 6.3.1–6.3.5) | Model DL berjalan end-to-end (training awal) |
| **5** | **O** – Optimization | Optimasi Model 3: eksperimen learning rate/batch size/regularisasi, plot loss/metric, diagnosis konvergensi (Bab 6.3.4–6.3.7) | Model DL terlatih (≥10 epoch) + tabel eksperimen. **Checkpoint 3** |
| **6** | **P** – Predict & Evaluate | Evaluasi test set, perbandingan 3 model, error analysis, iterasi perbaikan, kesimpulan, video, finalisasi repo (Bab 7–10) | Laporan final + GitHub + video + presentasi |

**Catatan untuk dosen:**
- Enam minggu memberi ruang tambahan untuk tahap Optimization, tempat mahasiswa biasanya paling banyak mengalami kendala (training deep learning, tuning, overfitting).
- Tiga checkpoint (akhir minggu 1, 3, dan 5) membantu mendeteksi keterlambatan lebih awal.

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|----------|-------|-------|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2) | L | 10% |
| Problem formulation (target, objective, data), latar belakang & referensi | L | 10% |
| Data understanding & data preparation (tanpa data leakage) | L | 10% |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias | M | 15% |
| Model deep learning: arsitektur, training ≥10 epoch, plot loss/metric | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P | 20% |
| Reproducibility: GitHub, README, requirements, struktur folder | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| **Total** | | **100%** |

> Bobot dapat disesuaikan dengan kebijakan mata kuliah.

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

Referensi resmi: [Transforming our world: the 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda)

**Aturan ide:**
- Ide **harus berasal dari mahasiswa**, bukan dari daftar contoh di bawah. Contoh hanya inspirasi.
- Pilih **satu** SDG utama dan sebutkan target (nomor target) yang relevan.
- Masalah harus nyata, spesifik, dan dapat dirumuskan sebagai tugas ML (klasifikasi, regresi, forecasting, dll.).
- Dataset harus tersedia (publik atau dikumpulkan sendiri secara etis) dan cukup untuk melatih tiga model, termasuk deep learning.
- Disarankan setiap mahasiswa/kelompok memiliki ide yang berbeda; dosen memeriksa di Checkpoint 1.

| SDG | Nama (Indonesia) | Contoh arah ide ML (inspirasi) |
|-----|------------------|--------------------------------|
| 1 | Tanpa Kemiskinan | Prediksi tingkat kemiskinan wilayah dari data sosial-ekonomi (tabular) |
| 2 | Tanpa Kelaparan | Klasifikasi penyakit daun tanaman (image); prediksi hasil panen (tabular) |
| 3 | Kehidupan Sehat dan Sejahtera | Deteksi penyakit dari data klinis atau citra medis; prediksi risiko diabetes |
| 4 | Pendidikan Berkualitas | Prediksi risiko putus studi/performa siswa; analisis ulasan pembelajaran (teks) |
| 5 | Kesetaraan Gender | Deteksi ujaran kebencian atau bias gender pada teks |
| 6 | Air Bersih dan Sanitasi Layak | Prediksi kelayakan/kualitas air minum (tabular) |
| 7 | Energi Bersih dan Terjangkau | Forecasting konsumsi atau produksi energi (time series) |
| 8 | Pekerjaan Layak dan Pertumbuhan Ekonomi | Klasifikasi lowongan kerja atau prediksi pengangguran |
| 9 | Industri, Inovasi, dan Infrastruktur | Deteksi kerusakan jalan (image); predictive maintenance |
| 10 | Berkurangnya Kesenjangan | Pengenalan bahasa isyarat (image); analisis akses layanan publik |
| 11 | Kota dan Permukiman yang Berkelanjutan | Prediksi kualitas udara kota; prediksi kepadatan lalu lintas |
| 12 | Konsumsi dan Produksi yang Bertanggung Jawab | Klasifikasi jenis sampah (image); prediksi limbah makanan |
| 13 | Penanganan Perubahan Iklim | Prediksi curah hujan/suhu; deteksi titik api atau kebakaran hutan |
| 14 | Ekosistem Lautan | Klasifikasi spesies ikan; deteksi sampah laut atau kesehatan terumbu karang |
| 15 | Ekosistem Daratan | Klasifikasi spesies flora/fauna; deteksi deforestasi dari citra |
| 16 | Perdamaian, Keadilan, dan Kelembagaan yang Tangguh | Deteksi hoaks/berita palsu (teks); deteksi fraud |
| 17 | Kemitraan untuk Mencapai Tujuan | Sistem rekomendasi kolaborasi/mitra; analisis data terbuka pemerintah |

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | [ ... ] |
| Judul ide | [ ... ] |
| SDG & target yang dipilih | [ ... ] |
| Masalah nyata yang diangkat | [ ... ] |
| Tugas ML (Target & jenis tugas) | [ ... ] |
| Objective & metrik utama | [ ... ] |
| Dataset (sumber, URL, ukuran, jenis data) | [ ... ] |
| Rencana tiga model (baseline / advanced / deep learning) | [ ... ] |
| Risiko & pertimbangan etika (bias, privasi) | [ ... ] |
| Referensi awal (minimal 2) | [ ... ] |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
