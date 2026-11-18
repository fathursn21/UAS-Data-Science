## 5. DATA PREPARATION  **[L]**

Bagian ini menjelaskan seluruh proses transformasi dan *preprocessing* data yang dijalankan melalui script `eda_dan_prep.py` terhadap `karhutla_palm_conversion_raw.parquet` hingga menghasilkan artefak siap latih `data/prepared_splits.joblib`.

### 5.1 Data Cleaning
1. **Filtering Noise Sensor Satelit (`confidence != 'low'`):**
   - Menghapus seluruh observasi titik panas dengan `confidence == 'low'` untuk menghindari *false alarm* deteksi termal, sehingga hanya menyisakan kategori `nominal` dan `high`.
2. **Pencegahan Data Leakage (`pre_existing_palm_flag == 0`):**
   - Mengeksklusi seluruh titik panas yang jatuh pada area yang **sudah menjadi perkebunan sawit sebelum atau pada saat tahun kebakaran** ($\text{palm\_planting\_year} \le \text{fire\_year}$), kemudian menghapus kolom audit `pre_existing_palm_flag`.
3. **Deduplikasi Spasio-Temporal Grid $500\text{ m} \times 500\text{ m}$:**
   - Membulatkan koordinat ke dalam grid `0.005°` ($\approx 500\text{ m}$), mengurutkan observasi berdasarkan `frp` dan `bright_ti4` tertinggi (`sort_values(ascending=False)`), lalu melakukan `drop_duplicates(subset=['grid_lat_500m', 'grid_lon_500m', 'fire_year'], keep='first')` agar setiap petak lahan yang terbakar pada tahun yang sama hanya diwakili oleh satu observasi puncak.
4. **Imputasi Missing Values Tanpa Leakage:**
   - Mengisi nilai `NaN` pada `temp_max_c`, `rh_avg_pct`, dan `rainfall_30d_mm` menggunakan **median per provinsi** (`X_train.groupby('province').median()`) yang dihitung **hanya dari `X_train`** setelah *data splitting*, lalu diterapkan ke `X_train`, `X_val`, dan `X_test`.
5. **Penanganan Outliers (Winsorizing P99 & Log-Transform):**
   - Memangkas (*clipping*) nilai ekstrem pada fitur yang *right-skewed* (`frp`, `dist_to_existing_palm_km`, `dist_to_road_km`, dan `dist_to_river_km`) menggunakan batas persentil ke-99 yang dihitung dari `X_train`, kemudian menerapkan transformasi $\log(1 + x)$ (`np.log1p`).

### 5.2 Feature Engineering
Empat kelompok fitur baru direkayasa untuk meningkatkan daya prediksi model:
1. **Ekstraksi Jarak Euclidean Geospasial (`cKDTree`):**
   - Menghitung jarak terdekat (dalam km) dari setiap titik panas ke titik kebun sawit eksisting (`dist_to_existing_palm_km`), jaringan jalan (`dist_to_road_km`), dan sungai (`dist_to_river_km`) dari file `sumatra-261116-free.gpkg` serta `kalimantan-261116-free.gpkg` menggunakan `scipy.spatial.cKDTree(workers=-1)`.
2. **Rasio Intensitas Termal (`fire_intensity_ratio`):**
   - Membuat fitur rasio $\frac{\text{bright\_ti4}}{\text{bright\_ti5}}$ untuk mengukur kontras suhu titik api terhadap latar belakang.
3. **Indikator Musim Kemarau Ekstrem (`is_extreme_dry`):**
   - Membuat fitur biner bernilai `1` apabila akumulasi curah hujan 30 hari sebelum kebakaran $\text{rainfall\_30d\_mm} < 50.0\text{ mm}$, dan `0` jika sebaliknya.
4. **Transformasi Siklikal Bulan Kebakaran (`fire_month_sin`, `fire_month_cos`):**
   - Mengubah `fire_month` ($1\text{--}12$) menjadi pasangan fitur trigonometri $\sin\left(\frac{2\pi \cdot \text{month}}{12}\right)$ dan $\cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$, lalu menghapus kolom `fire_month` mentah dan `hotspot_id`.

### 5.3 Data Transformation
Proses transformasi fitur dibungkus secara modular menggunakan `scikit-learn ColumnTransformer` yang di-*fit* **hanya pada `X_train`**:
- **Standardisasi Fitur Kontinu (`StandardScaler`):** Diterapkan pada 16 fitur numerik kontinu (`latitude`, `longitude`, `fire_year`, `bright_ti4`, `bright_ti5`, `frp`, `fire_freq_3yr`, `dist_to_existing_palm_km`, `dist_to_road_km`, `dist_to_river_km`, `rainfall_30d_mm`, `temp_max_c`, `rh_avg_pct`, `fire_intensity_ratio`, `fire_month_sin`, `fire_month_cos`) agar memiliki $\mu = 0$ dan $\sigma = 1$.
- **One-Hot Encoding (`OneHotEncoder`):** Diterapkan pada kolom kategorikal `province` (menghasilkan 4 kolom biner provinsi).
- **Ordinal Encoding (`OrdinalEncoder`):** Diterapkan pada kolom `confidence` dengan urutan tingkat kepercayaan `['nominal', 'high']` $\rightarrow \{0, 1\}$.
- **Passthrough Fitur Biner:** Kolom `in_palm_concession` dan `is_extreme_dry` dilewatkan tanpa perubahan skala.
- **Dimensi Matriks Fitur Akhir:** Menghasilkan matriks input berukuran **`23` kolom fitur** (`X_train_scaled`, `X_val_scaled`, `X_test_scaled`).

### 5.4 Data Splitting

Pembagian data dilakukan **sebelum** proses imputasi median, *winsorizing*, dan *scaling* menggunakan skema **Stratified Train-Validation-Test Split ($70\% : 15\% : 15\%$)** untuk menjaga proporsi kelas target `is_converted_to_palm` tetap identik di ketiga subset:

```text
Menggunakan Stratified Split (random_state = 42):
- Training set (X_train)   : 70% dari data bersih -> untuk melatih model & fitting preprocessors
- Validation set (X_val)   : 15% dari data bersih -> untuk hyperparameter tuning & early stopping (Tahap O)
- Test set (X_test)        : 15% dari data bersih -> diisolasi penuh, hanya digunakan sekali pada evaluasi akhir (Tahap P)
```

### 5.5 Data Balancing (jika diperlukan)
Untuk mengatasi ketimpangan kelas (`is_converted_to_palm` memiliki rasio ~`1 : 4`), diterapkan dua strategi penyeimbangan yang hanya dikenakan pada data training:
1. **Cost-Sensitive Learning (Pendekatan Utama):**
   - Menghitung bobot kelas otomatis `class_weights_dict` menggunakan `compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)` untuk digunakan pada parameter `class_weight` di *Logistic Regression* dan fungsi *Binary Cross-Entropy Loss* di *Multilayer Perceptron* (MLP).
   - Menghitung parameter `scale_pos_weight` $= \frac{N_{\text{neg, train}}}{N_{\text{pos, train}}}$ untuk dimasukkan ke dalam konfigurasi *XGBoost Classifier*.
2. **SMOTE Oversampling (Eksperimen Iterasi Lifecycle):**
   - Menerapkan `SMOTE(sampling_strategy=0.5, random_state=42)` hanya pada `X_train_scaled` (`X_train_smote`, `y_train_smote`) guna menguji dampak penambahan sampel sintetis kelas minoritas terhadap *Recall* dan *PR-AUC*, sementara `X_val_scaled` dan `X_test_scaled` dipertahankan murni sesuai distribusi asli.

### 5.6 Ringkasan Data Preparation

1. **Apa yang dilakukan:**
   Pembersihan *noise* sensor (`confidence != 'low'`), eksklusi area kebun sawit pra-kebakaran (`pre_existing_palm_flag == 0`), deduplikasi spasio-temporal pada grid $500\text{ m}$, rekayasa fitur jarak infrastruktur (`cKDTree`), rasio termal, indikator kekeringan, dan siklikal bulan, dilanjutkan dengan *Stratified Split* ($70:15:15$), imputasi median per provinsi, *winsorizing* persentil ke-99 + $\log(1+x)$, standardisasi `StandardScaler`, serta perhitungan `class_weight` dan `SMOTE`.
2. **Mengapa penting:**
   - Membuang titik api pada kebun sawit lama serta memisahkan data sebelum imputasi dan *scaling* memastikan eksperimen **100% bebas dari Data Leakage**.
   - Transformasi $\log(1+x)$ dan `StandardScaler` menstabilkan distribusi fitur yang miring (*skewed*) sehingga mempercepat konvergensi gradien pada *Logistic Regression* dan *Deep MLP*.
   - Penerapan `class_weight` / `scale_pos_weight` menjaga sensitivitas (*Recall*) model terhadap kelas minoritas ($Y=1$).
3. **Bagaimana implementasinya:**
   Seluruh tahapan diimplementasikan secara reproducible (`random_state=42`) di dalam script `eda_dan_prep.py` menggunakan `pandas`, `numpy`, `scipy.spatial.cKDTree`, `scikit-learn` (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`, `train_test_split`, `compute_class_weight`), dan `imbalanced-learn` (`SMOTE`), serta disimpan sebagai artefak `data/prepared_splits.joblib`.
