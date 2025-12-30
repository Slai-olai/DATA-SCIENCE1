# 📊 Dashboard Analisis TKDD & Kemiskinan Indonesia

Dashboard interaktif untuk menganalisis pengaruh Transfer Ke Daerah dan Desa (TKDD) dan Indikator Sosial Ekonomi terhadap Tingkat Kemiskinan di Kabupaten/Kota Indonesia (2020-2024).

## 🎯 Fitur Utama

### 1. 📊 Overview Data
- Statistik deskriptif dataset
- Preview data dan informasi dasar
- Distribusi data per tahun
- Ringkasan variabel target (kemiskinan)

### 2. 🔍 Exploratory Data Analysis (EDA)
- **Distribusi Kemiskinan**: Histogram, boxplot, Q-Q plot, violin plot
- **Trend Temporal**: Analisis perubahan kemiskinan 2020-2024
- **Ranking Daerah**: Top 10 kabupaten/kota dengan kemiskinan tertinggi/terendah
- **Analisis Korelasi**: Heatmap dan scatter plot variabel penting

### 3. 🤖 Machine Learning
- **3 Model Prediktif**:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Feature engineering dan scaling
- Train-test split dengan parameter yang dapat disesuaikan

### 4. 📈 Evaluasi Model
- Perbandingan performa model (R², RMSE, MAE, MAPE)
- Visualisasi Actual vs Predicted
- Residual analysis
- Feature importance analysis

### 5. 💡 Insight & Kesimpulan
- Temuan utama dari analisis
- Variabel paling berpengaruh
- Rekomendasi kebijakan
- Insight untuk pengambil keputusan

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Dashboard
```bash
streamlit run app.py
```

### 3. Akses Dashboard
Browser akan otomatis terbuka di `http://localhost:8501`

## 📁 Struktur Data

### Dataset TKDD (CSV)
Kolom yang diperlukan:
- `Tahun`: Tahun data (2020-2024)
- `Pemda`: Nama Kabupaten/Kota
- `Provinsi`: Nama Provinsi
- `DBH PAJAK`: Dana Bagi Hasil Pajak
- `DBH SDA`: Dana Bagi Hasil Sumber Daya Alam
- `DAU`: Dana Alokasi Umum
- `DAK Fisik`: Dana Alokasi Khusus Fisik
- `DAK Nonfisik`: Dana Alokasi Khusus Non-Fisik
- `Dana Desa`: Dana Desa

### Dataset Sosial Ekonomi (CSV)
Kolom yang diperlukan:
- `Tahun`: Tahun data (2020-2024)
- `Kabupaten/kota`: Nama Kabupaten/Kota
- `Index Pembangunan Manusia`: Nilai IPM
- `Rata-rata Lama Sekolah`: Rata-rata lama sekolah (tahun)
- `PDRB`: Produk Domestik Regional Bruto
- `Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)`: Target variabel

**Format:** CSV dengan encoding `latin-1`

## 🎛️ Konfigurasi

### Sidebar Settings
- **Upload Dataset**: Upload file CSV untuk TKDD dan Sosial Ekonomi
- **Parameter Modeling**: 
  - Test Size: 10-40% (default: 20%)
  - Random State: 0-100 (default: 42)
- **Model Selection**: Pilih model yang akan dijalankan

## 📊 Model Machine Learning

### 1. Linear Regression
- Baseline model untuk regresi linear
- Menggunakan StandardScaler untuk normalisasi
- Memberikan koefisien regresi untuk interpretasi

### 2. Random Forest Regressor
- Ensemble learning dengan 100 decision trees
- Max depth: 15
- Menghasilkan feature importance
- Robust terhadap outliers

### 3. Gradient Boosting Regressor
- Sequential ensemble learning
- Learning rate: 0.1
- Max depth: 5
- Performa tinggi dengan regularisasi

## 📈 Metrik Evaluasi

- **R² Score**: Coefficient of determination (0-1, semakin tinggi semakin baik)
- **RMSE**: Root Mean Squared Error (semakin rendah semakin baik)
- **MAE**: Mean Absolute Error (semakin rendah semakin baik)
- **MAPE**: Mean Absolute Percentage Error (%, semakin rendah semakin baik)

## 🔍 Analisis yang Dilakukan

### Preprocessing
1. Cleaning missing values (median untuk numerik, mode untuk kategorikal)
2. Konversi tipe data
3. Drop kolom yang tidak relevan
4. Merge dataset TKDD dan Sosial Ekonomi
5. Handling outliers

### Feature Engineering
9 fitur independen:
1. DBH PAJAK
2. DBH SDA
3. DAU
4. DAK Fisik
5. DAK Nonfisik
6. Dana Desa
7. Index Pembangunan Manusia
8. Rata-rata Lama Sekolah
9. PDRB

Target: Persentase Penduduk Miskin

### Visualisasi
- Histogram & Density Plot
- Boxplot untuk outlier detection
- Q-Q Plot untuk normalitas
- Violin Plot per tahun
- Time series trend
- Correlation heatmap
- Scatter plot dengan regression line
- Feature importance charts
- Actual vs Predicted plot
- Residual plot

## 💡 Use Cases

### Untuk Pemerintah Daerah
- Evaluasi efektivitas program TKDD
- Identifikasi daerah prioritas intervensi
- Monitoring trend kemiskinan
- Benchmarking antar daerah

### Untuk Peneliti
- Analisis faktor-faktor kemiskinan
- Validasi hipotesis penelitian
- Eksplorasi data yang interaktif
- Feature importance analysis

### Untuk Pembuat Kebijakan
- Data-driven decision making
- Alokasi anggaran yang optimal
- Evaluasi dampak program sosial
- Proyeksi tingkat kemiskinan

## 🛠️ Teknologi

- **Python 3.8+**
- **Streamlit**: Framework dashboard
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning
- **Matplotlib & Seaborn**: Visualisasi
- **SciPy**: Statistical analysis
- **Statsmodels**: Advanced statistics

## 📝 Catatan Penting

1. **Data Quality**: Pastikan data CSV sudah bersih dan sesuai format
2. **Encoding**: Gunakan encoding `latin-1` untuk file CSV Indonesia
3. **Memory**: Dataset besar mungkin memerlukan RAM yang cukup
4. **Reproducibility**: Gunakan random state yang sama untuk hasil konsisten
5. **Interpretasi**: Hasil model adalah prediksi, perlu validasi dengan domain expert

## 🤝 Kontribusi

Dashboard ini dibuat berdasarkan notebook analisis Python dengan struktur:
- Load & preprocessing data
- Exploratory Data Analysis
- Feature engineering
- Model training & evaluation
- Insight generation

## 📄 Lisensi

Dashboard ini dibuat untuk keperluan analisis data dan penelitian.

## 📞 Support

Untuk pertanyaan atau masalah teknis, silakan:
1. Periksa format data CSV
2. Pastikan semua dependencies terinstall
3. Cek console untuk error messages

---

**Selamat Menganalisis! 📊🚀**
