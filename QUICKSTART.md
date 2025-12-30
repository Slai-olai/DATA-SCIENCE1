# 🚀 Quick Start Guide

## Langkah Cepat (5 Menit)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Jalankan Dashboard
```bash
streamlit run app.py
```

### 3️⃣ Upload Data
1. Buka sidebar (ikon > di kiri atas)
2. Upload **Dataset TKDD** (CSV)
3. Upload **Dataset Sosial Ekonomi** (CSV)

### 4️⃣ Eksplorasi!
Navigasi melalui 5 tab:
- 📊 **Overview Data**: Lihat ringkasan dataset
- 🔍 **EDA**: Analisis mendalam dengan visualisasi
- 🤖 **Machine Learning**: Klik "Jalankan Modeling"
- 📈 **Evaluasi**: Lihat performa model
- 💡 **Insight**: Baca kesimpulan dan rekomendasi

## ⚙️ Konfigurasi Sidebar

### Parameter yang Bisa Disesuaikan:
- **Test Size**: 10-40% (rekomendasi: 20%)
- **Random State**: Untuk reproducibility (default: 42)
- **Model Selection**: Pilih model yang ingin dijalankan

## 📋 Checklist Data

Pastikan file CSV Anda memiliki kolom-kolom ini:

### ✅ Dataset TKDD:
- [ ] Tahun
- [ ] Pemda (atau Kabupaten/kota)
- [ ] Provinsi
- [ ] DBH PAJAK
- [ ] DBH SDA
- [ ] DAU
- [ ] DAK Fisik
- [ ] DAK Nonfisik
- [ ] Dana Desa

### ✅ Dataset Sosial Ekonomi:
- [ ] Tahun
- [ ] Kabupaten/kota
- [ ] Index Pembangunan Manusia
- [ ] Rata-rata Lama Sekolah
- [ ] PDRB
- [ ] Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)

## 🔧 Troubleshooting

### Error "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Error "File encoding"
Pastikan CSV menggunakan encoding `latin-1`:
```python
df.to_csv('file.csv', encoding='latin-1', index=False)
```

### Dashboard tidak muncul
1. Check port 8501 tidak digunakan aplikasi lain
2. Coba port lain:
```bash
streamlit run app.py --server.port 8502
```

### Data tidak muncul setelah upload
1. Periksa format kolom sesuai checklist
2. Pastikan tidak ada missing values di kolom kunci (Tahun, Kabupaten/kota)
3. Check console untuk error messages

## 💡 Tips Penggunaan

### Untuk Hasil Optimal:
1. **Upload data lengkap** - Semua tahun 2020-2024
2. **Jalankan semua model** - Bandingkan performanya
3. **Perhatikan feature importance** - Identifikasi variabel kunci
4. **Baca insight** - Lihat rekomendasi kebijakan

### Workflow Rekomendasi:
```
Upload Data → Overview → EDA → Modeling → Evaluasi → Insight
```

## 📊 Interpretasi Hasil

### R² Score:
- **> 0.90**: Excellent (90% variasi dijelaskan model)
- **0.80-0.90**: Very Good
- **0.70-0.80**: Good
- **< 0.70**: Need Improvement

### RMSE/MAE:
Semakin kecil semakin baik. Bandingkan dengan rentang nilai target.

### MAPE:
- **< 10%**: Excellent
- **10-20%**: Good
- **20-50%**: Reasonable
- **> 50%**: Inaccurate

## 🎯 Use Cases Cepat

### Skenario 1: Evaluasi Program
1. Upload data
2. Lihat tab "EDA" → Trend temporal
3. Bandingkan tahun 2020 vs 2024

### Skenario 2: Identifikasi Daerah Prioritas
1. Upload data
2. Tab "EDA" → Scroll ke "Top/Bottom Regions"
3. Lihat 10 kabupaten/kota dengan kemiskinan tertinggi

### Skenario 3: Prediksi Kemiskinan
1. Upload data
2. Tab "Machine Learning" → Jalankan modeling
3. Tab "Evaluasi" → Lihat actual vs predicted

### Skenario 4: Analisis Faktor Kunci
1. Jalankan modeling (Random Forest/Gradient Boosting)
2. Tab "Evaluasi" → Pilih model → Lihat Feature Importance
3. Fokus pada top 3 variabel

## 📞 Need Help?

### Urutan Debugging:
1. ✅ Check semua dependencies terinstall
2. ✅ Verify format data CSV
3. ✅ Check encoding file (latin-1)
4. ✅ Lihat error di console
5. ✅ Baca dokumentasi README.md

---

**Happy Analyzing! 📊✨**
