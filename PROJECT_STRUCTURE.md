# 📁 Struktur Project Dashboard

```
dashboard-tkdd-kemiskinan/
│
├── 📄 app.py                      # File utama dashboard Streamlit
├── 📄 requirements.txt            # Dependencies Python
├── 📄 README.md                   # Dokumentasi lengkap
├── 📄 QUICKSTART.md               # Panduan cepat
├── 📄 generate_dummy_data.py      # Script generate data testing
│
├── 📊 Data/ (opsional)
│   ├── dummy_tkdd.csv            # Contoh data TKDD
│   ├── dummy_sosial.csv          # Contoh data Sosial Ekonomi
│   ├── Merge_Realisasi_TKDD_2020-2024_38Prov_Fixed.csv (data asli)
│   └── Merge_Indikator Sosial Ekonomi Kabupaten kota_2020-2024.csv (data asli)
│
└── 📸 Screenshots/ (opsional)
    ├── overview.png
    ├── eda.png
    ├── modeling.png
    └── evaluation.png
```

## 📄 Deskripsi File

### Core Files (Wajib)

#### `app.py`
**File utama dashboard Streamlit**
- 1000+ baris kode Python
- Struktur modular dengan 5 tab utama
- Fungsi-fungsi helper untuk:
  - Load dan preprocessing data
  - Training model ML
  - Visualisasi
  - Evaluasi

**Sections:**
1. Imports & Configuration
2. Helper Functions
3. Sidebar Settings
4. Main Content (5 tabs)
5. Footer

#### `requirements.txt`
**Dependencies yang diperlukan**
```
streamlit          # Framework dashboard
pandas             # Data manipulation
numpy              # Numerical computing
matplotlib         # Plotting
seaborn            # Statistical visualization
scipy              # Scientific computing
scikit-learn       # Machine learning
statsmodels        # Statistical modeling
```

### Documentation Files

#### `README.md`
**Dokumentasi lengkap project**
- Overview fitur
- Cara instalasi dan running
- Struktur data yang diperlukan
- Penjelasan model ML
- Use cases
- Troubleshooting

#### `QUICKSTART.md`
**Panduan cepat untuk mulai dalam 5 menit**
- Langkah instalasi
- Cara upload data
- Checklist data
- Tips troubleshooting
- Workflow rekomendasi

### Utility Files

#### `generate_dummy_data.py`
**Script untuk generate data testing**
- Generate dummy TKDD data
- Generate dummy Sosial Ekonomi data
- Verifikasi merge compatibility
- Output: 2 file CSV

**Usage:**
```bash
python generate_dummy_data.py
```

**Output:**
- `dummy_tkdd.csv` (250 rows)
- `dummy_sosial.csv` (250 rows)

## 🚀 Setup Project

### Step 1: Clone atau Download
```bash
# Jika dari git
git clone <repository-url>
cd dashboard-tkdd-kemiskinan

# Atau extract dari zip
unzip dashboard-tkdd-kemiskinan.zip
cd dashboard-tkdd-kemiskinan
```

### Step 2: Install Dependencies
```bash
# Menggunakan pip
pip install -r requirements.txt

# Atau menggunakan virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Step 3: Prepare Data

**Option A: Gunakan Data Asli**
- Letakkan file CSV di folder `Data/`
- Upload via dashboard sidebar

**Option B: Generate Dummy Data**
```bash
python generate_dummy_data.py
```
- File `dummy_tkdd.csv` dan `dummy_sosial.csv` akan dibuat
- Upload via dashboard sidebar

### Step 4: Run Dashboard
```bash
streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`

## 🎯 Feature Checklist

### ✅ Implemented Features

**Data Management:**
- [x] Upload CSV via sidebar
- [x] Automatic preprocessing
- [x] Missing value handling
- [x] Data type conversion
- [x] Merge multiple datasets

**Exploratory Data Analysis:**
- [x] Descriptive statistics
- [x] Distribution plots (histogram, boxplot, violin, Q-Q)
- [x] Temporal trend analysis
- [x] Regional ranking (top/bottom)
- [x] Correlation heatmap
- [x] Scatter plots with regression lines

**Machine Learning:**
- [x] Train-test split (customizable)
- [x] Feature scaling (StandardScaler)
- [x] Linear Regression
- [x] Random Forest Regressor
- [x] Gradient Boosting Regressor
- [x] Model comparison

**Evaluation:**
- [x] Multiple metrics (R², RMSE, MAE, MAPE)
- [x] Actual vs Predicted plots
- [x] Residual analysis
- [x] Feature importance (tree-based models)
- [x] Model comparison table
- [x] Best model recommendation

**User Interface:**
- [x] Responsive layout
- [x] Interactive widgets
- [x] Custom CSS styling
- [x] Progress indicators
- [x] Error handling
- [x] Help tooltips

### 🔄 Potential Enhancements (Future)

**Data:**
- [ ] Support Excel files (.xlsx)
- [ ] Export processed data
- [ ] Data versioning
- [ ] Cache optimization

**Analysis:**
- [ ] More ML models (XGBoost, LightGBM)
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Feature engineering advanced
- [ ] Time series forecasting

**Visualization:**
- [ ] Interactive plots (Plotly)
- [ ] Download plots as PNG/PDF
- [ ] Custom color schemes
- [ ] Geographic maps (folium)

**Features:**
- [ ] User authentication
- [ ] Save analysis sessions
- [ ] Export reports (PDF)
- [ ] API integration
- [ ] Real-time data updates

## 📊 Data Flow

```
┌─────────────────┐
│  Upload CSV     │
│  (TKDD +        │
│   Sosial)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Clean        │
│  - Transform    │
│  - Merge        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EDA            │
│  - Statistics   │
│  - Visualization│
│  - Correlation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Engineering    │
│  - Selection    │
│  - Scaling      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model          │
│  Training       │
│  - LR           │
│  - RF           │
│  - GB           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │
│  - Metrics      │
│  - Visualization│
│  - Comparison   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Insights &     │
│  Recommendations│
└─────────────────┘
```

## 🔧 Customization Guide

### Menambah Model Baru

**Location:** `app.py` → function `train_models()`

```python
# Tambahkan di function train_models()

# 4. Model Baru
new_model = YourModelClass(
    param1=value1,
    param2=value2
)
new_model.fit(X_train, y_train)
y_pred_new = new_model.predict(X_test)

results['Your Model Name'] = {
    'model': new_model,
    'y_pred': y_pred_new,
    'r2': r2_score(y_test, y_pred_new),
    # ... metrics lainnya
}
```

### Mengubah Tampilan

**Location:** `app.py` → Custom CSS section

```python
st.markdown("""
    <style>
    /* Tambahkan CSS custom Anda di sini */
    .custom-class {
        /* Your styles */
    }
    </style>
""", unsafe_allow_html=True)
```

### Menambah Visualisasi

**Location:** `app.py` → Tab EDA atau Evaluation

```python
# Contoh tambah plot baru
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
st.pyplot(fig)
```

## 📝 Code Quality

**Standards:**
- ✅ PEP 8 compliant
- ✅ Docstrings untuk functions
- ✅ Type hints (opsional)
- ✅ Error handling dengan try-catch
- ✅ Modular design
- ✅ Caching untuk performa

**Best Practices:**
- Gunakan `@st.cache_data` untuk fungsi yang load data
- Gunakan `st.session_state` untuk menyimpan state
- Pisahkan logic dan presentation
- Handle errors gracefully
- Provide user feedback (progress bars, success messages)

## 🐛 Common Issues & Solutions

### Issue 1: Module not found
```bash
pip install -r requirements.txt --upgrade
```

### Issue 2: Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Issue 3: CSV encoding error
Save CSV dengan encoding `latin-1`:
```python
df.to_csv('file.csv', encoding='latin-1')
```

### Issue 4: Memory error (large dataset)
Reduce data atau increase RAM. Consider sampling:
```python
df_sample = df.sample(frac=0.5)  # Use 50% of data
```

## 📚 Resources

**Streamlit Documentation:**
- https://docs.streamlit.io

**Scikit-learn Documentation:**
- https://scikit-learn.org/stable/documentation.html

**Pandas Documentation:**
- https://pandas.pydata.org/docs/

**Matplotlib/Seaborn:**
- https://matplotlib.org/stable/contents.html
- https://seaborn.pydata.org/tutorial.html

## 🤝 Contributing

Untuk contribute:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

[Specify your license here]

---

**Project Structure Created with ❤️ for Data Analysis**
