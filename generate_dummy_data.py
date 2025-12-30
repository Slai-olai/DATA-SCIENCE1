"""
Script untuk generate dummy data sebagai contoh testing dashboard
Gunakan ini jika Anda belum memiliki data asli
"""

import pandas as pd
import numpy as np

# Set random seed untuk reproducibility
np.random.seed(42)

# ============================================================================
# GENERATE DUMMY DATA TKDD
# ============================================================================

print("Generating dummy TKDD data...")

# Definisi
tahun = [2020, 2021, 2022, 2023, 2024]
provinsi = ['Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Banten', 'DKI Jakarta']
n_kabkota_per_prov = 10  # 10 kab/kota per provinsi

data_tkdd = []

for prov in provinsi:
    for i in range(n_kabkota_per_prov):
        kabkota = f"{prov} - Kabupaten {i+1}"
        
        for thn in tahun:
            data_tkdd.append({
                'Tahun': thn,
                'Pemda': kabkota,
                'Provinsi': prov,
                'DBH PAJAK': np.random.randint(50000000, 500000000),
                'DBH SDA': np.random.randint(10000000, 200000000),
                'DAU': np.random.randint(500000000, 2000000000),
                'DAK Fisik': np.random.randint(100000000, 800000000),
                'DAK Nonfisik': np.random.randint(50000000, 300000000),
                'Dana Desa': np.random.randint(200000000, 1000000000),
                # Kolom dummy yang akan di-drop
                'DID Reguler': np.random.randint(10000000, 100000000),
                'Otsus': 0,
                'Hibah': np.random.randint(5000000, 50000000)
            })

df_tkdd = pd.DataFrame(data_tkdd)

# Save
df_tkdd.to_csv('dummy_tkdd.csv', index=False, encoding='latin1')
print(f"✓ Dummy TKDD data created: {len(df_tkdd)} rows")
print(f"  Columns: {list(df_tkdd.columns)}")

# ============================================================================
# GENERATE DUMMY DATA SOSIAL EKONOMI
# ============================================================================

print("\nGenerating dummy Sosial Ekonomi data...")

data_sosial = []

for prov in provinsi:
    for i in range(n_kabkota_per_prov):
        kabkota = f"{prov} - Kabupaten {i+1}"
        
        # Base values yang akan berfluktuasi per tahun
        base_ipm = np.random.uniform(60, 80)
        base_sekolah = np.random.uniform(7, 12)
        base_pdrb = np.random.uniform(50000000000, 200000000000)
        base_miskin = np.random.uniform(5, 25)
        
        for idx, thn in enumerate(tahun):
            # Tambahkan trend: IPM naik, kemiskinan turun
            trend_factor = idx * 0.02  # 2% improvement per year
            
            data_sosial.append({
                'Tahun': thn,
                'Kabupaten/kota': kabkota,
                'Provinsi': prov,
                'ID Provinsi': provinsi.index(prov) + 1,
                'Index Pembangunan Manusia': round(base_ipm * (1 + trend_factor), 2),
                'Rata-rata Lama Sekolah': round(base_sekolah * (1 + trend_factor), 2),
                'PDRB': round(base_pdrb * (1 + trend_factor * 2), 2),
                'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)': round(base_miskin * (1 - trend_factor), 2),
                # Kolom dummy yang akan di-drop
                'Pengeluaran Konsumsi Rumah Tangga': np.random.randint(1000000000, 5000000000),
                'Pengeluaran Konsumsi Pemerintah': np.random.randint(500000000, 2000000000),
                'Pembentukan Modal Tetap Bruto': np.random.randint(1000000000, 3000000000)
            })

df_sosial = pd.DataFrame(data_sosial)

# Save
df_sosial.to_csv('dummy_sosial.csv', index=False, encoding='latin1')
print(f"✓ Dummy Sosial Ekonomi data created: {len(df_sosial)} rows")
print(f"  Columns: {list(df_sosial.columns)}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DUMMY DATA GENERATION COMPLETE")
print("="*80)

print(f"\nFiles created:")
print(f"  1. dummy_tkdd.csv ({len(df_tkdd)} rows)")
print(f"  2. dummy_sosial.csv ({len(df_sosial)} rows)")

print(f"\nData coverage:")
print(f"  - Years: {min(tahun)} - {max(tahun)}")
print(f"  - Provinces: {len(provinsi)}")
print(f"  - Kab/Kota per province: {n_kabkota_per_prov}")
print(f"  - Total Kab/Kota: {len(provinsi) * n_kabkota_per_prov}")

print(f"\n📊 You can now upload these files to the dashboard for testing!")

print("\nPreview TKDD:")
print(df_tkdd.head())

print("\nPreview Sosial Ekonomi:")
print(df_sosial.head())

# ============================================================================
# VERIFY MERGE
# ============================================================================

print("\n" + "="*80)
print("VERIFICATION: Checking if data can be merged")
print("="*80)

# Rename untuk merge
df_tkdd_test = df_tkdd.rename(columns={'Pemda': 'Kabupaten/kota'})

# Merge test
df_merged_test = pd.merge(
    df_tkdd_test,
    df_sosial,
    on=['Tahun', 'Kabupaten/kota'],
    how='inner'
)

print(f"\n✓ Merge successful!")
print(f"  Merged rows: {len(df_merged_test)}")
print(f"  Expected rows: {len(df_tkdd)}")
print(f"  Match: {'YES' if len(df_merged_test) == len(df_tkdd) else 'NO'}")

if len(df_merged_test) == len(df_tkdd):
    print("\n✅ Data is ready for dashboard testing!")
else:
    print("\n⚠️ Warning: Some rows were lost during merge. Check data integrity.")
