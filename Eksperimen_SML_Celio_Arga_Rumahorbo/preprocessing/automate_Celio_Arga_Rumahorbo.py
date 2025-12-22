import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. FUNGSI LOGIKA
# ==========================================

def load_data(file_path):
    """Load data dari CSV."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File tak ditemukan: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Data dimuat. Shape: {df.shape}")
    return df

def handle_missing_values(df):
    """Imputasi missing value numerik dengan median."""
    if df.isnull().sum().sum() > 0:
        print("[INFO] Handling missing values...")
        cols = df.select_dtypes(include=['number']).columns
        for col in cols:
            df[col] = df[col].fillna(df[col].median())
    return df

def handle_outliers_iqr(df, target_col):
    """Handle outliers menggunakan metode IQR Capping."""
    print("[INFO] Handling outliers (IQR)...")
    cols = df.select_dtypes(include=['number']).columns
    
    for col in cols:
        if col == target_col: continue # Skip target
            
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        
        # Capping nilai ekstrim
        df[col] = np.where(df[col] < low, low, df[col])
        df[col] = np.where(df[col] > high, high, df[col])
        
    return df

def feature_engineering(df, target_col):
    """Scaling fitur (StandardScaler) & re-structure dataframe."""
    print("[INFO] Scaling fitur...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scaling
    X_scaled = StandardScaler().fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Gabung kembali X dan y
    return pd.concat([X_df, y.reset_index(drop=True)], axis=1)

# ==========================================
# 2. PIPELINE UTAMA
# ==========================================

def run_automation_pipeline(input_path, output_path, target_col):
    """Eksekusi seluruh proses preprocessing."""
    print(f"{'='*40}\n   START AUTOMATION\n{'='*40}")
    
    # 1. Load
    df = load_data(input_path)
    if df is None: return
    
    # 2. Process
    df = handle_missing_values(df)
    df = handle_outliers_iqr(df, target_col)
    df_clean = feature_engineering(df, target_col)
    
    # 3. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Buat folder jika belum ada
    df_clean.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Saved to: {output_path}")
    print(f"[INFO] Final Shape: {df_clean.shape}\n{'='*40}")

# ==========================================
# 3. EKSEKUSI
# ==========================================

if __name__ == "__main__":
    # --- CONFIG ---
    RAW_FILE = 'winequality-red.csv'
    CLEAN_FILE = 'winequality_clean.csv'
    TARGET = 'quality'

    # Ambil path file dari folder utama workspace
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    in_csv = os.path.join(workspace_dir, RAW_FILE)
    out_csv = os.path.join(os.path.dirname(__file__), CLEAN_FILE)

    # Run
    run_automation_pipeline(in_csv, out_csv, TARGET)