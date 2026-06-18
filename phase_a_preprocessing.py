"""
=============================================================================
  DEBUTANIZER SOFT SENSOR PROJECT
  Phase A: Data Preprocessing & Feature Engineering
  Author: AI Senior ML Engineer
=============================================================================
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------------------
print("=" * 65)
print("  PHASE A — DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 65)

RAW_FILE   = "debutanizer_cleaned_v1.csv"
OUT_FILE   = "debutanizer_model_ready.csv"

df = pd.read_csv(RAW_FILE)
print(f"\n[1] Raw CSV loaded  ->  shape: {df.shape}")
print(f"    Columns: {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 2. TIME INDEXING
# ---------------------------------------------------------------------------
df.rename(columns={"Unnamed: 0": "Timestamp"}, inplace=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.set_index("Timestamp", inplace=True)

print(f"\n[2] Timestamp index set")
print(f"    Index dtype : {df.index.dtype}")
print(f"    Date range  : {df.index.min()}  ->  {df.index.max()}")
freq = pd.infer_freq(df.index)
print(f"    Frequency   : {freq}")

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING - THERMODYNAMIC RATIOS
# ---------------------------------------------------------------------------
# Guard against division-by-zero with a small epsilon
EPSILON = 1e-9

df["Reflux_Ratio"] = (
    df["Reflux flow"] / (df["Feed Flow to DB"] + EPSILON)
)

df["Temp_Diff_Bottom_Top"] = (
    df["Column bottom temp"] - df["Column top Temp"]
)

df["Steam_to_Feed_Ratio"] = (
    df["Reboiling steam flow"] / (df["Feed Flow to DB"] + EPSILON)
)

print(f"\n[3] Engineered features created:")
print(f"    * Reflux_Ratio          - range [{df['Reflux_Ratio'].min():.4f}, "
      f"{df['Reflux_Ratio'].max():.4f}]")
print(f"    * Temp_Diff_Bottom_Top  - range [{df['Temp_Diff_Bottom_Top'].min():.4f}, "
      f"{df['Temp_Diff_Bottom_Top'].max():.4f}]")
print(f"    * Steam_to_Feed_Ratio   - range [{df['Steam_to_Feed_Ratio'].min():.4f}, "
      f"{df['Steam_to_Feed_Ratio'].max():.4f}]")

# ---------------------------------------------------------------------------
# 4. TARGET VARIABLE CLEANUP — drop redundant component columns
# ---------------------------------------------------------------------------
REDUNDANT_COLS = ["C4H6 in DB bottom", "C4H8 in DB bottom"]
df.drop(columns=REDUNDANT_COLS, inplace=True)

print(f"\n[4] Dropped redundant columns: {REDUNDANT_COLS}")

# ---------------------------------------------------------------------------
# 5. VALIDATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  VALIDATION REPORT")
print("=" * 65)

# 5-a  Shape
print(f"\n[5a] Final DataFrame shape : {df.shape}")
print(f"     Columns ({len(df.columns)}) : {df.columns.tolist()}")

# 5-b  Null check
null_counts = df.isnull().sum()
total_nulls  = null_counts.sum()
print(f"\n[5b] Null value check (total nulls = {total_nulls}):")
print(null_counts.to_string())

if total_nulls == 0:
    print("\n     [OK]  No nulls detected -- data is clean.")
else:
    print("\n     [!!]  Nulls detected -- review before modelling!")

# 5-c  First 5 rows focusing on engineered features + target
FOCUS_COLS = [
    "Reflux_Ratio",
    "Temp_Diff_Bottom_Top",
    "Steam_to_Feed_Ratio",
    "Total_C4_Slippage_Wt",
]
print(f"\n[5c] First 5 rows - engineered features + target:")
print(df[FOCUS_COLS].head(5).to_string())

# 5-d  Descriptive statistics for engineered features
print(f"\n[5d] Descriptive statistics - engineered features + target:")
print(df[FOCUS_COLS].describe().round(4).to_string())

# ---------------------------------------------------------------------------
# 6. EXPORT
# ---------------------------------------------------------------------------
df.to_csv(OUT_FILE)
print(f"[6] [DONE]  Model-ready DataFrame exported  ->  '{OUT_FILE}'")
print(f"    Rows : {len(df):,}  |  Columns : {len(df.columns)}")
print("\n" + "=" * 65)
print("  PHASE A COMPLETE - Awaiting your verification before Phase B.")
print("=" * 65 + "\n")
