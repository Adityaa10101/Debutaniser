"""
=============================================================================
  DEBUTANIZER SOFT SENSOR PROJECT
  Phase B-Revised: Temporal Feature Engineering & XGBoost Retraining
  Fix: Lag features + rolling statistics to resolve negative R2
  Author: AI Senior ML Engineer
=============================================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE   = "debutanizer_model_ready.csv"
LAG_CSV_OUT  = "debutanizer_features_with_lags.csv"
MODEL_FILE   = "xgb_debutanizer_model_v2.json"
TARGET_COL   = "Total_C4_Slippage_Wt"
TRAIN_RATIO  = 0.80
RANDOM_STATE = 42
N_CV_SPLITS  = 5
SEP = "=" * 70

# Top 5 physical drivers from Phase B feature importance ranking
TOP_DRIVERS = [
    "Steam_to_Feed_Ratio",
    "Column Top pressure",
    "Column bottom temp",
    "Reflux flow",
    "Control tay temp",
]

print(SEP)
print("  PHASE B-REVISED  --  Temporal Feature Engineering & Retraining")
print(SEP)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading model-ready dataset ...")
df = pd.read_csv(INPUT_FILE, index_col="Timestamp", parse_dates=True)
df.sort_index(inplace=True)

print(f"    Loaded shape  : {df.shape}")
print(f"    Date range    : {df.index.min()} -> {df.index.max()}")
print(f"    Sampling freq : hourly (inferred)")

original_cols = df.columns.tolist()

# ============================================================================
# 2. TEMPORAL FEATURE ENGINEERING
# ============================================================================
print("\n[2] Engineering temporal features ...")

new_features = []

# --- 2a. Lag features for the TARGET (t-1, t-2, t-3) ---
for lag in [1, 2, 3]:
    col = f"{TARGET_COL}_lag{lag}"
    df[col] = df[TARGET_COL].shift(lag)
    new_features.append(col)

print(f"    Target lags created    : {TARGET_COL} at t-1, t-2, t-3")

# --- 2b. Lag features at t-1 for the top 5 physical drivers ---
for driver in TOP_DRIVERS:
    col = f"{driver}_lag1"
    df[col] = df[driver].shift(1)
    new_features.append(col)

print(f"    Driver lags created    : t-1 for {len(TOP_DRIVERS)} top drivers")

# --- 2c. 3-hour rolling mean for the same top 5 drivers ---
# min_periods=1 would allow partial windows but we will drop NaNs anyway;
# using min_periods=3 to enforce a full window before computing the mean.
for driver in TOP_DRIVERS:
    col = f"{driver}_roll3mean"
    df[col] = df[driver].shift(1).rolling(window=3, min_periods=3).mean()
    new_features.append(col)

print(f"    Rolling 3h means done  : {len(TOP_DRIVERS)} features")
print(f"    Total new features     : {len(new_features)}")

# ============================================================================
# 3. DROP NaN ROWS CREATED BY SHIFTING / ROLLING
# ============================================================================
rows_before = len(df)
df.dropna(inplace=True)
rows_after  = len(df)
dropped     = rows_before - rows_after

print(f"\n[3] Dropped {dropped} NaN rows (from shifting/rolling)")
print(f"    Remaining rows : {rows_after:,}")

# Separate X and y
all_feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[all_feature_cols]
y = df[TARGET_COL]

print(f"    Final feature count : {len(all_feature_cols)}")
print(f"    Features: {all_feature_cols}")

# ============================================================================
# 4. CHRONOLOGICAL 80/20 SPLIT
# ============================================================================
split_idx = int(len(df) * TRAIN_RATIO)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n[4] Chronological 80/20 split:")
print(f"    Training : {len(X_train):>6,} rows  "
      f"({X_train.index.min()} -> {X_train.index.max()})")
print(f"    Test     : {len(X_test):>6,} rows  "
      f"({X_test.index.min()} -> {X_test.index.max()})")

# ============================================================================
# 5. MODEL DEFINITION & HYPERPARAMETER TUNING
# ============================================================================
print(f"\n[5] XGBoost + TimeSeriesSplit GridSearchCV ...")

base_xgb = xgb.XGBRegressor(
    objective    = "reg:squarederror",
    random_state = RANDOM_STATE,
    n_jobs       = -1,
    verbosity    = 0,
)

# Lean, focused grid anchored around Phase B best params (max_depth=7, lr=0.05)
# 3 x 3 x 2 x 2 = 36 combos x 5 folds = 180 fits  (~2-3 min)
param_grid = {
    "n_estimators"     : [200, 400, 600],
    "max_depth"        : [5, 7, 9],
    "learning_rate"    : [0.03, 0.05],
    "subsample"        : [0.8, 1.0],
    "colsample_bytree" : [0.8],
    "min_child_weight" : [1],
    "reg_alpha"        : [0, 0.1],
    "reg_lambda"       : [1],
}

n_combos = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["learning_rate"])
    * len(param_grid["subsample"])
    * len(param_grid["colsample_bytree"])
    * len(param_grid["min_child_weight"])
    * len(param_grid["reg_alpha"])
    * len(param_grid["reg_lambda"])
)
total_fits = n_combos * N_CV_SPLITS

print(f"    Param combinations : {n_combos}  x  {N_CV_SPLITS} CV folds  "
      f"=  {total_fits} total fits")
print(f"    Scoring            : neg_root_mean_squared_error")
print(f"    Running ... (this will take several minutes)\n")

tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

grid_search = GridSearchCV(
    estimator  = base_xgb,
    param_grid = param_grid,
    cv         = tscv,
    scoring    = "neg_root_mean_squared_error",
    refit      = True,
    verbose    = 1,
    n_jobs     = -1,
)

t0 = time.time()
grid_search.fit(X_train, y_train)
elapsed = time.time() - t0

print(f"\n    GridSearchCV done in {elapsed:.1f} seconds.")
print(f"\n    BEST PARAMETERS:")
for k, v in grid_search.best_params_.items():
    print(f"      {k:<22}: {v}")
print(f"\n    Best CV RMSE (train folds): "
      f"{-grid_search.best_score_:.6f} wt%")

# ============================================================================
# 6. EVALUATION ON THE HELD-OUT TEST SET
# ============================================================================
best_model = grid_search.best_estimator_

print(f"\n[6] Evaluating best model on held-out test set ...")
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = np.mean(np.abs(y_test - y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

residuals = y_test - y_pred

# Phase B baseline for comparison
BASELINE_RMSE = 0.381685
BASELINE_R2   = -0.072324
BASELINE_MAPE = 143.2597

rmse_delta = BASELINE_RMSE - rmse
r2_delta   = r2   - BASELINE_R2

print(f"\n    {'-'*52}")
print(f"    TEST SET PERFORMANCE  (Phase B-Revised)")
print(f"    {'-'*52}")
print(f"    RMSE  : {rmse:.6f} wt%    (was {BASELINE_RMSE:.6f}, delta {rmse_delta:+.6f})")
print(f"    R2    : {r2:.6f}           (was {BASELINE_R2:.6f}, delta {r2_delta:+.6f})")
print(f"    MAE   : {mae:.6f} wt%")
print(f"    MAPE  : {mape:.4f} %      (was {BASELINE_MAPE:.4f} %)")
print(f"    {'-'*52}")
print(f"    Actual    : mean={y_test.mean():.4f}  std={y_test.std():.4f}  "
      f"min={y_test.min():.4f}  max={y_test.max():.4f}")
print(f"    Predicted : mean={y_pred.mean():.4f}  std={y_pred.std():.4f}  "
      f"min={y_pred.min():.4f}  max={y_pred.max():.4f}")
print(f"    Residuals : mean={residuals.mean():.6f}  std={residuals.std():.6f}")
print(f"    {'-'*52}")

if r2 > 0.85:
    verdict = "[EXCELLENT] Model ready for dashboard deployment."
elif r2 > 0.70:
    verdict = "[GOOD] Acceptable for soft sensor deployment."
elif r2 > 0.50:
    verdict = "[FAIR] May need further tuning before deployment."
elif r2 > 0:
    verdict = "[POOR] Positive R2 but needs more work."
else:
    verdict = "[FAIL] R2 still negative -- revisit feature set."

print(f"\n    Verdict: {verdict}")

# Spec threshold check
SPEC = 0.5
above_actual    = (y_test > SPEC).sum()
above_predicted = (pd.Series(y_pred) > SPEC).sum()
print(f"\n    Spec check (C4 > {SPEC} wt%):")
print(f"      Actual violations    : {above_actual}/{len(y_test)} "
      f"({100*above_actual/len(y_test):.1f}%)")
print(f"      Predicted violations : {above_predicted}/{len(y_pred)} "
      f"({100*above_predicted/len(y_pred):.1f}%)")

# ============================================================================
# 5b. FEATURE IMPORTANCE (re-printed after evaluation for convenience)
# ============================================================================
print(f"\n[5b] Feature Importances (sorted highest -> lowest):")
print(f"    {'Rank':<6} {'Feature':<32} {'Importance':>12}  Bar")
print(f"    {'-'*72}")

importances  = best_model.feature_importances_
feat_imp_df  = (
    pd.DataFrame({"Feature": all_feature_cols, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

max_imp = feat_imp_df["Importance"].max()
for i, row in feat_imp_df.iterrows():
    bar = "#" * int(30 * row["Importance"] / max_imp)
    print(f"    {i+1:<6} {row['Feature']:<32} {row['Importance']:>12.6f}  {bar}")

# ============================================================================
# 7. EXPORT
# ============================================================================
print(f"\n[7] Saving outputs ...")

# 7a. Save model
best_model.save_model(MODEL_FILE)

# Sanity reload check
reload_model = xgb.XGBRegressor()
reload_model.load_model(MODEL_FILE)
sanity = reload_model.predict(X_test.iloc[[0]])[0]
match  = "OK" if abs(y_pred[0] - sanity) < 1e-4 else "MISMATCH"
print(f"    Model saved  : {MODEL_FILE}  [{match}]")

# 7b. Save the lag-enriched dataset
df.to_csv(LAG_CSV_OUT)
print(f"    Lag CSV saved: {LAG_CSV_OUT}  (shape: {df.shape})")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{SEP}")
print(f"  PHASE B-REVISED  --  FINAL SUMMARY")
print(f"{SEP}")
print(f"  Temporal features added  : {len(new_features)}")
print(f"    - Target lags          : t-1, t-2, t-3")
print(f"    - Driver lags (t-1)    : {len(TOP_DRIVERS)} drivers")
print(f"    - Rolling 3h means     : {len(TOP_DRIVERS)} drivers")
print(f"  Total features in model  : {len(all_feature_cols)}")
print(f"  Best Params:")
for k, v in grid_search.best_params_.items():
    print(f"    {k:<22}: {v}")
print(f"  {'-'*60}")
print(f"  METRIC        Phase-B (no lags)    Phase B-Revised")
print(f"  {'-'*60}")
print(f"  RMSE          {BASELINE_RMSE:.6f}          {rmse:.6f} wt%")
print(f"  R2            {BASELINE_R2:.6f}          {r2:.6f}")
print(f"  MAPE          {BASELINE_MAPE:.4f} %          {mape:.4f} %")
print(f"  {'-'*60}")
print(f"  Model file : {MODEL_FILE}")
print(f"  CSV file   : {LAG_CSV_OUT}")
print(f"{SEP}")
print(f"  {verdict}")
print(f"  PHASE B-REVISED COMPLETE.")
print(f"{SEP}\n")
