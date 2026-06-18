"""
=============================================================================
  DEBUTANIZER SOFT SENSOR PROJECT
  Phase B: XGBoost Model Training & Validation
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
MODEL_FILE   = "xgb_debutanizer_model.json"
TARGET_COL   = "Total_C4_Slippage_Wt"
TRAIN_RATIO  = 0.80
RANDOM_STATE = 42
N_CV_SPLITS  = 5           # TimeSeriesSplit folds on the training set

SEPARATOR = "=" * 70

print(SEPARATOR)
print("  PHASE B  --  XGBoost Model Training & Validation")
print(SEPARATOR)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading model-ready dataset ...")
df = pd.read_csv(INPUT_FILE, index_col="Timestamp", parse_dates=True)
df.sort_index(inplace=True)   # guarantee chronological order

feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols]
y = df[TARGET_COL]

print(f"    Dataset shape   : {df.shape}")
print(f"    Features ({len(feature_cols)}): {feature_cols}")
print(f"    Target          : {TARGET_COL}")
print(f"    Date range      : {df.index.min()} -> {df.index.max()}")

# ============================================================================
# 2. TIME-SERIES CHRONOLOGICAL SPLIT (NO SHUFFLE)
# ============================================================================
split_idx = int(len(df) * TRAIN_RATIO)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n[2] Chronological 80/20 split:")
print(f"    Training set : {len(X_train):>6,} rows  "
      f"({X_train.index.min()} -> {X_train.index.max()})")
print(f"    Test set     : {len(X_test):>6,} rows  "
      f"({X_test.index.min()} -> {X_test.index.max()})")

# ============================================================================
# 3. MODEL DEFINITION & HYPERPARAMETER TUNING
# ============================================================================
print(f"\n[3] Setting up XGBoost + TimeSeriesSplit GridSearchCV ...")

# Base estimator - GPU will be used if available, else CPU
base_xgb = xgb.XGBRegressor(
    objective      = "reg:squarederror",
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
    verbosity      = 0,
    early_stopping_rounds = None,   # disabled; CV handles overfitting guard
)

# Hyperparameter grid
param_grid = {
    "n_estimators"  : [200, 400, 600],
    "max_depth"     : [3, 5, 7],
    "learning_rate" : [0.05, 0.10, 0.20],
    "subsample"     : [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}

total_fits = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["learning_rate"])
    * len(param_grid["subsample"])
    * len(param_grid["colsample_bytree"])
    * len(param_grid["min_child_weight"])
    * N_CV_SPLITS
)
print(f"    Param combinations : "
      f"{total_fits // N_CV_SPLITS}  x  {N_CV_SPLITS} CV folds  "
      f"=  {total_fits} total fits")
print(f"    Scoring metric     : neg_root_mean_squared_error")
print(f"    This may take several minutes ... please wait.\n")

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

print(f"\n    GridSearchCV finished in {elapsed:.1f} seconds.")
print(f"\n    BEST PARAMETERS:")
for k, v in grid_search.best_params_.items():
    print(f"      {k:<22}: {v}")
print(f"\n    Best CV RMSE (train set): "
      f"{-grid_search.best_score_:.6f} wt%")

# ============================================================================
# 4. EVALUATION ON THE HELD-OUT TEST SET
# ============================================================================
best_model = grid_search.best_estimator_

print(f"\n[4] Evaluating best model on test set ...")
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = np.mean(np.abs(y_test - y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

# Residuals
residuals = y_test - y_pred

print(f"\n    {'-' * 48}")
print(f"    TEST SET PERFORMANCE METRICS")
print(f"    {'-' * 48}")
print(f"    RMSE (Root Mean Squared Error)  : {rmse:.6f} wt%")
print(f"    R2   (Coefficient of Determination): {r2:.6f}")
print(f"    MAE  (Mean Absolute Error)      : {mae:.6f} wt%")
print(f"    MAPE (Mean Abs Pct Error)       : {mape:.4f} %")
print(f"    {'-' * 48}")
print(f"    Target variable stats (test set):")
print(f"      Actual  - mean: {y_test.mean():.4f}  std: {y_test.std():.4f}  "
      f"min: {y_test.min():.4f}  max: {y_test.max():.4f}")
print(f"      Predicted - mean: {y_pred.mean():.4f}  std: {y_pred.std():.4f}  "
      f"min: {y_pred.min():.4f}  max: {y_pred.max():.4f}")
print(f"      Residuals - mean: {residuals.mean():.6f}  "
      f"std: {residuals.std():.6f}")
print(f"    {'-' * 48}")

# Spec threshold check
SPEC_THRESHOLD = 0.5   # wt% (from readme)
above_spec_actual    = (y_test > SPEC_THRESHOLD).sum()
above_spec_predicted = (pd.Series(y_pred) > SPEC_THRESHOLD).sum()
print(f"\n    Spec threshold check (C4 > {SPEC_THRESHOLD} wt%):")
print(f"      Actual violations in test set   : "
      f"{above_spec_actual} / {len(y_test)} "
      f"({100*above_spec_actual/len(y_test):.1f}%)")
print(f"      Predicted violations (by model) : "
      f"{above_spec_predicted} / {len(y_pred)} "
      f"({100*above_spec_predicted/len(y_pred):.1f}%)")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
print(f"\n[5] Feature Importances (gain-based, sorted highest to lowest):")
print(f"    {'Rank':<6} {'Feature':<26} {'Importance':>12}  {'Bar'}")
print(f"    {'-'*65}")

importances = best_model.feature_importances_
feat_imp_df = (
    pd.DataFrame({
        "Feature"   : feature_cols,
        "Importance": importances,
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

max_imp = feat_imp_df["Importance"].max()
for i, row in feat_imp_df.iterrows():
    bar_len = int(30 * row["Importance"] / max_imp)
    bar     = "#" * bar_len
    print(f"    {i+1:<6} {row['Feature']:<26} {row['Importance']:>12.6f}  {bar}")

# ============================================================================
# 6. EXPORT MODEL
# ============================================================================
print(f"\n[6] Saving model to '{MODEL_FILE}' ...")
best_model.save_model(MODEL_FILE)

# Sanity-check: reload and re-predict one row
reload_model = xgb.XGBRegressor()
reload_model.load_model(MODEL_FILE)
sanity_pred = reload_model.predict(X_test.iloc[[0]])[0]
print(f"    Model reload sanity check:")
print(f"      Actual  : {y_test.iloc[0]:.6f} wt%")
print(f"      Predicted (live)   : {y_pred[0]:.6f} wt%")
print(f"      Predicted (reload) : {sanity_pred:.6f} wt%")
if abs(y_pred[0] - sanity_pred) < 1e-4:
    print(f"    [OK] Model serialization verified -- reload matches live prediction.")
else:
    print(f"    [!!] Mismatch after reload -- investigate before deployment!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{SEPARATOR}")
print(f"  PHASE B FINAL SUMMARY")
print(f"{SEPARATOR}")
print(f"  Model     : XGBoost Regressor (gradient boosted trees)")
print(f"  Best Params:")
for k, v in grid_search.best_params_.items():
    print(f"    {k:<22}: {v}")
print(f"  -------------------------------------------------------")
print(f"  RMSE      : {rmse:.6f} wt%")
print(f"  R2 Score  : {r2:.6f}")
print(f"  MAE       : {mae:.6f} wt%")
print(f"  MAPE      : {mape:.4f} %")
print(f"  -------------------------------------------------------")
print(f"  Feature Importances (ranked):")
for i, row in feat_imp_df.iterrows():
    print(f"    {i+1}. {row['Feature']:<26} -> {row['Importance']:.6f}")
print(f"  -------------------------------------------------------")
print(f"  Model saved : {MODEL_FILE}")
print(f"{SEPARATOR}")
print(f"  PHASE B COMPLETE -- Model verified. Ready for Phase C (Dashboard).")
print(f"{SEPARATOR}\n")
