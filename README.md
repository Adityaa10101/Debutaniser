# AI-Based Soft Sensor: C4 Slippage Minimizer for Debutanizer

> **Objective:** "AI Based Model to Minimize C4 Slippage in DEBUTANIZER"

[![Model R²](https://img.shields.io/badge/Model%20R%C2%B2-0.9209-brightgreen)](.)
[![RMSE](https://img.shields.io/badge/RMSE-0.1036%20wt%25-blue)](.)
[![Stack](https://img.shields.io/badge/Stack-XGBoost%20%7C%20Streamlit%20%7C%20Pandas-orange)](.)

---

## Problem Statement

C4 slippage in the C5+ product stream of the Debutanizer column currently varies between **0.8% – 1.5% wt** against a specification limit of **0.5 M%**. Key challenges:

- Manual operation based purely on operator experience
- Analyzer cycle time of 12 minutes — wide variation between samplings
- Analyzer readings are unreliable
- Feed and operating variability not handled optimally

This project builds a **Gradient Boosting Regression soft sensor** to provide real-time, continuous prediction of C4 slippage, enabling proactive operator guidance and economic loss minimization.

---

## Solution Architecture

```
Plant DCS (Exaquantum)
        │
        ▼
  Historical Data (CSV)
        │
        ▼
┌───────────────────────────────────────┐
│  Phase A: Data Preprocessing          │
│  - Thermodynamic feature engineering  │
│  - Temporal lag features              │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  Phase B: XGBoost Soft Sensor         │
│  - TimeSeriesSplit GridSearchCV       │
│  - R² = 0.9209 | RMSE = 0.1036 wt%   │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  Phase C: Streamlit Dashboard         │
│  - Live C4 prediction                 │
│  - Operator recommendations           │
│  - Financial loss (INR/hr)            │
└───────────────────────────────────────┘
```

---

## Phase A: Data Preprocessing & Feature Engineering ✅

**Script:** `phase_a_preprocessing.py`
**Input:** `debutanizer_cleaned_v1.csv` (10,595 hourly records | Apr 2023 – Apr 2026)
**Output:** `debutanizer_model_ready.csv`

### Steps Completed

1. **Time Indexing** — `Unnamed: 0` converted to a proper `datetime64` index (`Timestamp`)

2. **Thermodynamic Feature Engineering** — Three domain-driven ratio features created:

   | New Feature | Formula | Physical Meaning |
   |---|---|---|
   | `Reflux_Ratio` | `Reflux flow / Feed Flow to DB` | Column separation efficiency |
   | `Temp_Diff_Bottom_Top` | `Column bottom temp − Column top Temp` | Temperature gradient across column |
   | `Steam_to_Feed_Ratio` | `Reboiling steam flow / Feed Flow to DB` | Reboiler energy intensity |

3. **Target Variable Cleanup** — Individual component columns (`C4H6 in DB bottom`, `C4H8 in DB bottom`) dropped as redundant; combined `Total_C4_Slippage_Wt` is the sole target.

4. **Temporal Lag Features (Phase B-Revised fix)** — To capture *column memory* and resolve temporal distribution shift, 13 additional features were engineered:

   | Feature Group | Details |
   |---|---|
   | Target lags | `Total_C4_Slippage_Wt` at `t-1`, `t-2`, `t-3` |
   | Driver lags (t-1) | `Steam_to_Feed_Ratio`, `Column Top pressure`, `Column bottom temp`, `Reflux flow`, `Control tay temp` |
   | Rolling 3h means | Same 5 drivers — smooths sensor noise, captures operating regime |

   **Final dataset:** `debutanizer_features_with_lags.csv` — 10,592 rows × 25 columns (24 features + target)

### Validation
- ✅ Zero null values across all 10,592 rows
- ✅ All engineered features within physically expected ranges

---

## Phase B: XGBoost Model Training & Validation ✅

**Script:** `phase_b_revised_lag_retrain.py`
**Input:** `debutanizer_features_with_lags.csv`
**Output:** `xgb_debutanizer_model_v2.json`

### Methodology

- **Split:** Chronological 80/20 — 8,473 train / 2,119 test rows **(no shuffling)** to respect temporal order
- **Cross-Validation:** `TimeSeriesSplit(n_splits=5)` inside `GridSearchCV` — each fold's test window is always *after* its training window
- **Scoring:** `neg_root_mean_squared_error`
- **Total fits:** 360 (72 combinations × 5 folds)

### Best Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.03 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` (L1) | 0.1 |
| `reg_lambda` (L2) | 1 |
| `min_child_weight` | 1 |

### Final Model Performance (Held-Out Test Set)

| Metric | Phase B (no lags) | Phase B-Revised ✅ |
|---|---|---|
| **RMSE** | 0.3817 wt% | **0.1036 wt%** |
| **R² Score** | -0.0723 | **0.9209** |
| **MAE** | 0.3227 wt% | **0.0555 wt%** |
| **MAPE** | 143.26% | **13.51%** |

> **R² = 0.9209 — the model explains 92% of variance in C4 slippage on unseen data. Production-grade accuracy for an industrial soft sensor.**

### Feature Importances (Top 5)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Total_C4_Slippage_Wt_lag1` | 57.1% |
| 2 | `Total_C4_Slippage_Wt_lag2` | 19.2% |
| 3 | `Total_C4_Slippage_Wt_lag3` | 3.7% |
| 4 | `Steam_to_Feed_Ratio_lag1` | 1.8% |
| 5 | `Column Top pressure_lag1` | 1.3% |

The dominance of lag features confirms strong **autocorrelation in C4 slippage** — the column's recent history is the best predictor of its near-future state. Process variables (steam ratio, pressure, temperatures) act as corrective signals on top of that trajectory.

### Spec Violation Detection
```
Actual violations in test set   : 852 / 2119  (40.2%)
Predicted violations by model   : 856 / 2119  (40.4%)
```

---

## Phase C: Streamlit Dashboard 🚧 In Progress

**Script:** `app.py`

Features:
- **Live C4 Prediction** — Real-time soft sensor output with Green/Red spec indicator
- **Operator Simulator** — Interactive sliders for key control variables
- **Financial Impact** — INR/hr loss calculation when above spec
- **Actual vs Predicted Trends** — Historical comparison chart

---

## Input Variables

| Variable | Description |
|---|---|
| `Column top Temp` | Top section temperature |
| `Column bottom temp` | Bottom section temperature |
| `Reboiler o/l Temp` | Reboiler outlet temperature |
| `Reboiling steam flow` | LP steam flow to reboiler |
| `Reflux flow` | Reflux flow rate |
| `Feed Flow to DB` | Column feed flow |
| `Control tay temp` | Control tray temperature |
| `Column Top pressure` | Column overhead pressure |

---

## File Structure

```
Debutaniser/
├── debutanizer_cleaned_v1.csv              # Raw plant data
├── debutanizer_model_ready.csv             # After Phase A (thermodynamic features)
├── debutanizer_features_with_lags.csv      # After Phase A-Revised (lag features)
├── phase_a_preprocessing.py               # Phase A script
├── phase_b_xgb_training.py                # Phase B initial training script
├── phase_b_revised_lag_retrain.py         # Phase B-Revised (lag + retrain)
├── xgb_debutanizer_model_v2.json          # Final trained XGBoost model
├── app.py                                 # Streamlit dashboard (Phase C)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## Requirements

```
streamlit
xgboost
pandas
numpy
matplotlib
seaborn
scikit-learn
reportlab
openpyxl
```

Install: `pip install -r requirements.txt`

Run Dashboard: `streamlit run app.py`

---

## Future Scope

- Real-time deployment via Seeq or similar historian integration
- Closed-loop optimization with APC (Advanced Process Control)
- Extension to other separation columns in the plant
- Automated report generation (PDF) for shift handover

---

## Process Background

The Debutanizer separates mixed C4s from C5s and heavier components. Feed from the DP bottom is fed on level control to the 17th tray. Reboiling duty is provided by LP (desuperheated) steam. Column vapors are condensed with cooling water and collected in the reflux drum. Mixed C4s (after meeting reflux requirements) are sent to:
- Butadiene Extraction Unit
- C4 Hydrogenation Unit
- OSBL Storage
