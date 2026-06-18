"""
=============================================================================
  DEBUTANIZER SOFT SENSOR — STREAMLIT DASHBOARD
  Phase C: Live C4 Slippage Prediction & Operator Assistant
=============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Debutanizer Soft Sensor",
    page_icon   = "🏭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
  }

  /* KPI cards */
  .kpi-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .kpi-label  { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em;
                text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }
  .kpi-value  { font-size: 2.6rem; font-weight: 800; line-height: 1; }
  .kpi-unit   { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

  /* Prediction hero card */
  .pred-hero {
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 8px;
    backdrop-filter: blur(12px);
    transition: all 0.4s ease;
  }
  .pred-hero.ok  {
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(5,150,105,0.08));
    border: 2px solid rgba(16,185,129,0.5);
    box-shadow: 0 0 40px rgba(16,185,129,0.15);
  }
  .pred-hero.warn {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(185,28,28,0.08));
    border: 2px solid rgba(239,68,68,0.5);
    box-shadow: 0 0 40px rgba(239,68,68,0.20);
  }
  .pred-label  { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.14em;
                 text-transform: uppercase; color: #94a3b8; margin-bottom: 12px; }
  .pred-value-ok   { font-size: 4.5rem; font-weight: 800; color: #10b981;
                     text-shadow: 0 0 20px rgba(16,185,129,0.4); }
  .pred-value-warn { font-size: 4.5rem; font-weight: 800; color: #ef4444;
                     text-shadow: 0 0 20px rgba(239,68,68,0.4); }
  .pred-status { font-size: 1.1rem; font-weight: 600; margin-top: 10px; }
  .status-ok   { color: #34d399; }
  .status-warn { color: #f87171; }

  /* Alert banner */
  .alert-loss {
    background: linear-gradient(90deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
  }
  .alert-ok {
    background: linear-gradient(90deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
  }

  /* Section headers */
  .section-header {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 8px; margin-bottom: 20px; margin-top: 28px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
  }
  [data-testid="stSidebar"] label { color: #cbd5e1 !important; }

  /* Metric delta */
  .metric-row { display:flex; justify-content:space-between; padding: 6px 0;
                border-bottom: 1px solid rgba(255,255,255,0.05); }
  .metric-name  { color: #94a3b8; font-size: 0.85rem; }
  .metric-val   { color: #e2e8f0; font-size: 0.85rem; font-weight: 600; }

  /* Recommendation card */
  .reco-card {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 16px 18px;
    margin: 8px 0;
  }
  .reco-title { font-weight: 700; color: #a5b4fc; font-size: 0.9rem; }
  .reco-text  { color: #c7d2fe; font-size: 0.82rem; margin-top: 4px; }

  /* Hide streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================
MODEL_PATH   = "xgb_debutanizer_model_v2.json"
DATA_PATH    = "debutanizer_features_with_lags.csv"
TARGET_COL   = "Total_C4_Slippage_Wt"
SPEC_LIMIT   = 0.5        # wt%
LOSS_FACTOR  = 1000       # INR per wt% above spec per hour

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, index_col="Timestamp", parse_dates=True)
    df.sort_index(inplace=True)
    return df

model = load_model()
df    = load_data()

feature_cols  = [c for c in df.columns if c != TARGET_COL]
last_row      = df[feature_cols].iloc[-1]          # "current live state"
history_df    = df[[TARGET_COL]].copy()            # for trend chart

# ============================================================================
# SIDEBAR — OPERATOR SIMULATOR
# ============================================================================
with st.sidebar:
    st.markdown("## 🎛️ Operator Simulator")
    st.markdown(
        "<p style='color:#64748b;font-size:0.78rem;'>Adjust control variables "
        "to simulate different operating conditions. Predictions update live.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("**Primary Control Variables**")

    steam_ratio = st.slider(
        "Steam-to-Feed Ratio",
        min_value = float(df["Steam_to_Feed_Ratio"].min()),
        max_value = float(df["Steam_to_Feed_Ratio"].max()),
        value     = float(last_row["Steam_to_Feed_Ratio"]),
        step      = 0.001,
        format    = "%.3f",
        help      = "Reboiling steam flow / Feed Flow to DB"
    )

    reflux_flow = st.slider(
        "Reflux Flow  (units)",
        min_value = float(df["Reflux flow"].min()),
        max_value = float(df["Reflux flow"].max()),
        value     = float(last_row["Reflux flow"]),
        step      = 0.1,
        format    = "%.1f",
        help      = "Column reflux flow rate"
    )

    col_bottom_temp = st.slider(
        "Column Bottom Temp  (°C)",
        min_value = float(df["Column bottom temp"].min()),
        max_value = float(df["Column bottom temp"].max()),
        value     = float(last_row["Column bottom temp"]),
        step      = 0.1,
        format    = "%.1f",
        help      = "Column bottom section temperature"
    )

    st.markdown("---")
    st.markdown("**Fixed at Last Live State**")
    fixed_vars = {
        k: last_row[k] for k in feature_cols
        if k not in ["Steam_to_Feed_Ratio", "Reflux flow", "Column bottom temp"]
    }
    for k, v in fixed_vars.items():
        st.markdown(
            f"<div class='metric-row'>"
            f"<span class='metric-name'>{k}</span>"
            f"<span class='metric-val'>{v:.3f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<p style='color:#475569;font-size:0.72rem;'>"
        "Model: XGBoost v2 | R²=0.9209 | RMSE=0.1036 wt%</p>",
        unsafe_allow_html=True
    )

# ============================================================================
# LIVE PREDICTION
# ============================================================================
input_vector = last_row.copy()
input_vector["Steam_to_Feed_Ratio"] = steam_ratio
input_vector["Reflux flow"]         = reflux_flow
input_vector["Column bottom temp"]  = col_bottom_temp

# Also update lag1 of reflux and steam to match (simulate steady-state)
if "Reflux flow_lag1" in input_vector.index:
    input_vector["Reflux flow_lag1"] = reflux_flow
if "Steam_to_Feed_Ratio_lag1" in input_vector.index:
    input_vector["Steam_to_Feed_Ratio_lag1"] = steam_ratio
if "Column bottom temp_lag1" in input_vector.index:
    input_vector["Column bottom temp_lag1"] = col_bottom_temp

X_input     = pd.DataFrame([input_vector[feature_cols]])
prediction  = float(model.predict(X_input)[0])
prediction  = max(0.0, prediction)        # clamp to physical range

is_ok       = prediction <= SPEC_LIMIT
loss_per_hr = max(0.0, (prediction - SPEC_LIMIT) * LOSS_FACTOR) if not is_ok else 0.0

# ============================================================================
# MAIN LAYOUT — HEADER
# ============================================================================
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(
        "<h1 style='font-size:1.9rem;font-weight:800;color:#f1f5f9;"
        "margin-bottom:2px;'>🏭 Debutanizer C4 Soft Sensor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;margin-top:0;'>"
        "Real-time C4 slippage prediction • AI-based operator assistant • "
        f"Last data point: {df.index[-1].strftime('%d %b %Y %H:%M')}</p>",
        unsafe_allow_html=True
    )
with col_badge:
    st.markdown(
        "<div style='text-align:right;padding-top:12px;'>"
        "<span style='background:rgba(16,185,129,0.15);border:1px solid "
        "rgba(16,185,129,0.4);border-radius:20px;padding:5px 14px;"
        "font-size:0.75rem;font-weight:700;color:#34d399;'>"
        "● LIVE SIMULATION</span></div>",
        unsafe_allow_html=True
    )

st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:8px 0 20px 0;'>",
            unsafe_allow_html=True)

# ── Row 1: Hero prediction + model KPIs ──────────────────────────────────────
col_pred, col_kpi1, col_kpi2, col_kpi3 = st.columns([2, 1, 1, 1])

with col_pred:
    hero_class = "pred-hero ok" if is_ok else "pred-hero warn"
    val_class  = "pred-value-ok" if is_ok else "pred-value-warn"
    status_txt = "✅ WITHIN SPEC" if is_ok else "⚠️ ABOVE SPEC — ACTION REQUIRED"
    status_cls = "status-ok" if is_ok else "status-warn"

    st.markdown(f"""
    <div class="{hero_class}">
      <div class="pred-label">Predicted C4 Slippage</div>
      <div class="{val_class}">{prediction:.3f}</div>
      <div class="kpi-unit">wt% in DB bottom</div>
      <div class="pred-status {status_cls}">{status_txt}</div>
      <div style="margin-top:10px;font-size:0.78rem;color:#64748b;">
        Spec limit: {SPEC_LIMIT} M% &nbsp;|&nbsp; Margin: {prediction - SPEC_LIMIT:+.3f} wt%
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Model R²</div>
      <div class="kpi-value" style="color:#818cf8;">0.9209</div>
      <div class="kpi-unit">on test set</div>
    </div>""", unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Model RMSE</div>
      <div class="kpi-value" style="color:#38bdf8;">0.1036</div>
      <div class="kpi-unit">wt%</div>
    </div>""", unsafe_allow_html=True)

with col_kpi3:
    actual_now = float(df[TARGET_COL].iloc[-1])
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Last Actual C4</div>
      <div class="kpi-value" style="color:#fb923c;">{actual_now:.3f}</div>
      <div class="kpi-unit">wt% (measured)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

# ── Row 2: Financial impact + Recommendations + Trend ────────────────────────
col_fin, col_reco = st.columns([1, 1])

with col_fin:
    st.markdown("<div class='section-header'>💰 Financial Impact</div>",
                unsafe_allow_html=True)
    if is_ok:
        st.markdown("""
        <div class="alert-ok">
          <span style="font-weight:700;color:#34d399;">✅ On-Spec — No Losses</span><br>
          <span style="color:#6ee7b7;font-size:0.85rem;">
            C4 slippage is within the 0.5 M% specification limit.
            Column operating efficiently.
          </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-loss">
          <span style="font-weight:700;color:#f87171;">⚠️ Off-Spec — Product Loss Occurring</span><br>
          <span style="color:#fca5a5;font-size:0.85rem;">
            Excess C4 in product stream: <strong>+{prediction - SPEC_LIMIT:.3f} wt%</strong>
            above specification.
          </span>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Excess C4", f"{max(0, prediction - SPEC_LIMIT):.3f} wt%",
                  delta=f"{prediction - SPEC_LIMIT:+.3f} vs spec",
                  delta_color="inverse")
    with c2:
        st.metric("Financial Loss", f"₹ {loss_per_hr:,.0f} /hr",
                  delta="Above spec" if not is_ok else "On spec",
                  delta_color="inverse" if not is_ok else "off")

    if not is_ok:
        daily_loss = loss_per_hr * 24
        monthly_loss = daily_loss * 30
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.08);border-radius:10px;padding:14px 16px;margin-top:8px;'>
          <div style='color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:8px;'>Projected Losses</div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;
                      border-bottom:1px solid rgba(255,255,255,0.05);'>
            <span style='color:#94a3b8;font-size:0.85rem;'>Per Hour</span>
            <span style='color:#fca5a5;font-weight:700;'>₹ {loss_per_hr:,.0f}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;
                      border-bottom:1px solid rgba(255,255,255,0.05);'>
            <span style='color:#94a3b8;font-size:0.85rem;'>Per Day</span>
            <span style='color:#fca5a5;font-weight:700;'>₹ {daily_loss:,.0f}</span>
          </div>
          <div style='display:flex;justify-content:space-between;padding:4px 0;'>
            <span style='color:#94a3b8;font-size:0.85rem;'>Per Month</span>
            <span style='color:#ef4444;font-weight:700;'>₹ {monthly_loss:,.0f}</span>
          </div>
        </div>""", unsafe_allow_html=True)

with col_reco:
    st.markdown("<div class='section-header'>🧠 Operator Recommendations</div>",
                unsafe_allow_html=True)
    if is_ok:
        st.markdown("""
        <div class="reco-card">
          <div class="reco-title">✅ Column Operating On-Spec</div>
          <div class="reco-text">
            Current control settings are maintaining C4 slippage within specification.
            Continue monitoring — no immediate action required.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        excess = prediction - SPEC_LIMIT
        if steam_ratio < df["Steam_to_Feed_Ratio"].quantile(0.6):
            st.markdown("""
            <div class="reco-card">
              <div class="reco-title">🔥 Increase Reboiling Steam Flow</div>
              <div class="reco-text">
                Steam-to-Feed Ratio is below the 60th percentile of normal operations.
                Increasing reboiler steam will improve C4/C5 separation at the bottom.
                <strong>Target: increase Steam-to-Feed Ratio by 0.01–0.02</strong>
              </div>
            </div>""", unsafe_allow_html=True)
        if reflux_flow < df["Reflux flow"].quantile(0.5):
            st.markdown("""
            <div class="reco-card">
              <div class="reco-title">🔄 Increase Reflux Flow</div>
              <div class="reco-text">
                Reflux is below median operating value. A higher reflux rate improves
                column fractionation and reduces C4 slip into the bottom product.
                <strong>Target: raise reflux by 2–5 units</strong>
              </div>
            </div>""", unsafe_allow_html=True)
        if col_bottom_temp < df["Column bottom temp"].quantile(0.45):
            st.markdown("""
            <div class="reco-card">
              <div class="reco-title">🌡️ Check Bottom Temperature</div>
              <div class="reco-text">
                Column bottom temperature is lower than typical on-spec operating range.
                Verify reboiler outlet temperature and steam supply pressure.
                <strong>Target: maintain bottom temp at historical median</strong>
              </div>
            </div>""", unsafe_allow_html=True)
        if excess > 0.5:
            st.markdown(f"""
            <div class="reco-card" style="border-color:rgba(239,68,68,0.5);">
              <div class="reco-title" style="color:#f87171;">
                🚨 High Excess ({excess:.3f} wt%) — Escalate to Shift Supervisor
              </div>
              <div class="reco-text">
                C4 slippage is significantly above spec. Verify analyzer readings,
                check for feed composition upsets, and consider reducing feed rate
                while adjusting steam and reflux simultaneously.
              </div>
            </div>""", unsafe_allow_html=True)

# ── Row 3: Historical Trend Chart ─────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Historical C4 Slippage Trend (Last 30 Days)</div>",
            unsafe_allow_html=True)

cutoff  = df.index.max() - pd.Timedelta(days=30)
last_30d = history_df.loc[cutoff:]
fig, ax  = plt.subplots(figsize=(14, 3.2))
fig.patch.set_facecolor("#0f0c29")
ax.set_facecolor("#0f0c29")

ax.plot(last_30d.index, last_30d[TARGET_COL],
        color="#818cf8", linewidth=1.2, alpha=0.9, label="Actual C4 Slippage")
ax.axhline(SPEC_LIMIT, color="#ef4444", linewidth=1.5, linestyle="--",
           alpha=0.8, label=f"Spec Limit ({SPEC_LIMIT} wt%)")
ax.fill_between(last_30d.index, last_30d[TARGET_COL], SPEC_LIMIT,
                where=(last_30d[TARGET_COL] > SPEC_LIMIT),
                color="#ef4444", alpha=0.15, label="Off-spec zone")
ax.axhline(prediction, color="#fbbf24", linewidth=1.5, linestyle=":",
           alpha=0.9, label=f"Current Prediction ({prediction:.3f} wt%)")

ax.set_xlabel("", fontsize=9, color="#64748b")
ax.set_ylabel("C4 Slippage (wt%)", fontsize=9, color="#94a3b8")
ax.tick_params(colors="#64748b", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#2d3748")
    spine.set_linewidth(0.5)
ax.grid(axis="y", color="#1e293b", linewidth=0.5)
legend = ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155",
                   labelcolor="#cbd5e1", loc="upper left")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Row 4: Feature Values Table ───────────────────────────────────────────────
with st.expander("🔍 Current Input Feature Vector (click to expand)"):
    display_df = pd.DataFrame({
        "Feature" : feature_cols,
        "Value"   : [input_vector[f] for f in feature_cols],
    }).set_index("Feature")
    st.dataframe(
        display_df.style.format({"Value": "{:.4f}"}),
        use_container_width=True,
        height=350,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:rgba(255,255,255,0.05);margin-top:30px;'>"
    "<p style='text-align:center;color:#334155;font-size:0.75rem;'>"
    "Debutanizer Soft Sensor v2.0 &nbsp;|&nbsp; XGBoost Model &nbsp;|&nbsp; "
    "R² = 0.9209 &nbsp;|&nbsp; RMSE = 0.1036 wt%"
    "</p>",
    unsafe_allow_html=True
)
