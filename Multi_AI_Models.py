import os
import sys
import subprocess

# --- STEP 1: AUTOMATIC DEPENDENCY INSTALLER ---
venv_path = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)


def safe_install(pip_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


# Ensure all model dependencies are available
for pkg, pip_pkg in [
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
    ("sklearn", "scikit-learn"),
]:
    try:
        __import__(pkg)
    except ImportError:
        safe_install(pip_pkg)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Institutional AI Forecaster", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    /* Dark premium theme */
    .main { background-color: #0d0f14; }
    .stApp { background-color: #0d0f14; color: #e0e6f0; }
    h1, h2, h3, h4 { color: #c9d4e8 !important; }

    /* Model card badges */
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-xgb   { background: #1a3a5c; color: #4db8ff; border: 1px solid #4db8ff; }
    .badge-lgb   { background: #1a3a2a; color: #4dff91; border: 1px solid #4dff91; }
    .badge-cat   { background: #3a2a1a; color: #ffaa4d; border: 1px solid #ffaa4d; }
    .badge-rf    { background: #2a1a3a; color: #cc88ff; border: 1px solid #cc88ff; }
    .badge-ens   { background: #3a1a2a; color: #ff88cc; border: 1px solid #ff88cc; }

    /* Accuracy metric box */
    .metric-box {
        background: #141820;
        border: 1px solid #2a3040;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 28px; font-weight: 800; color: #4db8ff; }
    .metric-label { font-size: 11px; color: #6a7a90; text-transform: uppercase; letter-spacing: 0.08em; }

    /* Compare table */
    .compare-row {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        background: #141820;
        border: 1px solid #2a3040;
    }
    .compare-row.best { border-color: #4dff91; background: #0d1f15; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Institutional AI Forecaster — Multi-Model Engine")
st.caption("XGBoost · LightGBM · CatBoost · Random Forest · Ensemble | by Vijay Madhu")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# ── Asset Selection ──────────────────────────
asset_display = st.sidebar.selectbox(
    "📌 Asset Pair",
    [
        "EUR/USD", "GBP/USD", "AUD/USD", "USD/CAD", "USD/JPY",
        "Bitcoin (BTC/USD)", "Ethereum (ETH/USD)", "Solana (SOL/USD)", "Ripple (XRP/USD)"
    ],
    index=5
)

ticker_mapping = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X", "USD/JPY": "JPY=X",
    "Bitcoin (BTC/USD)": "BTC-USD", "Ethereum (ETH/USD)": "ETH-USD",
    "Solana (SOL/USD)": "SOL-USD", "Ripple (XRP/USD)": "XRP-USD"
}
scale_mapping = {
    "EUR/USD": 0.0001, "GBP/USD": 0.0001, "AUD/USD": 0.0001, "USD/CAD": 0.0001, "USD/JPY": 0.01,
    "Bitcoin (BTC/USD)": 1.0, "Ethereum (ETH/USD)": 1.0, "Solana (SOL/USD)": 0.1, "Ripple (XRP/USD)": 0.001
}
symbol = ticker_mapping[asset_display]
point_scale = scale_mapping[asset_display]

# ── Timeframe ────────────────────────────────
timeframe = st.sidebar.selectbox("⏱ Timeframe", ["5m", "15m", "1h", "4h", "1d", "1wk"], index=1)
chart_style = st.sidebar.radio("📊 Chart Style", ["Candlestick", "Line Chart"], index=0)

# ── AI MODEL SELECTOR ────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Model Selection")

MODEL_OPTIONS = {
    "XGBoost": {
        "desc": "Gradient boosted trees. Fast, accurate, handles feature interactions well. Best for tabular financial data.",
        "badge": "badge-xgb", "color": "#4db8ff", "emoji": "⚡"
    },
    "LightGBM": {
        "desc": "Microsoft's leaf-wise gradient boosting. Fastest training, excellent on large datasets & high-frequency data.",
        "badge": "badge-lgb", "color": "#4dff91", "emoji": "🌿"
    },
    "CatBoost": {
        "desc": "Yandex's ordered boosting. Superior with categorical features, robust to overfitting on smaller datasets.",
        "badge": "badge-cat", "color": "#ffaa4d", "emoji": "🐱"
    },
    "Random Forest": {
        "desc": "Bagged decision trees. Very stable, low variance. Great baseline and reliable in choppy market conditions.",
        "badge": "badge-rf", "color": "#cc88ff", "emoji": "🌲"
    },
    "Ensemble (All Models)": {
        "desc": "Soft-voting ensemble of all 4 models. Highest stability — averages out individual model biases.",
        "badge": "badge-ens", "color": "#ff88cc", "emoji": "🔀"
    },
}

# Display model info cards in sidebar
selected_model = st.sidebar.radio(
    "Choose AI Engine:",
    list(MODEL_OPTIONS.keys()),
    index=0
)

# Show description for selected model
m = MODEL_OPTIONS[selected_model]
st.sidebar.markdown(f"""
<div style="background:#141820; border:1px solid #2a3040; border-radius:10px; padding:12px; margin-top:8px;">
    <span style="font-size:20px">{m['emoji']}</span>
    <span class="model-badge {m['badge']}" style="margin-left:8px">{selected_model}</span>
    <p style="color:#8899aa; font-size:12px; margin-top:8px; margin-bottom:0">{m['desc']}</p>
</div>
""", unsafe_allow_html=True)

# Optional: compare all models
compare_all = st.sidebar.checkbox("📊 Compare All Models Side-by-Side", value=False)

# ─────────────────────────────────────────────
# PERIOD MAPPING
# ─────────────────────────────────────────────
period_map = {"5m": "10d", "15m": "30d", "1h": "90d", "4h": "730d", "1d": "2y", "1wk": "5y"}
period = period_map[timeframe]


# ─────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────
def build_model(name):
    """Returns a fresh, configured model instance for the given name."""
    if name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0
        )
    elif name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
    elif name == "CatBoost":
        return CatBoostClassifier(
            iterations=150, learning_rate=0.05, depth=4,
            random_seed=42, verbose=0
        )
    elif name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=10,
            random_state=42, n_jobs=-1
        )
    elif name == "Ensemble (All Models)":
        return VotingClassifier(
            estimators=[
                ("xgb", build_model("XGBoost")),
                ("lgb", build_model("LightGBM")),
                ("cat", build_model("CatBoost")),
                ("rf", build_model("Random Forest")),
            ],
            voting="soft"
        )


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (shared across all models)
# ─────────────────────────────────────────────
def build_features(data, point_scale):
    close = data['Close'].squeeze().astype(float)
    open_p = data['Open'].squeeze().astype(float)
    high = data['High'].squeeze().astype(float)
    low = data['Low'].squeeze().astype(float)
    volume = data['Volume'].squeeze().astype(float)

    # ATR
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()
    data['Normalized_Movement'] = data['ATR'] / point_scale

    # SMC Order Blocks
    data['Bullish_OB'] = np.nan
    data['Bearish_OB'] = np.nan
    for i in range(2, len(data)):
        if close.iloc[i] > high.iloc[i - 1] and close.iloc[i - 1] < open_p.iloc[i - 1]:
            data.iloc[i, data.columns.get_loc('Bullish_OB')] = low.iloc[i - 1]
        if close.iloc[i] < low.iloc[i - 1] and close.iloc[i - 1] > open_p.iloc[i - 1]:
            data.iloc[i, data.columns.get_loc('Bearish_OB')] = high.iloc[i - 1]
    data['Bullish_OB'] = data['Bullish_OB'].ffill()
    data['Bearish_OB'] = data['Bearish_OB'].ffill()

    # Pivot Points
    data['PP'] = (high + low + close) / 3
    data['R1'] = (2 * data['PP']) - low
    data['S1'] = (2 * data['PP']) - high

    # Volume Profile / POC
    price_min, price_max = close.min(), close.max()
    bins = np.linspace(price_min, price_max, num=20)
    bin_indices = np.digitize(close, bins) - 1
    volume_by_bin = np.zeros(len(bins))
    for idx, vol in zip(bin_indices, volume):
        if 0 <= idx < len(bins):
            volume_by_bin[idx] += vol
    poc_price = bins[np.argmax(volume_by_bin)]

    # Feature DataFrame
    df_feats = pd.DataFrame(index=data.index)
    df_feats['Normalized_Movement'] = data['Normalized_Movement']
    volume_sma = volume.rolling(window=20).mean()
    df_feats['Relative_Volume'] = (volume / volume_sma).fillna(1.0)
    atr_norm = data['ATR'].replace(0, 1e-5)
    df_feats['Dist_to_BullOB'] = ((close - data['Bullish_OB']) / atr_norm).fillna(0)
    df_feats['Dist_to_BearOB'] = ((close - data['Bearish_OB']) / atr_norm).fillna(0)
    df_feats['Dist_to_POC'] = (close - poc_price) / atr_norm
    df_feats['Dist_to_PP'] = (close - data['PP']) / atr_norm
    df_feats['Target'] = (close.shift(-3) > close).astype(int)

    return data, df_feats, poc_price, close, open_p, high, low, volume


FEATURE_COLS = ['Normalized_Movement', 'Relative_Volume', 'Dist_to_BullOB',
                'Dist_to_BearOB', 'Dist_to_POC', 'Dist_to_PP']


def train_and_predict(df_feats, model_name):
    """Train model, return probabilities + test accuracy."""
    live_features = df_feats.iloc[[-1]].copy()
    df_clean = df_feats.dropna()

    X = df_clean[FEATURE_COLS]
    y = df_clean['Target']
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = build_model(model_name)
    model.fit(X_train, y_train)

    # Test accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Live prediction
    up_prob = model.predict_proba(live_features[FEATURE_COLS])[0][1]
    return up_prob, 1 - up_prob, acc, model


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────
if st.sidebar.button("🚀 Analyze & Train Engine"):
    with st.spinner(f"Fetching data and training {selected_model}..."):

        # ── Download Data ────────────────────
        data = yf.download(symbol, period=period, interval=timeframe)
        if data.empty:
            st.error("Market data unavailable. Try a different timeframe/asset.")
            st.stop()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # ── Feature Engineering ──────────────
        data, df_feats, poc_price, close, open_p, high, low, volume = build_features(data, point_scale)

        is_fx = any(x in symbol for x in ["=X", "CAD=", "JPY="])
        fmt = lambda p: f"${p:.5f}" if is_fx else f"${p:,.2f}"

        # ─────────────────────────────────────
        # MODE A: Single Selected Model
        # ─────────────────────────────────────
        if not compare_all:
            up_prob, dn_prob, acc, model = train_and_predict(df_feats, selected_model)
            m_info = MODEL_OPTIONS[selected_model]

            # ── Model Header ─────────────────
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; padding:16px; 
                        background:#141820; border-radius:12px; border:1px solid #2a3040; margin-bottom:16px">
                <span style="font-size:32px">{m_info['emoji']}</span>
                <div>
                    <span class="model-badge {m_info['badge']}">{selected_model}</span>
                    <p style="color:#8899aa; margin:4px 0 0 0; font-size:13px">{m_info['desc']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability Gauges ────────────
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:#4dff91">{up_prob * 100:.1f}%</div>
                    <div class="metric-label">🟢 BUY Probability</div></div>""", unsafe_allow_html=True)
            with g2:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:#ff4d4d">{dn_prob * 100:.1f}%</div>
                    <div class="metric-label">🔴 SELL Probability</div></div>""", unsafe_allow_html=True)
            with g3:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value">{fmt(close.iloc[-1])}</div>
                    <div class="metric-label">💹 Live Price</div></div>""", unsafe_allow_html=True)
            with g4:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:#ffaa4d">{acc * 100:.1f}%</div>
                    <div class="metric-label">🎯 Test Accuracy</div></div>""", unsafe_allow_html=True)

            # ── Recommendations ───────────────
            st.subheader("🤖 AI Trading Recommendations")
            rec_col1, rec_col2 = st.columns(2)

            bull_ob_fmt = fmt(data['Bullish_OB'].iloc[-1])
            bear_ob_fmt = fmt(data['Bearish_OB'].iloc[-1])
            live_fmt = fmt(close.iloc[-1])

            with rec_col1:
                st.markdown("### 🟢 BUY Setup")
                if up_prob > 0.65:
                    st.success(f"**EXECUTE BUY ORDER NOW!**\n\n"
                               f"- **Entry:** {live_fmt}\n"
                               f"- **{selected_model} Confidence:** {up_prob * 100:.1f}%\n"
                               f"- **Signal:** Institutional accumulation detected above support.")
                elif up_prob > 0.45:
                    st.warning(f"**WAIT — PULLBACK SETUP**\n\n"
                               f"- **Current Price:** {live_fmt}\n"
                               f"- **Target Entry:** {bull_ob_fmt} (Bullish OB)\n"
                               f"- **Strategy:** Set limit at Bullish OB floor. Avoid chasing.")
                else:
                    st.error(f"**DO NOT BUY**\n\n"
                             f"- **Upside Confidence:** {up_prob * 100:.1f}%\n"
                             f"- **Signal:** High bearish momentum. Falling knife risk.")

            with rec_col2:
                st.markdown("### 🔴 SELL Setup")
                if dn_prob > 0.65:
                    st.success(f"**EXECUTE SELL ORDER NOW!**\n\n"
                               f"- **Entry:** {live_fmt}\n"
                               f"- **{selected_model} Confidence:** {dn_prob * 100:.1f}%\n"
                               f"- **Signal:** Distribution complete, resistance rejection confirmed.")
                elif dn_prob > 0.45:
                    st.warning(f"**WAIT — RALLY SETUP**\n\n"
                               f"- **Current Price:** {live_fmt}\n"
                               f"- **Target Entry:** {bear_ob_fmt} (Bearish OB)\n"
                               f"- **Strategy:** Wait for retracement to Bearish OB ceiling.")
                else:
                    st.error(f"**DO NOT SELL**\n\n"
                             f"- **Downside Confidence:** {dn_prob * 100:.1f}%\n"
                             f"- **Signal:** Strong support, high demand. Unsafe to short.")

            # ── Feature Importance (where applicable) ──
            if selected_model in ["XGBoost", "LightGBM", "Random Forest"]:
                st.subheader("📐 Feature Importance")
                if selected_model == "XGBoost":
                    importances = model.feature_importances_
                elif selected_model == "LightGBM":
                    importances = model.feature_importances_ / model.feature_importances_.sum()
                else:
                    importances = model.feature_importances_

                fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
                fi_df = fi_df.sort_values("Importance", ascending=True)
                fig_fi = go.Figure(go.Bar(
                    x=fi_df["Importance"], y=fi_df["Feature"], orientation='h',
                    marker_color='#4db8ff', marker_line_color='#1a3a5c', marker_line_width=1
                ))
                fig_fi.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Importance Score", yaxis_title=""
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        # ─────────────────────────────────────
        # MODE B: Compare All Models
        # ─────────────────────────────────────
        else:
            st.subheader("📊 Model Comparison — All 5 Engines")
            model_names = ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Ensemble (All Models)"]
            results = {}

            prog = st.progress(0)
            for i, mname in enumerate(model_names):
                with st.spinner(f"Training {mname}..."):
                    up, dn, acc, _ = train_and_predict(df_feats, mname)
                    results[mname] = {"up": up, "dn": dn, "acc": acc}
                prog.progress((i + 1) / len(model_names))
            prog.empty()

            # Find best model by accuracy
            best_model = max(results, key=lambda k: results[k]["acc"])

            # Render comparison table
            st.markdown(f"**Best Model by Test Accuracy:** `{best_model}` 🏆")
            st.markdown("---")

            # Header
            hcols = st.columns([2.5, 1.5, 1.5, 1.5])
            hcols[0].markdown("**Model**")
            hcols[1].markdown("**🟢 BUY Prob**")
            hcols[2].markdown("**🔴 SELL Prob**")
            hcols[3].markdown("**🎯 Test Accuracy**")

            for mname, res in results.items():
                is_best = mname == best_model
                m_info = MODEL_OPTIONS[mname]
                border = "#4dff91" if is_best else "#2a3040"
                bg = "#0d1f15" if is_best else "#141820"

                cols = st.columns([2.5, 1.5, 1.5, 1.5])
                with cols[0]:
                    st.markdown(f"""
                    <div style="background:{bg}; border:1px solid {border}; border-radius:8px; padding:10px">
                        {m_info['emoji']} <span class="model-badge {m_info['badge']}">{mname}</span>
                        {"&nbsp;🏆" if is_best else ""}
                    </div>""", unsafe_allow_html=True)
                with cols[1]:
                    color = "#4dff91" if res["up"] > 0.65 else "#ffaa4d" if res["up"] > 0.45 else "#ff4d4d"
                    st.markdown(f"""<div style="background:{bg}; border:1px solid {border}; 
                        border-radius:8px; padding:10px; text-align:center;
                        color:{color}; font-weight:700; font-size:18px">
                        {res['up'] * 100:.1f}%</div>""", unsafe_allow_html=True)
                with cols[2]:
                    color = "#ff4d4d" if res["dn"] > 0.65 else "#ffaa4d" if res["dn"] > 0.45 else "#4dff91"
                    st.markdown(f"""<div style="background:{bg}; border:1px solid {border}; 
                        border-radius:8px; padding:10px; text-align:center;
                        color:{color}; font-weight:700; font-size:18px">
                        {res['dn'] * 100:.1f}%</div>""", unsafe_allow_html=True)
                with cols[3]:
                    acc_color = "#4dff91" if res["acc"] > 0.55 else "#ffaa4d"
                    st.markdown(f"""<div style="background:{bg}; border:1px solid {border}; 
                        border-radius:8px; padding:10px; text-align:center;
                        color:{acc_color}; font-weight:700; font-size:18px">
                        {res['acc'] * 100:.1f}%</div>""", unsafe_allow_html=True)

            # Signal consensus
            st.markdown("---")
            st.subheader("🧠 Signal Consensus")
            buy_votes = sum(1 for r in results.values() if r["up"] > 0.65)
            sell_votes = sum(1 for r in results.values() if r["dn"] > 0.65)
            wait_votes = len(results) - buy_votes - sell_votes
            avg_up = np.mean([r["up"] for r in results.values()])

            cv1, cv2, cv3, cv4 = st.columns(4)
            with cv1:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:#4dff91">{buy_votes}/5</div>
                    <div class="metric-label">🟢 BUY Votes</div></div>""", unsafe_allow_html=True)
            with cv2:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:#ff4d4d">{sell_votes}/5</div>
                    <div class="metric-label">🔴 SELL Votes</div></div>""", unsafe_allow_html=True)
            with cv3:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value" style="color:#ffaa4d">{wait_votes}/5</div>
                    <div class="metric-label">⏸ WAIT Votes</div></div>""", unsafe_allow_html=True)
            with cv4:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-value">{avg_up * 100:.1f}%</div>
                    <div class="metric-label">📈 Avg BUY Prob</div></div>""", unsafe_allow_html=True)

            # Overall consensus signal
            if buy_votes >= 3:
                st.success(
                    f"**🟢 CONSENSUS: BUY** — {buy_votes} out of 5 models signal upward movement (avg {avg_up * 100:.1f}% confidence)")
            elif sell_votes >= 3:
                st.error(f"**🔴 CONSENSUS: SELL** — {sell_votes} out of 5 models signal downward movement")
            else:
                st.warning(
                    f"**⚡ MIXED SIGNALS** — No clear consensus. Models are divided. Consider waiting for confirmation.")

        # ─────────────────────────────────────
        # PRICE CHART (shared for both modes)
        # ─────────────────────────────────────
        st.subheader("📉 Price Chart with Institutional Levels")
        fig = go.Figure()
        live_fmt = fmt(close.iloc[-1])

        if chart_style == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index, open=open_p, high=high, low=low, close=close,
                name=f"{asset_display}", increasing_line_color='#4dff91', decreasing_line_color='#ff4d4d'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data.index, y=close, mode='lines',
                line=dict(color='#4db8ff', width=2), name="Spot Price"
            ))

        fig.add_trace(go.Scatter(
            x=data.index, y=[poc_price] * len(data), mode="lines",
            line=dict(color="#ffaa4d", width=2, dash="dashdot"),
            name=f"Volume POC ({fmt(poc_price)})"
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['PP'], mode="lines",
            line=dict(color="rgba(160, 100, 220, 0.6)", width=1, dash="dot"),
            name="Pivot Point"
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Bullish_OB'], mode="lines",
            line=dict(color="rgba(77, 255, 145, 0.5)", width=1.5),
            name="Bullish OB Floor"
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Bearish_OB'], mode="lines",
            line=dict(color="rgba(255, 77, 77, 0.5)", width=1.5),
            name="Bearish OB Ceiling"
        ))

        fig.update_layout(
            height=580, template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; background:#141820; border-radius:16px; 
                border:1px dashed #2a3040; margin-top:30px">
        <div style="font-size:48px">🛡️</div>
        <h3 style="color:#c9d4e8">Select your asset, timeframe, and AI model</h3>
        <p style="color:#6a7a90">Then click <strong>Analyze & Train Engine</strong> in the sidebar to begin.</p>
        <p style="color:#4a5a70; font-size:13px">Available engines: XGBoost · LightGBM · CatBoost · Random Forest · Ensemble</p>
    </div>
    """, unsafe_allow_html=True)