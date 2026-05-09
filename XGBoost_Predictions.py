import os
import sys
import subprocess

# --- STEP 1: AUTOMATIC DEPENDENCY INSTALLER ---
venv_path = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)


def install_and_import(package, pip_name=None):
    try:
        return __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic==1.10.11", pip_name or package])
        return __import__(package)


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Install XGBoost safely
xgb = install_and_import("xgboost")

# --- APP CONFIG ---
st.set_page_config(page_title="XGBoost Institutional Forecaster", layout="wide")
st.title("🛡️ Institutional Engine: XGBoost + Volume Profile & SMC")

# --- SIDEBAR ---
st.sidebar.header("Market Selector")

# Combined FX & Crypto Dropdown
asset_display = st.sidebar.selectbox(
    "Select Asset Pair",
    [
        "EUR/USD", "GBP/USD", "AUD/USD", "USD/CAD", "USD/JPY",  # Forex
        "Bitcoin (BTC/USD)", "Ethereum (ETH/USD)", "Solana (SOL/USD)", "Ripple (XRP/USD)"  # Crypto
    ],
    index=5  # Defaults to Bitcoin
)

# Convert display names to Yahoo Finance Tickers
ticker_mapping = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/JPY": "JPY=X",
    "Bitcoin (BTC/USD)": "BTC-USD",
    "Ethereum (ETH/USD)": "ETH-USD",
    "Solana (SOL/USD)": "SOL-USD",
    "Ripple (XRP/USD)": "XRP-USD"
}
symbol = ticker_mapping[asset_display]

# Scale factor mapping to normalize "pip/movement size" across very different asset classes
scale_mapping = {
    "EUR/USD": 0.0001,
    "GBP/USD": 0.0001,
    "AUD/USD": 0.0001,
    "USD/CAD": 0.0001,
    "USD/JPY": 0.01,
    "Bitcoin (BTC/USD)": 1.0,  # $1.00 move
    "Ethereum (ETH/USD)": 1.0,  # $1.00 move
    "Solana (SOL/USD)": 0.1,  # $0.10 move
    "Ripple (XRP/USD)": 0.001  # Tenth of a cent move
}
point_scale = scale_mapping[asset_display]

timeframe = st.sidebar.selectbox("Select Timeframe", ["5m", "15m", "1h"], index=1)
chart_style = st.sidebar.radio("Select Chart Style", ["Candlestick", "Line Chart"], index=0)
intent = st.sidebar.radio("What Action are you Planning?", ["BUY / GO LONG", "SELL / GO SHORT"])

if timeframe == "5m":
    period = "10d"
elif timeframe == "15m":
    period = "30d"
else:
    period = "90d"

# --- RUN ENGINE ---
if st.sidebar.button("Analyze & Train Engine"):
    with st.spinner("Mining institutional volume profiles and training XGBoost model..."):
        data = yf.download(symbol, period=period, interval=timeframe)
        if data.empty:
            st.error("Market data feed offline. Try again shortly.")
            st.stop()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Base series
        close = data['Close'].squeeze().astype(float)
        open_p = data['Open'].squeeze().astype(float)
        high = data['High'].squeeze().astype(float)
        low = data['Low'].squeeze().astype(float)
        volume = data['Volume'].squeeze().astype(float)

        # 1. MOVEMENT/VOLATILITY CALCULATION (Normalized)
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()
        data['Normalized_Movement'] = data['ATR'] / point_scale

        # 2. SMC ORDER BLOCKS (OB)
        data['Bullish_OB'] = np.nan
        data['Bearish_OB'] = np.nan

        for i in range(2, len(data)):
            # Bullish OB check: Engulfing of previous down close
            if close.iloc[i] > high.iloc[i - 1] and close.iloc[i - 1] < open_p.iloc[i - 1]:
                data.iloc[i, data.columns.get_loc('Bullish_OB')] = low.iloc[i - 1]
            # Bearish OB check: Engulfing of previous up close
            if close.iloc[i] < low.iloc[i - 1] and close.iloc[i - 1] > open_p.iloc[i - 1]:
                data.iloc[i, data.columns.get_loc('Bearish_OB')] = high.iloc[i - 1]

        data['Bullish_OB'] = data['Bullish_OB'].ffill()
        data['Bearish_OB'] = data['Bearish_OB'].ffill()

        # 3. CLASSIC PIVOT POINTS
        data['PP'] = (high + low + close) / 3
        data['R1'] = (2 * data['PP']) - low
        data['S1'] = (2 * data['PP']) - high

        # 4. VOLUME PROFILE / POINT OF CONTROL (POC)
        price_min, price_max = close.min(), close.max()
        bins = np.linspace(price_min, price_max, num=20)

        # Aggregate volume per bin
        bin_indices = np.digitize(close, bins) - 1
        volume_by_bin = np.zeros(len(bins))
        for idx, vol in zip(bin_indices, volume):
            if 0 <= idx < len(bins):
                volume_by_bin[idx] += vol

        poc_bin_index = np.argmax(volume_by_bin)
        poc_price = bins[poc_bin_index]

        # 5. XGBoost FEATURE PREPARATION
        df_feats = pd.DataFrame(index=data.index)
        df_feats['Normalized_Movement'] = data['Normalized_Movement']
        df_feats['Volume'] = volume
        df_feats['Dist_to_BullOB'] = (close - data['Bullish_OB']).fillna(0)
        df_feats['Dist_to_BearOB'] = (close - data['Bearish_OB']).fillna(0)
        df_feats['Dist_to_POC'] = (close - poc_price)
        df_feats['Dist_to_PP'] = (close - data['PP'])

        # Target: 1 if the price rises over the next 3 candles, 0 otherwise
        df_feats['Target'] = (close.shift(-3) > close).astype(int)
        df_feats = df_feats.dropna()

        # Train/Test Split
        X = df_feats[['Normalized_Movement', 'Volume', 'Dist_to_BullOB', 'Dist_to_BearOB', 'Dist_to_POC', 'Dist_to_PP']]
        y = df_feats['Target']

        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        # Train XGBoost Classifier Model
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict current probability
        latest_live_features = pd.DataFrame([[
            data['Normalized_Movement'].iloc[-1],
            volume.iloc[-1],
            (close.iloc[-1] - data['Bullish_OB'].iloc[-1]),
            (close.iloc[-1] - data['Bearish_OB'].iloc[-1]),
            (close.iloc[-1] - poc_price),
            (close.iloc[-1] - data['PP'].iloc[-1])
        ]], columns=X.columns)

        upward_probability = model.predict_proba(latest_live_features)[0][1]

        # --- PROCESS INTERACTIVE INSTRUCTIONS ---
        st.subheader("🤖 XGBoost Live Trading Execution Recommendation")

        live_price = close.iloc[-1]
        currency_format = f"${live_price:.5f}" if "USD=X" in symbol or "CAD=X" in symbol or "JPY=X" in symbol else f"${live_price:,.2f}"

        if intent == "BUY / GO LONG":
            if upward_probability > 0.65:
                st.success(
                    f"🟢 **RIGHT TIME TO BUY NOW!**\n\n* **Current Spot Price:** {currency_format}\n* **Upward Trend Probability:** {upward_probability * 100:.1f}%\n* **Analysis:** Institutional accumulation is complete. Current price is sitting optimally above support structures and the Order Block.")
            elif upward_probability > 0.45:
                ob_format = f"${data['Bullish_OB'].iloc[-1]:.5f}" if "USD=X" in symbol or "CAD=X" in symbol or "JPY=X" in symbol else f"${data['Bullish_OB'].iloc[-1]:,.2f}"
                st.warning(
                    f"🟡 **HOLD ON: WAIT FOR A PULLBACK**\n\n* **Current Spot Price:** {currency_format}\n* **Upward Trend Probability:** {upward_probability * 100:.1f}%\n* **Strategy:** Do not buy yet. Wait for a correction down to the nearest Order Block at **{ob_format}** before executing your long order.")
            else:
                st.error(
                    f"🔴 **DO NOT BUY!**\n\n* **Current Spot Price:** {currency_format}\n* **Upward Trend Probability:** {upward_probability * 100:.1f}%\n* **Analysis:** XGBoost predicts high downward momentum. Entering a buy right now will likely result in a loss.")

        else:  # SELL / GO SHORT
            downward_probability = 1 - upward_probability
            if downward_probability > 0.65:
                st.success(
                    f"🔴 **RIGHT TIME TO ENTER SELL NOW!**\n\n* **Current Spot Price:** {currency_format}\n* **Downward Trend Probability:** {downward_probability * 100:.1f}%\n* **Analysis:** Distribution is complete. Price has rejected swing resistance and the bearish order block. Excellent short setup.")
            elif downward_probability > 0.45:
                ob_format = f"${data['Bearish_OB'].iloc[-1]:.5f}" if "USD=X" in symbol or "CAD=X" in symbol or "JPY=X" in symbol else f"${data['Bearish_OB'].iloc[-1]:,.2f}"
                st.warning(
                    f"🟡 **HOLD ON: WAIT FOR A BETTER ENTRY**\n\n* **Current Spot Price:** {currency_format}\n* **Downward Trend Probability:** {downward_probability * 100:.1f}%\n* **Strategy:** Wait for a short-term rally back up toward the Bearish Order Block boundary at **{ob_format}** or the Pivot Point line to maximize your risk-to-reward ratio.")
            else:
                st.error(
                    f"🟢 **DO NOT SELL!**\n\n* **Current Spot Price:** {currency_format}\n* **Downward Trend Probability:** {downward_probability * 100:.1f}%\n* **Analysis:** Strong upward structural momentum detected. Selling now is trading against institutional flow.")

        # --- PLOTLY DATA VISUALIZATION ---
        fig = go.Figure()

        if chart_style == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index, open=open_p, high=high, low=low, close=close, name=f"{asset_display} Candles"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data.index, y=close, mode='lines',
                line=dict(color='#3498db', width=2), name="Spot Price"
            ))

        # Dynamic Point Of Control (POC) Line
        fig.add_trace(go.Scatter(
            x=data.index, y=[poc_price] * len(data), mode="lines",
            line=dict(color="#e67e22", width=2, dash="dashdot"),
            name=f"Volume POC ({currency_format})"
        ))

        # Support & Resistance Pivot Dot lines
        fig.add_trace(go.Scatter(
            x=data.index, y=data['PP'], mode="lines",
            line=dict(color="rgba(142, 68, 173, 0.6)", width=1, dash="dot"),
            name="Pivot Point"
        ))

        # Shaded Bullish / Bearish Order Block Bands
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Bullish_OB'], mode="lines",
            line=dict(color="rgba(46, 204, 113, 0.4)", width=1.5),
            name="Bullish OB Floor"
        ))

        fig.add_trace(go.Scatter(
            x=data.index, y=data['Bearish_OB'], mode="lines",
            line=dict(color="rgba(231, 76, 60, 0.4)", width=1.5),
            name="Bearish OB Ceiling"
        ))

        fig.update_layout(
            height=600,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select your asset pair, set your trading bias in the sidebar, and click 'Analyze & Train Engine'.")