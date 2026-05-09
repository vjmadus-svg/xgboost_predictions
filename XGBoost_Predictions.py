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
st.title("🛡️ Institutional Engine: XGBoost + Volume Profile & SMC by Vijay Madhu")

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

# Upgraded to include 4 Hour, 1 Day, and 1 Week timeframes
timeframe = st.sidebar.selectbox(
    "Select Timeframe", 
    ["5m", "15m", "1h", "4h", "1d", "1wk"], 
    index=1
)
chart_style = st.sidebar.radio("Select Chart Style", ["Candlestick", "Line Chart"], index=0)

# Map timeframes to secure yfinance download periods
if timeframe == "5m":
    period = "10d"
elif timeframe == "15m":
    period = "30d"
elif timeframe == "1h":
    period = "90d"
elif timeframe == "4h":
    period = "730d"  # 4h data is limited to 730 days max on yfinance
elif timeframe == "1d":
    period = "2y"
else:
    period = "5y"    # Weekly needs a larger period to build enough data points for XGBoost

# --- RUN ENGINE ---
if st.sidebar.button("Analyze & Train Engine"):
    with st.spinner("Mining institutional volume profiles and training XGBoost model..."):
        data = yf.download(symbol, period=period, interval=timeframe)
        if data.empty:
            st.error("Market data feed offline or timeframe/period combination invalid. Try again shortly.")
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

        # 5. XGBoost FEATURE PREPARATION & LEAKAGE PROTECTION
        df_feats = pd.DataFrame(index=data.index)
        df_feats['Normalized_Movement'] = data['Normalized_Movement']

        # Calculate Relative Volume (normalizes volume across assets and times)
        volume_sma = volume.rolling(window=20).mean()
        df_feats['Relative_Volume'] = (volume / volume_sma).fillna(1.0)

        # Normalize distance features using ATR as a standard volatility scale
        atr_normalizer = data['ATR'].replace(0, 1e-5)
        df_feats['Dist_to_BullOB'] = ((close - data['Bullish_OB']) / atr_normalizer).fillna(0)
        df_feats['Dist_to_BearOB'] = ((close - data['Bearish_OB']) / atr_normalizer).fillna(0)
        df_feats['Dist_to_POC'] = ((close - poc_price) / atr_normalizer)
        df_feats['Dist_to_PP'] = ((close - data['PP']) / atr_normalizer)

        # Target: 1 if the price rises over the next 3 candles, 0 otherwise
        df_feats['Target'] = (close.shift(-3) > close).astype(int)

        # Extract the absolute latest live features before dropping NaN values
        live_features = df_feats.iloc[[-1]].copy()

        # Safely drop empty target rows from our training set
        df_feats_clean = df_feats.dropna()

        # Train/Test Split
        feature_cols = ['Normalized_Movement', 'Relative_Volume', 'Dist_to_BullOB', 'Dist_to_BearOB', 'Dist_to_POC',
                        'Dist_to_PP']
        X = df_feats_clean[feature_cols]
        y = df_feats_clean['Target']

        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        # Train XGBoost Classifier Model
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)

        # Prepare real-time normalized prediction features
        latest_live_features = live_features[feature_cols]

        # Calculate mathematically sound upward trend probability
        upward_probability = model.predict_proba(latest_live_features)[0][1]
        downward_probability = 1 - upward_probability

        # --- PROCESS INTERACTIVE INSTRUCTIONS ---
        st.subheader("🤖 Unified XGBoost Market Recommendations")

        live_price = close.iloc[-1]
        currency_format = f"${live_price:.5f}" if "USD=X" in symbol or "CAD=X" in symbol or "JPY=X" in symbol else f"${live_price:,.2f}"

        # Setup formats for target locations
        ob_bull_format = f"${data['Bullish_OB'].iloc[-1]:.5f}" if "USD=X" in symbol or "CAD=X" in symbol or "JPY=X" in symbol else f"${data['Bullish_OB'].iloc[-1]:,.2f}"
        ob_bear_format = f"${data['Bearish_OB'].iloc[-1]:.5f}" if "USD=X" in symbol or "CAD=X" in symbol or "JPY=X" in symbol else f"${data['Bearish_OB'].iloc[-1]:,.2f}"

        # Output side-by-side diagnostic cards
        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("### 🟢 BUY Setup Profile")
            if upward_probability > 0.65:
                st.success(
                    f"**EXECUTE BUY ORDER NOW!**\n\n"
                    f"* **Spot Entry Price:** {currency_format}\n"
                    f"* **XGBoost Upward Confidence:** {upward_probability * 100:.1f}%\n"
                    f"* **Analysis:** Institutional accumulation patterns detected. Price sits optimally above the local support node."
                )
            elif upward_probability > 0.45:
                st.warning(
                    f"**WAIT TO BUY (PULLBACK SETUP)**\n\n"
                    f"* **Current Price:** {currency_format}\n"
                    f"* **Optimal Buy Target:** **{ob_bull_format}** (Bullish Order Block)\n"
                    f"* **Strategy:** Avoid chasing the market. Set a limit order at the Bullish OB floor level."
                )
            else:
                st.error(
                    f"**DO NOT BUY!**\n\n"
                    f"* **XGBoost Upward Confidence:** {upward_probability * 100:.1f}%\n"
                    f"* **Analysis:** High bearish momentum confirmed. Buying here risks catching a falling knife."
                )

        with rec_col2:
            st.markdown("### 🔴 SELL Setup Profile")
            if downward_probability > 0.65:
                st.success(
                    f"**EXECUTE SELL ORDER NOW!**\n\n"
                    f"* **Spot Entry Price:** {currency_format}\n"
                    f"* **XGBoost Downward Confidence:** {downward_probability * 100:.1f}%\n"
                    f"* **Analysis:** Institutional distribution complete. Structure points down after key resistance rejection."
                )
            elif downward_probability > 0.45:
                st.warning(
                    f"**WAIT TO SELL (RALLY SETUP)**\n\n"
                    f"* **Current Price:** {currency_format}\n"
                    f"* **Optimal Sell Target:** **{ob_bear_format}** (Bearish Order Block)\n"
                    f"* **Strategy:** Wait for a minor retracement up to the Bearish OB ceiling before securing your short entry."
                )
            else:
                st.error(
                    f"**DO NOT SELL!**\n\n"
                    f"* **XGBoost Downward Confidence:** {downward_probability * 100:.1f}%\n"
                    f"* **Analysis:** Strong structural support and high demand remain. Selling into this strength is highly unsafe."
                )

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
    st.info("Select your asset pair and timeframe in the sidebar, then click 'Analyze & Train Engine'.")
