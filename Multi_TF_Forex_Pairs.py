# requirements:
# pip install yfinance pandas numpy statsmodels tabulate

import datetime as dt
import warnings
import pandas as pd
#import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from tabulate import tabulate

warnings.filterwarnings("ignore")

# -----------------------------------
# CONFIG
# -----------------------------------
PAIRS = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
}

# Define the timeframes to analyze
# Format: (yfinance_interval, label, lookback_days)
TIMEFRAMES = [
    ("1h", "1 Hour", 30),
    ("4h", "4 Hour", 60),
    ("1d", "Daily", 365),
]

ARIMA_ORDER = (2, 1, 2)


# -----------------------------------
# CORE FUNCTIONS
# -----------------------------------
def fetch_data(ticker, interval, days):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        return None

    # Support for MultiIndex columns in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"][ticker].dropna()
    else:
        close = df["Close"].dropna()

    return close


def forecast_next(series):
    try:
        # We use .values to ignore index frequency gaps (weekends/holidays)
        model = ARIMA(series.values, order=ARIMA_ORDER)
        fitted = model.fit()
        return float(fitted.forecast(steps=1)[0])
    except:
        return None


def run_multi_timeframe_forecast():
    all_results = []

    for ticker, label in PAIRS.items():
        for interval_code, interval_label, lookback in TIMEFRAMES:
            series = fetch_data(ticker, interval_code, lookback)

            if series is not None and len(series) > 10:
                next_price = forecast_next(series)
                last_price = series.iloc[-1]

                if next_price:
                    change = next_price - last_price
                    change_pct = (change / last_price) * 100
                    signal = "▲ BULL" if change > 0 else "▼ BEAR"

                    all_results.append([
                        label,
                        interval_label,
                        f"{last_price:.5f}",
                        f"{next_price:.5f}",
                        f"{change_pct:+.3f}%",
                        signal
                    ])

    headers = ["Pair", "Timeframe", "Last Price", "Forecast", "% Change", "Signal"]
    print(f"\nForex Multi-Timeframe Forecast | {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(tabulate(all_results, headers=headers, tablefmt="grid", stralign="center"))


if __name__ == "__main__":
    run_multi_timeframe_forecast()
