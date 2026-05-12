"""
╔══════════════════════════════════════════════════════════════╗
║        INFINITE EXTREME SCALPING MT5 TRADING BOT             ║
║                                                              ║
║   Strategies Integrated:                                     ║
║   ✓ EMA Crossover Filter (9 EMA / 21 EMA)                    ║
║   ✓ Bollinger Bands Reversal Band-Walk & Snapback            ║
║   ✓ High-Velocity Candlestick Price Rejection (Candle Tails) ║
║   ✓ Persistent Reset Loop Engine (Non-stop hunting)          ║
║   ✓ Interactive Startup Asset Option Interface               ║
╚══════════════════════════════════════════════════════════════╝
"""

import time
import sys
import subprocess


def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


for _pkg, _pip in [("pandas", "pandas"), ("numpy", "numpy")]:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"  Installing {_pip}...")
        _install(_pip)

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
import urllib.parse
import json

# ══════════════════════════════════════════════════════════════
# TELEGRAM CONFIGURATION
# ══════════════════════════════════════════════════════════════
TELEGRAM_TOKEN = "8626664513:AAHFRULcbsvbq_yZzNL_Pje5pvOw_47pbSM"
TELEGRAM_CHAT_ID = "7103037298"
TELEGRAM_ENABLED = True

ALERT_ON_SIGNAL = True
ALERT_ON_TRADE_OPEN = True
ALERT_ON_TRADE_CLOSE = True
ALERT_ON_DAILY_SUMMARY = True
ALERT_ON_ERROR = True

# ══════════════════════════════════════════════════════════════
# EXTREME SCALPING TARGET LIMITS (USD BASED)
# ══════════════════════════════════════════════════════════════
PROFIT_TARGET_USD = 10.0  # Close trade at +$10 floating profit
LOSS_LIMIT_USD = 5.0  # Close trade at -$5 floating loss

# ══════════════════════════════════════════════════════════════
# EXPANDED GLOBAL INSTRUMENT CATALOGUE WITH USER POOL (.r Suffix)
# ══════════════════════════════════════════════════════════════
INSTRUMENTS = {
    # --- Major Currency Pairs ---
    "1": dict(name="EUR/USD", symbol="EURUSD.r", sl_price=0.0008, tp_price=0.0016, lot=0.50),
    "2": dict(name="GBP/USD", symbol="GBPUSD.r", sl_price=0.0010, tp_price=0.0020, lot=0.50),
    "3": dict(name="AUD/USD", symbol="AUDUSD.r", sl_price=0.0008, tp_price=0.0016, lot=0.50),
    "4": dict(name="USD/JPY", symbol="USDJPY.r", sl_price=0.100, tp_price=0.200, lot=0.50),
    "5": dict(name="USD/CAD", symbol="USDCAD.r", sl_price=0.0008, tp_price=0.0016, lot=0.50),
    "6": dict(name="USD/CHF", symbol="USDCHF.r", sl_price=0.0008, tp_price=0.0016, lot=0.50),
    "7": dict(name="NZD/USD", symbol="NZDUSD.r", sl_price=0.0009, tp_price=0.0018, lot=0.50),

    # --- Precious Metals ---
    "8": dict(name="Gold (XAU/USD)", symbol="XAUUSD.r", sl_price=1.50, tp_price=3.00, lot=0.10),
    "9": dict(name="Silver (XAG/USD)", symbol="XAGUSD.r", sl_price=0.15, tp_price=0.35, lot=0.20),

    # --- Global Indices ---
    "10": dict(name="US Wall St 30 (Dow Jones)", symbol="US30.r", sl_price=15.0, tp_price=35.0, lot=0.10),
    "11": dict(name="US Tech 100 (NASDAQ)", symbol="USTEC.r", sl_price=10.0, tp_price=25.0, lot=0.10),
    "12": dict(name="US SPX 500 (S&P 500)", symbol="US500.r", sl_price=3.0, tp_price=7.0, lot=0.20),
    "13": dict(name="Germany 40 (DAX)", symbol="GER40.r", sl_price=8.0, tp_price=20.0, lot=0.10),
}

# ══════════════════════════════════════════════════════════════
# EXTENDED SCALPING SYSTEM CONFIGURATION
# ══════════════════════════════════════════════════════════════
SCALPING_TIMEFRAME = mt5.TIMEFRAME_M1  # M1 Chart execution
MAX_DAILY_TRADES = 500  # High-frequency upper cap
COOLDOWN_SECONDS = 5  # Accelerated reset safety buffer

# Indicator Tuning Params
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
BB_PERIOD = 20
BB_DEV = 2.0
CANDLE_TAIL_RATIO = 0.45  # Rejection tail ratio filter

# ══════════════════════════════════════════════════════════════
# DATA TRACKER MATRICES
# ══════════════════════════════════════════════════════════════
open_trade_register: dict = {}
session_stats = {
    "total": 0, "wins": 0, "losses": 0, "breakeven": 0,
    "total_profit": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
}
daily_trade_count = 0
last_trade_date = None


class TradeState:
    def __init__(self):
        self.last_trade_time = None
        self.position_open_time = None

    def can_trade(self):
        if daily_trade_count >= MAX_DAILY_TRADES:
            return False
        if self.last_trade_time is not None:
            seconds_passed = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_passed < COOLDOWN_SECONDS:
                return False
        return True

    def record_trade(self):
        self.last_trade_time = datetime.now()
        self.position_open_time = datetime.now()


trade_state = TradeState()


# ══════════════════════════════════════════════════════════════
# TELEGRAM ENGINE
# ══════════════════════════════════════════════════════════════
def _tg_send(message: str, silent: bool = False) -> bool:
    if not TELEGRAM_ENABLED:
        return True
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": str(silent).lower(),
        }).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")
        return False


def test_telegram():
    print("\n  📱 Testing Telegram connection...")
    ok = _tg_send(
        f"⚡ <b>Infinite Extreme Scalper Booted!</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Target: +${PROFIT_TARGET_USD:.2f} | Loss: -${LOSS_LIMIT_USD:.2f}\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if ok:
        print("  ✅ Telegram link active.")
    else:
        print("  ❌ Telegram link setup failed.")
    return ok


def alert_signal(display_name, mt5_symbol, signal, reason, entry_price, sl, tp):
    if not ALERT_ON_SIGNAL:
        return
    icon = "🟢" if signal == "BUY" else "🔴"
    _tg_send(
        f"⚡ <b>SCALPING SIGNAL DETECTED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{icon} <b>{signal}</b> | {display_name} ({mt5_symbol})\n"
        f"📋 <b>Triggers:</b> {reason}\n\n"
        f"💹 <b>Entry:</b> {entry_price}\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def alert_trade_opened(display_name, mt5_symbol, signal, entry_price, sl, tp, lot, ticket, trade_num):
    if not ALERT_ON_TRADE_OPEN:
        return
    icon = "🟢" if signal == "BUY" else "🔴"
    _tg_send(
        f"🚀 <b>SCALP POSITION OPENED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{icon} <b>{signal}</b> | {display_name}\n"
        f"🎫 <b>Ticket:</b> #{ticket}\n"
        f"📦 <b>Lot size:</b> {lot}\n\n"
        f"💹 <b>Entry:</b> {entry_price}\n"
        f"🛑 <b>SL:</b>    {sl}\n"
        f"✅ <b>TP:</b>    {tp}"
    )


def alert_trade_closed(display_name, mt5_symbol, ticket, direction, entry_price, close_price, profit, reason, s):
    if not ALERT_ON_TRADE_CLOSE:
        return
    ri = "✅" if profit > 0.01 else "❌" if profit < -0.01 else "⚖️"
    rl = f"PROFIT +${profit:.2f}" if profit > 0.01 else f"LOSS -${abs(profit):.2f}" if profit < -0.01 else "BREAKEVEN"
    icon = "🟢" if direction == "BUY" else "🔴"
    wr = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0
    _tg_send(
        f"{ri} <b>SCALP POSITION CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{icon} <b>{direction}</b> | {display_name}\n"
        f"🎫 <b>Ticket:</b> #{ticket}\n\n"
        f"🏁 <b>Exit Price:</b> {close_price}\n"
        f"📋 <b>Reason:</b>     {reason}\n"
        f"{ri} <b>Result:</b>     {rl}\n\n"
        f"📊 WR: {wr:.1f}% | Account Session P&L: ${s['total_profit']:.2f}"
    )


def alert_pnl_exit(display_name, mt5_symbol, ticket, direction, entry_price, exit_price, profit, reason):
    icon = "✅" if profit >= 0 else "❌"
    _tg_send(
        f"{icon} <b>FAST TARGET POSITION EXIT</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎫 #{ticket} | {display_name} ({direction})\n"
        f"🏁 <b>Exit:</b> {exit_price}\n"
        f"💰 <b>P&L:</b> {'+' if profit >= 0 else ''}{profit:.2f}\n"
        f"📝 <b>Reason:</b> {reason}"
    )


def alert_error(context, detail):
    if not ALERT_ON_ERROR:
        return
    _tg_send(f"⚠️ <b>BOT ERROR</b>\n<b>Ctx:</b> {context}\n❗ {detail}")


# ══════════════════════════════════════════════════════════════
# DATA ACQUISITION & STRUCTURAL SCALPING PROCESSING
# ══════════════════════════════════════════════════════════════
def get_live_data(mt5_symbol, timeframe, num_bars=100):
    rates = mt5.copy_rates_from_pos(mt5_symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'},
              inplace=True)
    return df


def analyze_scalping_signals(df):
    if df is None or len(df) < max(EMA_SLOW_PERIOD, BB_PERIOD):
        return "WAIT", "Insufficient Bars Loaded"

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    open_p = df['Open'].astype(float)

    # 1. EMA Calculations
    ema_fast = close.ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()

    # 2. Bollinger Bands
    bb_mid = close.rolling(window=BB_PERIOD).mean()
    bb_std = close.rolling(window=BB_PERIOD).std()
    bb_upper = bb_mid + (BB_DEV * bb_std)
    bb_lower = bb_mid - (BB_DEV * bb_std)

    # Current Live / Most Recent Bar Metrics
    curr_close = close.iloc[-1]
    curr_high = high.iloc[-1]
    curr_low = low.iloc[-1]
    curr_open = open_p.iloc[-1]

    curr_fast = ema_fast.iloc[-1]
    curr_slow = ema_slow.iloc[-1]
    curr_upper = bb_upper.iloc[-1]
    curr_lower = bb_lower.iloc[-1]

    prev_fast = ema_fast.iloc[-2]
    prev_slow = ema_slow.iloc[-2]

    # Candlestick structural tail metrics
    candle_range = max((curr_high - curr_low), 1e-8)
    body_range = abs(curr_close - curr_open)
    upper_wick = curr_high - max(curr_open, curr_close)
    lower_wick = min(curr_open, curr_close) - curr_low

    # ══════════════════════════════════════════════════════════
    # INTERCONNECTED CONVERGENCE RULES
    # ══════════════════════════════════════════════════════════
    # CONDITION A: BULLISH REVERSAL SNAPBACK
    ema_bullish = (curr_fast > curr_slow) or (prev_fast <= prev_slow and curr_fast > curr_slow)
    bb_bullish = (curr_low <= curr_lower) or (curr_close <= curr_lower * 1.0005)
    pinbar_bullish = (lower_wick / candle_range >= CANDLE_TAIL_RATIO) and (body_range / candle_range < 0.4)

    # CONDITION B: BEARISH REVERSAL SNAPBACK
    ema_bearish = (curr_fast < curr_slow) or (prev_fast >= prev_slow and curr_fast < curr_slow)
    bb_bearish = (curr_high >= curr_upper) or (curr_close >= curr_upper * 0.9995)
    pinbar_bearish = (upper_wick / candle_range >= CANDLE_TAIL_RATIO) and (body_range / candle_range < 0.4)

    if bb_bullish and pinbar_bullish and ema_bullish:
        reason = f"Bullish Convergence: BB Touch + Pin Rejection Tail ({lower_wick / candle_range:.2f})"
        return "BUY", reason

    if bb_bearish and pinbar_bearish and ema_bearish:
        reason = f"Bearish Convergence: BB Touch + Pin Rejection Tail ({upper_wick / candle_range:.2f})"
        return "SELL", reason

    return "WAIT", "Market Balancing"


# ══════════════════════════════════════════════════════════════
# BROKER UTILITIES
# ══════════════════════════════════════════════════════════════
def get_filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    if info is None: return mt5.ORDER_FILLING_FOK
    fm = info.filling_mode
    if fm & 1: return mt5.ORDER_FILLING_FOK
    if fm & 2: return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def normalize_price(symbol, price):
    info = mt5.symbol_info(symbol)
    if info is None: return round(price, 5)
    return round(round(price / info.trade_tick_size) * info.trade_tick_size, info.digits)


def calculate_sl_tp(mt5_symbol, action_type, entry_price, instrument_cfg):
    desired_sl = instrument_cfg['sl_price']
    desired_tp = instrument_cfg['tp_price']

    if action_type == "BUY":
        sl = entry_price - desired_sl
        tp = entry_price + desired_tp
    else:
        sl = entry_price + desired_sl
        tp = entry_price - desired_tp

    return normalize_price(mt5_symbol, sl), normalize_price(mt5_symbol, tp)


# ══════════════════════════════════════════════════════════════
# MONITOR & RE-ARMING PIPELINE ENGINE
# ══════════════════════════════════════════════════════════════
def _close_position_now(pos, mt5_symbol, reason_label, display_name):
    filling_mode = get_filling_mode(mt5_symbol)
    tick = mt5.symbol_info_tick(mt5_symbol)
    close_price = tick.bid if pos.type == 0 else tick.ask
    norm_price = normalize_price(mt5_symbol, close_price)
    captured_profit = pos.profit

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": float(pos.volume),
        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "position": pos.ticket,
        "price": float(norm_price),
        "deviation": 15,
        "magic": 20260512,
        "comment": reason_label[:30],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  ❌ Emergency stop execution failure on position #{pos.ticket}: code {res.retcode}")
        return None, None

    direction = "BUY" if pos.type == 0 else "SELL"
    if pos.ticket in open_trade_register:
        open_trade_register[pos.ticket]["captured_profit"] = captured_profit
        open_trade_register[pos.ticket]["exit_price"] = norm_price

    alert_pnl_exit(display_name, mt5_symbol, pos.ticket, direction, pos.price_open, norm_price, captured_profit,
                   reason_label)
    trade_state.position_open_time = None
    return norm_price, captured_profit


def monitor_pnl_targets(mt5_symbol, display_name=""):
    positions = mt5.positions_get(symbol=mt5_symbol) or []
    if not positions:
        return

    for pos in positions:
        profit = pos.profit
        if profit >= PROFIT_TARGET_USD:
            print(f"  🎯 TAKE PROFIT BALANCER TRIGGERED: +${profit:.2f}")
            _close_position_now(pos, mt5_symbol, "Session Equity Target Reached", display_name)
        elif profit <= -LOSS_LIMIT_USD:
            print(f"  🛑 LOSS LIMIT PROTECTION TRIGGERED: -${abs(profit):.2f}")
            _close_position_now(pos, mt5_symbol, "Max Safety Bracket Breach", display_name)


def execute_trade(action_type, mt5_symbol, lot, instrument_cfg, display_name="", reason=""):
    global daily_trade_count, last_trade_date
    today = datetime.now().date()
    if last_trade_date != today:
        daily_trade_count = 0
        last_trade_date = today

    if not trade_state.can_trade():
        return

    positions = mt5.positions_get(symbol=mt5_symbol) or []
    if len(positions) > 0:
        return  # Keep scalper cleanly locked onto single-leg runs

    tick = mt5.symbol_info_tick(mt5_symbol)
    raw_price = tick.ask if action_type == "BUY" else tick.bid
    norm_price = normalize_price(mt5_symbol, raw_price)
    norm_sl, norm_tp = calculate_sl_tp(mt5_symbol, action_type, norm_price, instrument_cfg)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY if action_type == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": float(norm_price),
        "sl": float(norm_sl),
        "tp": float(norm_tp),
        "deviation": 15,
        "magic": 20260512,
        "comment": "Infinite Scalper Engine",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(mt5_symbol),
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  ❌ Scalp trade placement rejected: {result.comment}")
        alert_error(f"Execution Failure {mt5_symbol}", result.comment)
    else:
        daily_trade_count += 1
        trade_state.record_trade()
        print(f"  🚀 SCALP OPENED | {action_type} {mt5_symbol} at {norm_price}")

        alert_trade_opened(display_name, mt5_symbol, action_type, norm_price, norm_sl, norm_tp, lot, result.order,
                           daily_trade_count)

        open_trade_register[result.order] = {
            "symbol": mt5_symbol,
            "direction": action_type,
            "entry": norm_price,
            "sl": norm_sl,
            "tp": norm_tp,
            "lot": lot,
            "open_time": datetime.now(),
        }


def check_closed_trades(mt5_symbol, display_name=""):
    global session_stats
    live_positions = mt5.positions_get(symbol=mt5_symbol) or []
    live_tickets = {p.ticket for p in live_positions}

    # Check if any registered trade is missing from live positions (meaning broker hit TP/SL)
    registered_tickets = list(open_trade_register.keys())
    closed_tickets = [t for t in registered_tickets if t not in live_tickets]

    for ticket in closed_tickets:
        entry_data = open_trade_register.pop(ticket)
        close_profit = entry_data.get("captured_profit", None)
        close_price = entry_data.get("exit_price", None)

        # If closing info wasn't filled by our target loop, query history for broker bracket data
        if close_profit is None:
            time.sleep(0.2)  # Yield briefly for history cache sync
            history = mt5.history_deals_get(position=ticket)
            if history:
                close_profit = sum(deal.profit + deal.commission + deal.swap for deal in history if deal.entry == 1)
                close_price = history[-1].price
            else:
                close_profit = 0.0
                close_price = entry_data["entry"]

        session_stats["total"] += 1
        session_stats["total_profit"] += close_profit
        if close_profit > 0.01:
            session_stats["wins"] += 1
        elif close_profit < -0.01:
            session_stats["losses"] += 1
        else:
            session_stats["breakeven"] += 1

        print(f"  ✨ Scalp Cycle Completed: Ticket #{ticket} | Net Result: ${close_profit:.2f}")
        alert_trade_closed(display_name, mt5_symbol, ticket, entry_data["direction"], entry_data["entry"], close_price,
                           close_profit, "Target Checked", session_stats)


def _divider(title=""):
    print(f"\n⚡─── {title} " + "─" * (65 - len(title)))


# ══════════════════════════════════════════════════════════════
# MAIN MENU USER SELECTION CONFIG INTERFACE
# ══════════════════════════════════════════════════════════════
def select_trading_instrument():
    _divider("EXTREME SCALPER ASSET DEPLOYMENT INTERFACE")
    print(f"{'No.':<4} | {'Asset Class / Pair':<30} | {'MT5 Mapping Name':<15}")
    print("=" * 60)
    for key, value in INSTRUMENTS.items():
        print(f"{key:<4} | {value['name']:<30} | {value['symbol']:<15}")
    print("=" * 60)

    while True:
        choice = input("\nEnter the Asset index number to trade (e.g., 1 for EUR/USD): ").strip()
        if choice in INSTRUMENTS:
            return INSTRUMENTS[choice]
        print("  ❌ Invalid assignment index. Please try again.")


# ══════════════════════════════════════════════════════════════
# MAIN RUNTIME ENGINE PIPELINE
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Initializing Scalper Deployment Array...")
    if not mt5.initialize():
        print("  ❌ Terminal Linkage Initialization Failed.")
        sys.exit(1)

    # Allow user to pick their instrument directly from the dashboard terminal
    cfg = select_trading_instrument()
    chosen_label = cfg['name']
    mt5_symbol = cfg['symbol']

    # Test communications paths before loop launch
    test_telegram()
    _divider("ACTIVE RUNTIME NODE")

    if not mt5.symbol_select(mt5_symbol, True):
        print(f"  ❌ Terminal Matrix failed to map or subscribe to: {mt5_symbol}")
        print("  💡 Check that this symbol is fully visible inside your MT5 Market Watch window.")
        mt5.shutdown()
        sys.exit(1)

    print(f"  🎯 Target Lock: {chosen_label} [{mt5_symbol}] | Interval: M1")
    print("  🔄 Loop status: INFINITE CYCLING ARMED (Persistent hunt active)")

    try:
        while True:
            # 1. Continually check and reconcile trade states instantly
            monitor_pnl_targets(mt5_symbol, chosen_label)
            check_closed_trades(mt5_symbol, chosen_label)

            # 2. Extract latest candles and update multi-variable layers
            df = get_live_data(mt5_symbol, SCALPING_TIMEFRAME, num_bars=100)
            signal, explanation = analyze_scalping_signals(df)

            # 3. Only attempt entry if there are no open positions on the asset
            live_positions = mt5.positions_get(symbol=mt5_symbol) or []
            if len(live_positions) == 0:
                if signal in ["BUY", "SELL"]:
                    alert_signal(chosen_label, mt5_symbol, signal, explanation, df['Close'].iloc[-1], 0, 0)
                    execute_trade(signal, mt5_symbol, cfg['lot'], cfg, chosen_label, explanation)

            # High fidelity scanning iteration cadence
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n  🛑 Engine processing requested manual safety termination sequence.")
        mt5.shutdown()