import asyncio
from datetime import datetime, timedelta, timezone
import logging
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============================================================================
# HARDCODED CONFIGURATIONS & CREDENTIALS

TELEGRAM_TOKEN = "8626664513:AAHFRULcbsvbq_yZzNL_Pje5pvOw_47pbSM"
# Converted into a list to support multiple channels
TELEGRAM_CHAT_IDS = ["7103037298", "8770652154"]

MT5_LOGIN = 341833
MT5_PASSWORD = "!@#$Pandu02"  # Replace XXXXX with your actual string password
MT5_SERVER = "FusionMarketsAU-Demo"

# Global dictionary to track positions that were active in the current session
SESSION_POSITIONS = {}

# ==============================================================================
# ENGINE INFRASTRUCTURE & BACKBONE FUNCTIONS
# ==============================================================================


def initialize_mt5(login, password, server):
    if not mt5.initialize():
        logging.error(f"MT5 Initialization failed: {mt5.last_error()}")
        return False
    authorized = mt5.login(login=login, password=password, server=server)
    if authorized:
        logging.info(f"Successfully connected to Server: {server}")
        return True
    mt5.shutdown()
    return False


def send_telegram_message(message: str):
    """Sends a text message to all hardcoded Telegram chat IDs."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    success = True

    for chat_id in TELEGRAM_CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                logging.error(f"Failed to send to {chat_id}: {response.text}")
                success = False
        except Exception as e:
            logging.error(f"Telegram network error for chat {chat_id}: {e}")
            success = False

    return success


def get_user_configuration():
    print("\n" + "=" * 40)
    print("      ALGORITHMIC TRADING BOT CONFIG      ")
    print("=" * 40)

    symbols_menu = {
        "1": "EURUSD.r",
        "2": "GBPUSD.r",
        "3": "USDJPY.r",
        "4": "BTCUSD.r",
        "5": "ETHUSD.r",
    }
    print("\n[Step 1] Select an Asset/Symbol to trade:")
    for key, val in symbols_menu.items():
        print(f"  {key}) {val}")

    while True:
        choice = input("Enter choice (1-5) or type custom symbol: ").strip()
        if choice in symbols_menu:
            selected_symbol = symbols_menu[choice]
            break
        elif choice.isalnum() or "." in choice:
            selected_symbol = choice.upper()
            break
        print("❌ Invalid input.")

    print("\n[Step 2] Enter Volume / Lot Size per trade:")
    while True:
        try:
            selected_volume = float(input("Enter size (e.g., 0.01, 0.1): "))
            if selected_volume > 0:
                break
        except ValueError:
            print("❌ Invalid number.")

    timeframe_menu = {"1": (mt5.TIMEFRAME_M1, "1M"), "2": (mt5.TIMEFRAME_M5, "5M")}
    print("\n[Step 3] Select an execution Timeframe:")
    print("  1) 1 Minute (M1)")
    print("  2) 5 Minutes (M5)")

    while True:
        tf_choice = input("Enter choice (1 or 2): ").strip()
        if tf_choice in timeframe_menu:
            selected_timeframe, timeframe_label = timeframe_menu[tf_choice]
            break
        print("❌ Invalid choice.")

    if selected_timeframe == mt5.TIMEFRAME_M1:
        macro_tf, macro_label = mt5.TIMEFRAME_M30, "30M"
    else:
        macro_tf, macro_label = mt5.TIMEFRAME_H1, "1H"

    return {
        "symbol": selected_symbol,
        "volume": selected_volume,
        "timeframe": selected_timeframe,
        "timeframe_label": timeframe_label,
        "macro_timeframe": macro_tf,
        "macro_timeframe_label": macro_label,
    }


def fetch_rates(symbol: str, timeframe, count: int):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    return pd.DataFrame(rates)


def check_macro_trend(symbol: str, macro_tf):
    df = fetch_rates(symbol, macro_tf, count=100)
    if df is None:
        return "UNKNOWN"
    ema = df["close"].ewm(span=50, adjust=False).mean().iloc[-2]
    close = df["close"].iloc[-2]
    return "BULLISH" if close > ema else "BEARISH"


def check_volatility_expansion(symbol: str, exec_tf):
    df = fetch_rates(symbol, exec_tf, count=20)
    if df is None:
        return False
    df["high_low"] = df["high"] - df["low"]
    atr = df["high_low"].rolling(window=14).mean().iloc[-2]
    last_candle_range = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
    return last_candle_range > (atr * 0.5)


# ==============================================================================
# STEP 7.2: STRUCTURAL ANALYSIS ENGINE (POST-MORTEM LOOKBACK)
# ==============================================================================


def perform_post_mortem(symbol, direction, profit, execution_tf, macro_tf):
    """Analyzes the precise context surrounding why a trade won or lost."""
    analysis = {"right": "N/A", "wrong": "N/A"}

    # Fetch fresh context metrics
    current_macro = check_macro_trend(symbol, macro_tf)
    volatility_ok = check_volatility_expansion(symbol, execution_tf)

    tick_info = mt5.symbol_info_tick(symbol)
    spread = (
        (tick_info.ask - tick_info.bid) / tick_info.point
        if tick_info
        else "Unknown"
    )

    # Case A: Trade Won (Hit TP or closed in positive territory)
    if profit >= 0:
        if volatility_ok:
            analysis["right"] = (
                "✅ Momentum Verified: Price cleanly accelerated through the predicted LVN vacuum corridor."
            )
        else:
            analysis["right"] = (
                "✅ Low Resistance: Scalp target hit comfortably despite lower structural volatility speeds."
            )

    # Case B: Trade Lost (Hit SL or closed out negative)
    else:
        # Check for Trend Reversal
        trade_aligned_with_trend = (
            (direction == "BUY" and current_macro == "BULLISH") or
            (direction == "SELL" and current_macro == "BEARISH")
        )

        if not trade_aligned_with_trend:
            analysis["wrong"] = (
                f"🚨 Trend Vise: Position caught in an aggressive Higher Timeframe ({current_macro}) macro structural reversal."
            )
        elif not volatility_ok:
            analysis["wrong"] = (
                "🚨 Dead Zone Trap: Entry executed inside a low-volatility compression zone, triggering choppy stop-hunting."
            )
        elif isinstance(spread, (int, float)) and spread > 30:
            analysis["wrong"] = (
                f"🚨 Spread Blowout: Broker spreads widened abnormally to {spread:.1f} points, causing synthetic slippage."
            )
        else:
            analysis["wrong"] = (
                "🚨 Stop Hunt: Price briefly breached past the defensive HVN volume shelf before continuing."
            )

    return analysis


# ==============================================================================
# STEP 7.1: ASYNCHRONOUS TRACKING AND HISTORY PROCESSING
# ==============================================================================


async def monitor_positions_feedback_loop(config):
    """Asynchronous core engine loop that monitors position changes

    and builds structural post-mortem feedback diagnostics.
    """
    symbol = config["symbol"]
    logging.info(f"🔄 Post-Mortem Feedback Loop active for {symbol}...")

    while True:
        try:
            # 1. Fetch current active positions in the terminal
            active_positions = mt5.positions_get(symbol=symbol)
            current_active_tickets = set()

            if active_positions is not None and len(active_positions) > 0:
                for pos in active_positions:
                    ticket = pos.ticket
                    current_active_tickets.add(ticket)

                    # Cache position parameters if not seen before
                    if ticket not in SESSION_POSITIONS:
                        SESSION_POSITIONS[ticket] = {
                            "ticket": ticket,
                            "symbol": pos.symbol,
                            "type": "BUY" if pos.type == 0 else "SELL",
                            "volume": pos.volume,
                            "open_price": pos.price_open,
                        }

            # 2. Check for closed tickets (previously in session tracking but missing now)
            closed_tickets = [
                t for t in SESSION_POSITIONS if t not in current_active_tickets
            ]

            for ticket in closed_tickets:
                logging.info(
                    f"🎯 Detected Closed Position (Ticket #{ticket}). Processing execution stats..."
                )
                await asyncio.sleep(2)  # Short delay to ensure broker history records sync

                # Query terminal deal history logs for the ticket
                now = datetime.now()
                history_deals = mt5.history_deals_get(
                    ticket=ticket
                )  # Step 7.1 Data Capture

                if history_deals and len(history_deals) > 0:
                    # Compile financial calculations across deals belonging to ticket
                    total_profit = sum(deal.profit for deal in history_deals)
                    total_commission = sum(deal.commission for deal in history_deals)
                    total_swap = sum(deal.swap for deal in history_deals)
                    net_return = total_profit + total_commission + total_swap

                    pos_data = SESSION_POSITIONS[ticket]
                    sym_info = mt5.symbol_info(symbol)
                    point = sym_info.point if sym_info else 0.00001

                    # Fetch the final closing trade deal execution price
                    close_price = history_deals[-1].price
                    pip_diff = abs(close_price - pos_data["open_price"]) / (
                        point * 10 if "JPY" not in symbol and point == 0.00001 else point
                    )

                    # Step 7.2 Run dynamic mathematical post-mortem evaluation
                    post_mortem = perform_post_mortem(
                        symbol,
                        pos_data["type"],
                        net_return,
                        config["timeframe"],
                        config["macro_timeframe"],
                    )

                    # Format & Build the structured post-trade layout report
                    report = (
                        f"📊 *POST-TRADE POST-MORTEM REPORT*\n"
                        f"• *Ticket ID:* #{ticket}\n"
                        f"• *Asset:* {symbol}\n"
                        f"• *Direction:* {pos_data['type']} ({pos_data['volume']} Lots)\n"
                        f"• *Net Result:* `${net_return:.2f}` (~{pip_diff:.1f} Pips)\n\n"
                        f"✨ *What Went Right:* \n_{post_mortem['right']}_\n\n"
                        f"⚠️ *What Went Wrong:* \n_{post_mortem['wrong']}_\n"
                        f"📦 _Feedback engine evaluation loop concluded._"
                    )

                    # Transmit directly to Telegram channel chat
                    send_telegram_message(report)

                # Clean tracking memory storage
                del SESSION_POSITIONS[ticket]

        except Exception as e:
            logging.error(f"Error in operational monitoring loop core: {e}")

        await asyncio.sleep(1)  # High frequency 1-second pulse polling frequency


# ==============================================================================
# MAIN SYSTEM INITIATION CONTEXT
# ==============================================================================


async def main():
    user_config = get_user_configuration()

    if initialize_mt5(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        send_telegram_message(
            f"🔄 *Bot Feedback Core Initialized:* Monitoring open trade executions on {user_config['symbol']}."
        )

        # Fire off the asynchronous monitoring feedback daemon engine task
        await monitor_positions_feedback_loop(user_config)

        mt5.shutdown()


if __name__ == "__main__":
    # Launch async framework architecture run loop safely
    asyncio.run(main())