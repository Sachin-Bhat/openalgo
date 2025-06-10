import threading
import time
import signal
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from openalgo import api

# Initialize OpenAlgo client
client = api(
    api_key="openalgo-api-key",
    host="http://127.0.0.1:5000",
    ws_url="ws://127.0.0.1:8765"
)

# Configuration
STRATEGY_NAME = "EMA_Crossover_RELIANCE"
SYMBOL = "RELIANCE"
EXCHANGE = "NSE"
QUANTITY = 1
PRODUCT = "MIS"
PRICE_TYPE = "MARKET"
STOPLOSS_BUFFER = 10.0
TARGET_BUFFER = 20.0
instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]

# State Variables
ltp = None
in_position = False
entry_price = None
stoploss_price = None
target_price = None
current_position = None
exit_signal = False
stop_event = threading.Event()

# WebSocket LTP Handler
def on_data_received(data):
    global ltp, exit_signal
    if data.get("type") == "market_data" and data.get("symbol") == SYMBOL:
        ltp = float(data["data"]["ltp"])
        print(f"LTP Update {EXCHANGE}:{SYMBOL} => ₹{ltp}")
        if in_position and not exit_signal:
            if ltp <= stoploss_price or ltp >= target_price:
                print(f"Exit Triggered: LTP ₹{ltp} hit stoploss or target.")
                exit_signal = True

# WebSocket Thread
def websocket_thread():
    try:
        client.connect()
        client.subscribe_ltp(instrument, on_data_received=on_data_received)
        print("WebSocket LTP thread started.")
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        print("Shutting down WebSocket...")
        client.unsubscribe_ltp(instrument)
        client.disconnect()
        print("WebSocket connection closed.")

# EMA Signal Logic
def get_latest_signals():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)

    df = client.history(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        interval="5m",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    df.ta.ema(length=5, append=True)
    df.ta.ema(length=10, append=True)

    if len(df) < 3:
        print("Waiting for sufficient data...")
        return None

    prev = df.iloc[-3]
    last = df.iloc[-2]

    print(f"{datetime.now().strftime('%H:%M:%S')} | EMA5: {last['EMA_5']:.2f}, EMA10: {last['EMA_10']:.2f}")

    if prev['EMA_5'] < prev['EMA_10'] and last['EMA_5'] > last['EMA_10']:
        print("Confirmed BUY crossover.")
        return "BUY"
    elif prev['EMA_5'] > prev['EMA_10'] and last['EMA_5'] < last['EMA_10']:
        print("Confirmed SELL crossover.")
        return "SELL"
    return None

# Place Order
def place_order(action):
    global in_position, entry_price, stoploss_price, target_price, current_position

    print(f"Placing {action} order for {SYMBOL}")
    resp = client.placeorder(
        strategy=STRATEGY_NAME,
        symbol=SYMBOL,
        exchange=EXCHANGE,
        action=action,
        price_type=PRICE_TYPE,
        product=PRODUCT,
        quantity=QUANTITY
    )
    print("Order Response:", resp)

    if resp.get("status") == "success":
        order_id = resp.get("orderid")
        time.sleep(1)
        status = client.orderstatus(order_id=order_id, strategy=STRATEGY_NAME)
        data = status.get("data", {})
        if data.get("order_status", "").lower() == "complete":
            entry_price = float(data["price"])
            stoploss_price = round(entry_price - STOPLOSS_BUFFER, 2)
            target_price = round(entry_price + TARGET_BUFFER, 2)
            current_position = action
            in_position = True
            print(f"Entry @ ₹{entry_price} | SL ₹{stoploss_price} | Target ₹{target_price}")

# Exit Order
def exit_trade():
    global in_position, exit_signal
    action = "SELL" if current_position == "BUY" else "BUY"
    print(f"Exiting trade with {action}")
    client.placeorder(
        strategy=STRATEGY_NAME,
        symbol=SYMBOL,
        exchange=EXCHANGE,
        action=action,
        price_type=PRICE_TYPE,
        product=PRODUCT,
        quantity=QUANTITY
    )
    in_position = False
    exit_signal = False

# Strategy Thread
def strategy_thread():
    global exit_signal
    while not stop_event.is_set():
        if not in_position:
            signal = get_latest_signals()
            if signal:
                place_order(signal)
        elif exit_signal:
            exit_trade()
        time.sleep(5)

# Main Execution
def main():
    print("EMA Crossover Strategy is running...")

    ws_thread = threading.Thread(target=websocket_thread)
    strat_thread = threading.Thread(target=strategy_thread)

    ws_thread.start()
    strat_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down...")
        stop_event.set()
        ws_thread.join()
        strat_thread.join()
        print("Strategy shutdown complete.")

if __name__ == "__main__":
    main()