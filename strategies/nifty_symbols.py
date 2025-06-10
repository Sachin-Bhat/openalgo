# NIFTY 50 and BANKNIFTY Symbol Configuration for OpenAlgo Trading

# NIFTY 50 Stocks (as per latest composition)
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "ASIANPAINT", "HCLTECH", "AXISBANK", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "ONGC", "BAJFINANCE",
    "NESTLEIND", "WIPRO", "POWERGRID", "NTPC", "TECHM",
    "BAJAJFINSV", "COALINDIA", "HDFCLIFE", "GRASIM", "ADANIENTE",
    "EICHERMOT", "INDUSINDBK", "TATACONSUM", "TATAMOTORS", "M&M",
    "SBILIFE", "JSWSTEEL", "BAJAJ-AUTO", "BRITANNIA", "APOLLOHOSP",
    "DIVISLAB", "DRREDDY", "CIPLA", "HEROMOTOCO", "HINDALCO",
    "TATAPOWER", "BPCL", "ADANIPORTS", "TATASTEEL", "UPL"
]

# BANKNIFTY Stocks (Banking sector stocks)
BANKNIFTY_SYMBOLS = [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    "AUBANK", "RBLBANK"
]

# Index symbols (if your broker supports index trading)
INDEX_SYMBOLS = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"
]

# Exchange mapping
SYMBOL_EXCHANGE_MAP = {
    # All NIFTY 50 stocks are on NSE
    **{symbol: "NSE" for symbol in NIFTY50_SYMBOLS},
    # All BANKNIFTY stocks are on NSE
    **{symbol: "NSE" for symbol in BANKNIFTY_SYMBOLS},
    # Indices are also on NSE
    **{symbol: "NSE" for symbol in INDEX_SYMBOLS}
}

# Popular symbols for different strategies
POPULAR_SYMBOLS = {
    "high_volume": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
    "banking": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN"],
    "it_stocks": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "auto_stocks": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT"],
    "pharma_stocks": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "fmcg_stocks": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"]
}

def get_symbol_info(symbol):
    """Get exchange information for a symbol"""
    return {
        "symbol": symbol,
        "exchange": SYMBOL_EXCHANGE_MAP.get(symbol, "NSE"),
        "is_nifty50": symbol in NIFTY50_SYMBOLS,
        "is_banknifty": symbol in BANKNIFTY_SYMBOLS,
        "is_index": symbol in INDEX_SYMBOLS
    }

def validate_symbol(symbol):
    """Validate if symbol is in supported list"""
    all_symbols = NIFTY50_SYMBOLS + BANKNIFTY_SYMBOLS + INDEX_SYMBOLS
    return symbol in all_symbols

def get_symbols_by_category(category):
    """Get symbols by category"""
    if category.lower() == "nifty50":
        return NIFTY50_SYMBOLS
    elif category.lower() == "banknifty":
        return BANKNIFTY_SYMBOLS
    elif category.lower() == "index":
        return INDEX_SYMBOLS
    elif category in POPULAR_SYMBOLS:
        return POPULAR_SYMBOLS[category]
    else:
        return []

def print_available_symbols():
    """Print all available symbols"""
    print("=== NIFTY 50 Symbols ===")
    for i, symbol in enumerate(NIFTY50_SYMBOLS, 1):
        print(f"{i:2d}. {symbol}")
    
    print("\n=== BANKNIFTY Symbols ===")
    for i, symbol in enumerate(BANKNIFTY_SYMBOLS, 1):
        print(f"{i:2d}. {symbol}")
    
    print("\n=== Index Symbols ===")
    for i, symbol in enumerate(INDEX_SYMBOLS, 1):
        print(f"{i:2d}. {symbol}")
    
    print("\n=== Popular Categories ===")
    for category, symbols in POPULAR_SYMBOLS.items():
        print(f"{category}: {', '.join(symbols)}")

if __name__ == "__main__":
    print_available_symbols() 