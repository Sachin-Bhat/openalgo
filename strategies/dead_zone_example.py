"""
Dead Zone Strategy Examples for Different Symbols

This file demonstrates how to use the dead zone live trading strategy
with different NIFTY 50 and BANKNIFTY symbols and configurations.
"""

import sys
import os

# Add the current directory to Python path so we can import the strategy modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty_symbols import (
    POPULAR_SYMBOLS, 
    get_symbol_info, validate_symbol, print_available_symbols
)

def example_1_basic_usage():
    """Example 1: Basic usage with RELIANCE"""
    print("=== Example 1: Basic Dead Zone Strategy with RELIANCE ===")
    
    # This is the simplest way to run the strategy
    # Just edit the SYMBOL variable in dead_zone_live.py and run it
    
    symbol = "RELIANCE"
    if validate_symbol(symbol):
        info = get_symbol_info(symbol)
        print(f"Symbol: {symbol}")
        print(f"Exchange: {info['exchange']}")
        print(f"Is NIFTY50: {info['is_nifty50']}")
        print(f"Is BANKNIFTY: {info['is_banknifty']}")
        print("\nTo run: Edit SYMBOL = 'RELIANCE' in dead_zone_live.py and execute")
        print("python dead_zone_live.py")
    
    print("\n" + "="*60 + "\n")

def example_2_banking_stocks():
    """Example 2: Configuration for banking stocks"""
    print("=== Example 2: Dead Zone Strategy for Banking Stocks ===")
    
    banking_stocks = POPULAR_SYMBOLS["banking"]
    print("Recommended banking stocks for dead zone strategy:")
    
    for i, symbol in enumerate(banking_stocks, 1):
        info = get_symbol_info(symbol)
        print(f"{i}. {symbol} ({info['exchange']})")
    
    print("\nBanking stocks typically have:")
    print("- Higher volatility")
    print("- Good technical patterns") 
    print("- Sufficient volume for trading")
    print("\nRecommended configuration for banking stocks:")
    print("DEAD_ZONE_LOWER = -0.0015  # -0.15% (wider zone)")
    print("DEAD_ZONE_UPPER = 0.0015   # +0.15%")
    print("SIGNAL_PROBABILITY_THRESHOLD = 0.65  # Higher confidence")
    print("QUANTITY = 1  # Start small due to higher volatility")
    
    print("\n" + "="*60 + "\n")

def example_3_it_stocks():
    """Example 3: Configuration for IT stocks"""
    print("=== Example 3: Dead Zone Strategy for IT Stocks ===")
    
    it_stocks = POPULAR_SYMBOLS["it_stocks"]
    print("IT stocks suitable for dead zone strategy:")
    
    for i, symbol in enumerate(it_stocks, 1):
        print(f"{i}. {symbol}")
    
    print("\nIT stocks characteristics:")
    print("- Generally less volatile than banking")
    print("- Good for trending strategies")
    print("- Strong technical patterns")
    print("\nRecommended configuration for IT stocks:")
    print("DEAD_ZONE_LOWER = -0.0012  # -0.12%")
    print("DEAD_ZONE_UPPER = 0.0012   # +0.12%")
    print("SIGNAL_PROBABILITY_THRESHOLD = 0.62  # Moderate confidence")
    print("RETRAIN_FREQUENCY_DAYS = 5  # More frequent retraining")
    
    print("\n" + "="*60 + "\n")

def example_4_high_volume_stocks():
    """Example 4: High volume stocks for better execution"""
    print("=== Example 4: Dead Zone Strategy for High Volume Stocks ===")
    
    high_volume = POPULAR_SYMBOLS["high_volume"]
    print("High volume stocks (better for MARKET orders):")
    
    for i, symbol in enumerate(high_volume, 1):
        info = get_symbol_info(symbol)
        print(f"{i}. {symbol} - NIFTY50: {info['is_nifty50']}")
    
    print("\nAdvantages of high volume stocks:")
    print("- Better price discovery")
    print("- Lower impact cost")
    print("- More reliable MARKET order execution")
    print("- Reduced slippage")
    print("\nRecommended for beginners and MARKET order strategies")
    
    print("\n" + "="*60 + "\n")

def example_5_custom_configuration():
    """Example 5: Custom configuration template"""
    print("=== Example 5: Custom Configuration Template ===")
    
    print("To customize the strategy for any symbol, modify these parameters:")
    print()
    
    config_template = """
# Symbol Selection
SYMBOL = "TCS"  # Change to your preferred symbol
EXCHANGE = "NSE"  # Usually NSE for NIFTY50/BANKNIFTY
QUANTITY = 1  # Number of shares to trade

# Dead Zone Parameters (adjust based on symbol volatility)
DEAD_ZONE_LOWER = -0.0010  # Conservative: -0.0008, Aggressive: -0.0015
DEAD_ZONE_UPPER = 0.0010   # Conservative: 0.0008, Aggressive: 0.0015

# Model Configuration (adjust based on symbol behavior)
SIGNAL_PROBABILITY_THRESHOLD = 0.60  # Conservative: 0.65, Aggressive: 0.55
DAYS_OF_HISTORY = 100  # More data for stable stocks, less for trending
RETRAIN_FREQUENCY_DAYS = 7  # More frequent for volatile stocks

# Risk Management
PRODUCT = "CNC"  # Cash and Carry for delivery trading
PRICE_TYPE = "MARKET"  # Use MARKET for high volume stocks
"""
    
    print(config_template)
    
    print("Volatility-based recommendations:")
    print("High Volatility (Banking): Wider dead zone, higher confidence")
    print("Medium Volatility (IT): Standard settings")
    print("Low Volatility (FMCG): Narrower dead zone, lower confidence")
    
    print("\n" + "="*60 + "\n")

def example_6_risk_management():
    """Example 6: Risk management considerations"""
    print("=== Example 6: Risk Management Guidelines ===")
    
    print("1. Position Sizing Guidelines:")
    print("   - Start with QUANTITY = 1 for testing")
    print("   - Max 2-3% of portfolio per trade")
    print("   - Consider stock price for position value")
    print()
    
    print("2. Symbol-Specific Risk Factors:")
    print("   High Risk: Banking stocks (higher volatility)")
    print("   Medium Risk: IT, Auto, Pharma")
    print("   Lower Risk: FMCG, Utilities")
    print()
    
    print("3. Dead Zone Tuning:")
    print("   Wider zones (0.15%): Fewer trades, higher quality signals")
    print("   Narrower zones (0.08%): More trades, may include noise")
    print()
    
    print("4. Confidence Threshold:")
    print("   Higher threshold (70%): Fewer but higher quality trades")
    print("   Lower threshold (55%): More trades but some false signals")
    print()
    
    print("5. Market Conditions:")
    print("   - Avoid during major news events")
    print("   - Consider market volatility (VIX levels)")
    print("   - Monitor sector-specific news")
    
    print("\n" + "="*60 + "\n")

def example_7_step_by_step_setup():
    """Example 7: Step-by-step setup guide"""
    print("=== Example 7: Step-by-Step Setup Guide ===")
    
    print("Step 1: Choose Your Symbol")
    print("   - Review available symbols in nifty_symbols.py")
    print("   - Consider your risk tolerance and sector preference")
    print("   - Start with high-volume stocks for better execution")
    print()
    
    print("Step 2: Configure Strategy Parameters")
    print("   - Edit SYMBOL in dead_zone_live.py")
    print("   - Adjust dead zone thresholds based on volatility")
    print("   - Set appropriate confidence threshold")
    print()
    
    print("Step 3: Update OpenAlgo Configuration")
    print("   - Set correct API key")
    print("   - Verify host and WebSocket URLs")
    print("   - Test connection with broker")
    print()
    
    print("Step 4: Test with Paper Trading")
    print("   - Run strategy in paper trading mode first")
    print("   - Monitor signal generation and model performance")
    print("   - Verify order placement logic")
    print()
    
    print("Step 5: Monitor and Maintain")
    print("   - Watch logs for errors and performance")
    print("   - Monitor model retraining process")
    print("   - Track strategy performance metrics")
    
    print("\n" + "="*60 + "\n")

def print_symbol_recommendations():
    """Print symbol recommendations for different strategies"""
    print("=== Symbol Recommendations by Strategy Type ===")
    print()
    
    recommendations = {
        "Beginner Friendly": ["RELIANCE", "TCS", "HDFCBANK", "INFY"],
        "High Volume": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"],
        "Low Volatility": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA"],
        "Trending Stocks": ["TCS", "INFY", "HCLTECH", "WIPRO"],
        "Value Plays": ["SBIN", "ONGC", "NTPC", "POWERGRID"],
        "Growth Stocks": ["RELIANCE", "HDFCBANK", "ICICIBANK", "BAJFINANCE"]
    }
    
    for category, symbols in recommendations.items():
        print(f"{category}:")
        for symbol in symbols:
            print(f"  - {symbol}")
        print()

def main():
    """Run all examples"""
    print("Dead Zone Strategy Examples for OpenAlgo")
    print("=" * 60)
    print()
    
    # Show available symbols first
    print("Available Symbols:")
    print_available_symbols()
    print("\n" + "="*60 + "\n")
    
    # Run all examples
    example_1_basic_usage()
    example_2_banking_stocks()
    example_3_it_stocks()
    example_4_high_volume_stocks()
    example_5_custom_configuration()
    example_6_risk_management()
    example_7_step_by_step_setup()
    
    # Symbol recommendations
    print_symbol_recommendations()
    
    print("=== Quick Start Commands ===")
    print()
    print("1. For RELIANCE (default):")
    print("   python dead_zone_live.py")
    print()
    print("2. For TCS (edit SYMBOL first):")
    print("   # Edit SYMBOL = 'TCS' in dead_zone_live.py")
    print("   python dead_zone_live.py")
    print()
    print("3. For banking stocks (HDFCBANK with custom config):")
    print("   # Edit SYMBOL = 'HDFCBANK' and adjust dead zone to ±0.15%")
    print("   python dead_zone_live.py")
    print()
    print("Remember to test with paper trading first!")

if __name__ == "__main__":
    main() 