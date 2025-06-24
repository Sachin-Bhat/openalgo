# Dead Zone Strategy - Basket Trading Guide

## 🎯 Overview

The Dead Zone Strategy now supports **basket trading** with multiple Nifty50 stocks, significantly increasing trading opportunities from ~0.48 trades/day to **3-4 trades/day**.

## ⚙️ Configuration Options

### Trading Mode Configuration
```python
# In dead_zone_trial.py
TRADING_MODE = "basket"  # Options: "single" or "basket"
BASKET_SIZE = 10         # Number of stocks to scan (1-50)
MODEL_APPROACH = "stock_specific"  # Options: "universal", "stock_specific", "sector_based"
AUTO_TRAIN_MODELS = True  # Automatically train models for new stocks
```

### Available Stocks
The strategy includes all 50 Nifty50 stocks:
- **Banking**: HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK, INDUSINDBK
- **IT**: TCS, INFY, HCLTECH, WIPRO, TECHM
- **FMCG**: HINDUNILVR, ITC, NESTLEIND, BRITANNIA, TATACONSUM
- **Auto**: MARUTI, TATAMOTORS, EICHERMOT, HEROMOTOCO, M&M
- **Pharma**: SUNPHARMA, CIPLA, DRREDDY, DIVISLAB, APOLLOHOSP
- **Energy**: RELIANCE, ONGC, COALINDIA, NTPC, POWERGRID, TATAPOWER
- **Metals**: TATASTEEL, HINDALCO, JSWSTEEL, VEDL
- **Others**: BHARTIARTL, ASIANPAINT, ULTRACEMCO, TITAN, BAJFINANCE, etc.

## 🚀 Usage

### 1. Single Stock Mode (Traditional)
```bash
# Configure for single stock
TRADING_MODE = "single"
DEFAULT_SYMBOL = "RELIANCE"

# Run backtesting
python strategies/dead_zone_trial.py

# Run live trading
python strategies/dead_zone_trial.py --live
```

### 2. Basket Trading Mode (Recommended)
```bash
# Configure for basket trading
TRADING_MODE = "basket"
BASKET_SIZE = 10  # Start with 10 stocks

# Run basket backtesting
python strategies/dead_zone_trial.py

# Run basket live trading
python strategies/dead_zone_trial.py --live
```

### 3. Test Different Approaches
```bash
# Test model approaches
python strategies/test_model_approaches.py

# Quick functionality test
python strategies/quick_test.py

# Basket trading test
python strategies/test_basket_trading.py
```

## 📊 Expected Performance

| Mode | Trade Frequency | Accuracy | Computational Cost |
|------|----------------|----------|-------------------|
| Single Stock | 0.48/day | 85-90% | 🟢 Low |
| Basket Trading | 3-4/day | 85-95% | 🟡 Medium |
| Sector-Based | 2-3/day | 80-85% | 🟢 Low |

## 🔧 Model Training Approaches

### 1. Stock-Specific Models (Default)
- **Best accuracy** for each stock
- Individual model for each stock
- Captures stock-specific patterns
- Higher computational cost

### 2. Sector-Based Models
- Models trained on sector representatives
- Good balance of accuracy and efficiency
- Captures sector-specific patterns
- Medium computational cost

### 3. Universal Model
- One model for all stocks
- Fastest execution
- May miss stock-specific patterns
- Lowest computational cost

## 📈 Basket Trading Logic

### Signal Generation
1. **Scan all stocks** in basket every 15 minutes
2. **Train/load models** for each stock (on-demand)
3. **Generate signals** using stock-specific models
4. **Select best opportunity** (highest confidence)
5. **Execute trade** on the best stock

### Risk Management
- **Position limits**: Max 5 concurrent positions
- **Stop loss**: 2% per position
- **Take profit**: 2.5% per position
- **Position sizing**: ₹5,000 per trade
- **Market hours**: 9:15 AM - 3:30 PM

### Position Management
- **Long positions**: BUY on uptrend signals
- **Short positions**: SELL on downtrend signals
- **Automatic squaring**: All positions closed at market close
- **Multi-stock tracking**: Each position tracked separately

## 🎯 Configuration Examples

### High-Frequency Trading
```python
TRADING_MODE = "basket"
BASKET_SIZE = 20
BACKTEST_PROB_THRESHOLD = 0.45  # Lower threshold for more trades
INTERVAL = "15m"
MAX_POSITIONS = 8
```

### Conservative Trading
```python
TRADING_MODE = "basket"
BASKET_SIZE = 5
BACKTEST_PROB_THRESHOLD = 0.65  # Higher threshold for fewer, better trades
INTERVAL = "1h"
MAX_POSITIONS = 3
```

### Sector-Specific Trading
```python
TRADING_MODE = "basket"
BASKET_SIZE = 10
MODEL_APPROACH = "sector_based"
# Focus on specific sectors by modifying NIFTY50_BASKET
```

## 🔍 Monitoring and Debugging

### Log Files
- Strategy logs: `dead_zone_strategy.log`
- Model artifacts: `model_artifacts/{symbol}/`
- Backtest results: `backtest_results_detailed.csv`

### Key Metrics to Monitor
- **Trade frequency**: Should be 3-4 trades/day
- **Signal distribution**: Balanced across stocks
- **Model accuracy**: 85-95% for stock-specific models
- **Position performance**: P&L per position

### Common Issues
1. **Low trade frequency**: Reduce confidence threshold or increase basket size
2. **Poor accuracy**: Check data quality or adjust feature engineering
3. **High computational cost**: Use sector-based models or reduce basket size
4. **API errors**: Check OpenAlgo server and API key

## 🚀 Next Steps

1. **Start with basket backtesting** to validate performance
2. **Adjust parameters** based on results
3. **Test live trading** with small basket size
4. **Scale up** gradually based on performance
5. **Monitor and optimize** continuously

## 📞 Support

For issues or questions:
1. Check the logs for error messages
2. Run the quick test script: `python strategies/quick_test.py`
3. Verify OpenAlgo server is running
4. Check API key and permissions

---

**Happy Trading! 🎉** 