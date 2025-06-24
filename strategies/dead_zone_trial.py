import pandas as pd
import polars as pl
import numpy as np
import pandas_ta_remake as ta
from quantmod.indicators import BBands

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from statsmodels.stats.outliers_influence import variance_inflation_factor

from xgboost import XGBClassifier
import optuna

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from openalgo import api

import threading
import signal
from datetime import datetime, timedelta
import traceback
import logging
import sys
from functools import wraps
import pickle
from collections import Counter

"""
Dead Zone ML Strategy - High-Frequency Intraday Trading with Short Selling

This strategy uses machine learning to predict market trends and execute high-frequency intraday trades.
Features:
- Three-class classification: Uptrend (1), Neutral (0), Downtrend (-1)
- High-frequency intraday trading with 15-minute intervals
- Short selling capability for downtrend predictions
- Automatic position squaring at market close
- Market hours validation (9:15 AM - 3:30 PM, Monday-Friday)
- Real-time stop loss and take profit management
- Multi-position management with risk controls
- Basket trading support for Nifty50 stocks to increase trading opportunities
- Target: 3-4 trades per day

Trading Modes:
- Single Stock Mode: Trade one specific stock (e.g., RELIANCE)
- Basket Mode: Scan multiple Nifty50 stocks and trade the best opportunities

Signal Types:
- BUY (1): Long position on uptrend prediction
- SELL (-1): Short position on downtrend prediction  
- HOLD (0): No position on neutral prediction or low confidence

Optimized for:
- 15-minute intervals for frequent signals
- Lower confidence threshold (0.5) for more trades
- Smaller dead zone (±0.05%) for sensitivity
- Tighter risk management for frequent trading
- Basket trading to increase trade frequency to 3-4 trades per day
"""

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dead_zone_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('DeadZoneStrategy')

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    stop_event.set()
    exit_all_positions()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def error_handler(func):
    """Decorator for error handling and logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise
    return wrapper

# %%
START_DATE = "2017-07-03"
END_DATE_TRAIN_VAL = "2021-12-31" # Training + Validation data ends here
END_DATE_TEST = pd.Timestamp.now().strftime("%Y-%m-%d")      # Test data ends here (1 year of backtesting)

# Basket of Nifty50 stocks for increased trading opportunities
NIFTY50_BASKET = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN", 
    "BHARTIARTL", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA", "TATAMOTORS",
    "WIPRO", "ULTRACEMCO", "TITAN", "BAJFINANCE", "NESTLEIND", "POWERGRID", "NTPC", 
    "TECHM", "BAJAJFINSV", "ADANIENT", "JSWSTEEL", "ONGC", "COALINDIA", "TATASTEEL", "HINDALCO",
    "CIPLA", "DRREDDY", "SHREECEM", "DIVISLAB", "EICHERMOT", "HEROMOTOCO", "BRITANNIA", 
    "KOTAKBANK", "LT", "ADANIPORTS", "GRASIM", "TATACONSUM", "BPCL", "UPL", "VEDL", 
    "INDUSINDBK", "SBILIFE", "HDFC", "TATAPOWER", "M&M", "APOLLOHOSP"
]

# Strategy Configuration
EXCHANGE = "NSE"
INTERVAL = "15m"  # 15-minute intervals for frequent signals
DEFAULT_SYMBOL = "RELIANCE"  # Default symbol for single-stock mode

# Trading Mode Configuration
TRADING_MODE = "basket"  # Options: "single" or "basket"
BASKET_SIZE = 10  # Number of stocks to scan in basket mode (max 50)

# Model Training Configuration
MODEL_APPROACH = "stock_specific"  # Options: "universal", "stock_specific", "sector_based"
AUTO_TRAIN_MODELS = True  # Automatically train models for new stocks

# Feature Handling Configuration
FEATURE_ALIGNMENT_METHOD = "zero"  # Options: "zero", "mean", "median", "forward_fill"
MIN_FEATURES_REQUIRED = 10  # Minimum features required for model training
MIN_FINAL_FEATURES = 5  # Minimum final features after selection
SKIP_INCOMPATIBLE_STOCKS = True  # Skip stocks with feature mismatches instead of failing

# Target variable definition
DEAD_ZONE_LOWER = -0.0005  # Reduced from -0.0010 for more sensitive signals
DEAD_ZONE_UPPER = 0.0005   # Reduced from 0.0010 for more sensitive signals

# Feature Selection Parameters
CORRELATION_THRESHOLD = 0.96
VIF_THRESHOLD = 10
N_UNIVARIATE_FEATURES = 40 # Number of features to select with SelectKBest
N_XGB_IMPORTANCE_FEATURES = 25 # Number of features to select based on initial XGBoost importance
RFECV_MIN_FEATURES = 10 # Minimum features for RFECV

# Hyperparameter Tuning
N_OPTUNA_TRIALS = 10 # Number of Optuna trials
OPTUNA_CV_SPLITS = 3 # TimeSeriesSplits for Optuna cross-validation

# Backtesting
BACKTEST_PROB_THRESHOLD = 0.50  # Lowered from 0.60 to generate more signals for 3-4 trades per day
RISK_FREE_RATE_ANNUAL = 0.08 # For Sharpe Ratio calculation

# Random Seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Live Trading Configuration
STRATEGY_NAME = "DeadZone_ML_Strategy"
PRODUCT = "MIS"  # Changed from CNC to MIS for intraday trading with short selling
PRICE_TYPE = "MARKET"

# Position and Risk Management Parameters
MAX_POSITIONS = 5  # Maximum number of concurrent positions
POSITION_VALUE = 5000  # position size
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.025  # Reduced take profit for 15m intervals

# Balance and Cash Management Parameters
MIN_BALANCE = 50000  # Minimum balance required to start trading (₹50,000)
MAX_BALANCE = 55000  # Maximum balance to trade with (₹55,000)
MIN_CASH_PER_TRADE = 4000  # Minimum cash required per trade (₹4,000)
MAX_CASH_PER_TRADE = 10000  # Maximum cash to use per trade (₹10,000)
MAX_PORTFOLIO_RISK = 0.015  # Maximum 1.5% risk per trade

# Live Trading State Variables
ltp = None
active_positions = {}  # Dictionary to track multiple positions
stop_event = threading.Event()
instrument = [{"exchange": EXCHANGE, "symbol": DEFAULT_SYMBOL}]  # Define instrument at top level

pl.config.Config.set_tbl_rows(100)

# %%
# Initialize OpenAlgo client
client = api(
    api_key="917dd42d55f63ae8f00117abfbe5b05465fc3bd76a3efbee3be7c085df0be579",
    host="http://127.0.0.1:5000",
    ws_url="ws://127.0.0.1:8765"
)

# %%
@error_handler
def fetch_historical_data(client: api, symbol, exchange, interval, start_date, end_date, chunk_size=90, max_retries=5, retry_delay=10):
    """
    Fetch historical data in chunks with retry logic and save to CSV.
    Will read from existing CSV if available and only fetch missing data.
    """
    # Create filename for the data
    filename = f"data/{symbol}_{exchange}_{interval}.csv"
    
    # Convert dates to datetime objects and ensure they are timezone-aware
    start_dt = pd.to_datetime(start_date).tz_localize('UTC+05:30')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC+05:30')
    
    # Adjust chunk size based on interval
    if interval == "1m" or interval == "5m" or interval == "15m" or interval == "30m":
        chunk_size = 20  # For 1-minute data, fetch 20 days at a time
    elif interval == "1h":
        chunk_size = 30  # For hourly data, fetch 30 days at a time
    else:  # Daily or larger intervals
        chunk_size = 90  # Default chunk size for daily data
    
    # Try to load existing data
    existing_data = None
    if os.path.exists(filename):
        print(f"Found existing data file: {filename}")
        try:
            existing_data = pd.read_csv(filename, index_col=0, parse_dates=True)
            # Ensure the index is timezone-aware
            if not existing_data.empty and existing_data.index.tz is None:
                existing_data.index = existing_data.index.tz_localize('UTC+05:30')
            print(f"Loaded {len(existing_data)} existing records")
            
            # Use safe datetime conversion for existing data
            existing_data.index = safe_datetime_conversion(existing_data.index)
            
            # Check if existing data covers the requested date range
            if not existing_data.empty:
                # Convert start_dt and end_dt to timezone-naive for comparison
                start_dt_naive = start_dt.tz_localize(None) if start_dt.tz is not None else start_dt
                end_dt_naive = end_dt.tz_localize(None) if end_dt.tz is not None else end_dt
                
                first_date = existing_data.index.min()
                last_date = existing_data.index.max()
                
                print(f"Existing data range: {first_date.date()} to {last_date.date()}")
                print(f"Requested data range: {start_dt_naive.date()} to {end_dt_naive.date()}")
                
                # Check if we already have all the requested data
                if first_date <= start_dt_naive and last_date >= end_dt_naive:
                    print("✅ All requested data already exists in file - no fetching needed!")
                    print(f"   First date in file: {first_date.date()}")
                    print(f"   Last date in file: {last_date.date()}")
                    print(f"   Requested start: {start_dt_naive.date()}")
                    print(f"   Requested end: {end_dt_naive.date()}")
                    # Filter to requested date range and return
                    filtered_data = existing_data[(existing_data.index >= start_dt_naive) & (existing_data.index <= end_dt_naive)]
                    print(f"Returning {len(filtered_data)} records from existing data")
                    return filtered_data
                else:
                    print("❌ Data not fully available - will fetch missing data")
                    if first_date > start_dt_naive:
                        print(f"   Missing: Data before {first_date.date()}")
                    if last_date < end_dt_naive:
                        print(f"   Missing: Data after {last_date.date()}")
                
                # Check if we need to fetch more recent data
                if last_date >= start_dt_naive:
                    start_dt = last_date + pd.Timedelta(minutes=1 if interval == "1m" else 1)
                    print(f"Updating start date to {start_dt} based on existing data")
                
                # Check if we need to fetch older data
                if first_date > start_dt_naive:
                    print(f"Existing data starts from {first_date.date()}, but we need from {start_dt_naive.date()}")
                    # We'll fetch the missing older data
                    
        except Exception as e:
            print(f"Error reading existing data file: {e}")
            print("Will create new data file")
    
    # If we have all the data already, return it
    # Ensure both timestamps are timezone-naive for comparison
    start_dt_naive = start_dt.tz_localize(None) if start_dt.tz is not None else start_dt
    end_dt_naive = end_dt.tz_localize(None) if end_dt.tz is not None else end_dt
    
    if existing_data is not None and start_dt_naive >= end_dt_naive:
        print("All requested data already exists in file")
        return existing_data
    
    # Calculate total days and number of chunks needed for new data
    total_days = (end_dt_naive - start_dt_naive).days
    if total_days <= 0:
        return existing_data
        
    # Calculate number of chunks based on max_days_per_request
    num_chunks = (total_days + chunk_size - 1) // chunk_size
    
    print(f"Fetching {total_days} days of new data")
    print(f"Breaking into {num_chunks} chunks of max {chunk_size} days each")
    
    # Initialize empty list to store DataFrames
    dfs = []
    chunk_dates = []  # Store the date ranges for each chunk
    
    # Process data in chunks
    for chunk in range(num_chunks):
        chunk_start = start_dt_naive + pd.Timedelta(days=chunk * chunk_size)
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_size), end_dt_naive)
        
        print(f"\nProcessing chunk {chunk + 1}/{num_chunks}")
        print(f"Period: {chunk_start.date()} to {chunk_end.date()}")
        
        # Download data for this chunk with retry logic
        chunk_success = False
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to fetch chunk data...")
                chunk_data = client.history(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    start_date=chunk_start.strftime("%Y-%m-%d"),
                    end_date=chunk_end.strftime("%Y-%m-%d"),
                )
                
                if chunk_data is not None and not chunk_data.empty:
                    print(f"Chunk data fetched successfully! Got {len(chunk_data)} records")
                    
                    # Ensure timezone consistency
                    if 'timestamp' in chunk_data.columns:
                        chunk_data['timestamp'] = pd.to_datetime(chunk_data['timestamp'])
                        if chunk_data['timestamp'].dt.tz is not None:
                            chunk_data['timestamp'] = chunk_data['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                        chunk_data.set_index('timestamp', inplace=True)
                    elif chunk_data.index.name == 'timestamp':
                        if chunk_data.index.tz is not None:
                            chunk_data.index = chunk_data.index.tz_convert('UTC').tz_localize(None)
                    
                    # Use safe datetime conversion
                    chunk_data.index = safe_datetime_conversion(chunk_data.index)
                    
                    dfs.append(chunk_data)
                    chunk_dates.append((chunk_start, chunk_end))
                    chunk_success = True
                    break
                else:
                    print("Received empty data for chunk")
                    time.sleep(retry_delay)
                    
            except Exception as e:
                print(f"Error fetching chunk data (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to fetch chunk data after {max_retries} attempts")
                    # Save what we have so far
                    if dfs:
                        print("Saving partial data...")
                        partial_data = pd.concat(dfs)
                        if existing_data is not None:
                            partial_data = pd.concat([existing_data, partial_data])
                        partial_data = partial_data[~partial_data.index.duplicated(keep='first')]
                        partial_data = partial_data.sort_index()
                        os.makedirs('data', exist_ok=True)
                        partial_data.to_csv(filename)
                        print(f"Saved {len(partial_data)} records to {filename}")
                    raise
        
        if not chunk_success:
            print(f"Failed to fetch chunk {chunk + 1} after all retries")
            continue
            
        # Save progress after each successful chunk
        if dfs:
            print("Saving progress...")
            progress_data = pd.concat(dfs)
            if existing_data is not None:
                progress_data = pd.concat([existing_data, progress_data])
            progress_data = progress_data[~progress_data.index.duplicated(keep='first')]
            progress_data = progress_data.sort_index()
            os.makedirs('data', exist_ok=True)
            progress_data.to_csv(filename)
            print(f"Saved {len(progress_data)} records to {filename}")
    
    # Combine all chunks
    if dfs:
        new_data = pd.concat(dfs)
        print("\nColumns in new data:", new_data.columns.tolist())
        
        # Remove duplicates and sort
        new_data = new_data[~new_data.index.duplicated(keep='first')]
        new_data = new_data.sort_index()
        
        print(f"\nSuccessfully combined {len(dfs)} chunks into new dataset")
        print(f"New records: {len(new_data)}")
        
        # Combine with existing data if any
        if existing_data is not None:
            data_full = pd.concat([existing_data, new_data])
            data_full = data_full[~data_full.index.duplicated(keep='first')]
            data_full = data_full.sort_index()
        else:
            data_full = new_data
            
        # Save to CSV with timestamp header
        os.makedirs('data', exist_ok=True)
        data_full.index.name = 'timestamp'  # Set the index name to 'timestamp'
        data_full.to_csv(filename)
        print(f"Saved {len(data_full)} total records to {filename}")
        
        return data_full
    else:
        print("No new data was fetched successfully")
        return existing_data if existing_data is not None else pd.DataFrame()

# %%
@error_handler
def map_class_labels(y_series):
    """
    Map class labels from [-1, 0, 1] to [0, 1, 2] for XGBoost compatibility.
    
    Args:
        y_series: Series with class labels [-1, 0, 1]
        
    Returns:
        tuple: (mapped_series, class_mapping)
    """
    # Create mapping from original labels to XGBoost-compatible labels
    original_labels = sorted(y_series.unique())
    xgboost_labels = list(range(len(original_labels)))
    class_mapping = dict(zip(original_labels, xgboost_labels))
    
    # Map the labels
    y_mapped = y_series.map(class_mapping)
    
    print(f"Class mapping: {class_mapping}")
    print(f"Original labels: {original_labels}")
    print(f"XGBoost labels: {xgboost_labels}")
    
    return y_mapped, class_mapping

@error_handler
def prepare_target_variable(data_full, dead_zone_upper, dead_zone_lower):
    """
    Prepare the target variable for the trading strategy by calculating returns and labeling trends.
    
    Parameters:
    -----------
    data_full : pandas.DataFrame
        Input dataframe containing price data
    dead_zone_upper : float
        Upper threshold for dead zone (e.g., 0.0010 for 0.10%)
    dead_zone_lower : float
        Lower threshold for dead zone (e.g., -0.0010 for -0.10%)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added target variable and cleaned data
    """
    # Drop any rows with missing values
    data_full = data_full.dropna()
    
    # Calculate daily returns
    data_full.loc[:, 'Return'] = data_full['close'].pct_change()
    
    # Target Return is the next day's return
    data_full.loc[:, 'TargetReturn'] = data_full['Return'].shift(-1)
    
    # Define target variable: 1 if uptrend, -1 if downtrend, 0 if neutral
    data_full.loc[:, 'Trend'] = np.nan
    
    # Assign labels for three-class classification
    data_full.loc[data_full['TargetReturn'] > dead_zone_upper, 'Trend'] = 1    # Uptrend
    data_full.loc[data_full['TargetReturn'] < dead_zone_lower, 'Trend'] = -1   # Downtrend
    data_full.loc[(data_full['TargetReturn'] >= dead_zone_lower) & 
                  (data_full['TargetReturn'] <= dead_zone_upper), 'Trend'] = 0  # Neutral
    
    # Drop rows with NaN values (shouldn't happen with the above logic, but just in case)
    data_full = data_full.dropna(subset=['Trend'])
    data_full.loc[:, 'Trend'] = data_full['Trend'].astype(int)
    
    # Print summary statistics
    print(f"Full data shape: {data_full.shape}")
    print(f"Trend distribution:\n{data_full['Trend'].value_counts(normalize=True)}")
    print(f"Trend counts:\n{data_full['Trend'].value_counts()}")
    
    return data_full

# Example usage:
# data_full = prepare_target_variable(data_full, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)

# %%
@error_handler
def engineer_features(data_full):
    """
    Engineer technical indicators and features for the trading strategy.
    
    Parameters:
    -----------
    data_full : pandas.DataFrame
        Input dataframe containing price data
        
    Returns:
    --------
    tuple
        (data_full, initial_features) where:
        - data_full is the DataFrame with engineered features
        - initial_features is a list of feature column names
    """
    try:
        # Validate input data
        if data_full is None:
            raise ValueError("Input DataFrame is None")
        if not isinstance(data_full, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(data_full)}")
        if data_full.empty:
            raise ValueError("Input DataFrame is empty")
        if not all(col in data_full.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Input DataFrame must contain 'open', 'high', 'low', 'close', and 'volume' columns")

        # Price-based features
        data_full['H-L'] = data_full['high'] - data_full['low']
        data_full['C-O'] = data_full['close'] - data_full['open']
        data_full['Amplitude'] = (data_full['high'] - data_full['low']) / data_full['close'].shift(1)
        data_full['Difference'] = (data_full['close'] - data_full['open']) / data_full['close'].shift(1)
        data_full['High_Low_Range'] = data_full['high'] - data_full['low']
        data_full['Open_Close_Range'] = data_full['open'] - data_full['close']

        # Bollinger Bands
        data_full['BB_lower'], data_full['BB_middle'], data_full['BB_upper'] = BBands(data_full['close'], lookback=20)
        data_full['BB_width'] = data_full['BB_upper'] - data_full['BB_lower']

        # Lagged Returns
        for lag in [1, 2, 3, 5, 10]:
            data_full[f'Return_lag{lag}'] = data_full['Return'].shift(lag)

        # Moving Averages & Differentials
        for ma_period in [10, 20, 50]:
            data_full[f'SMA{ma_period}'] = ta.sma(data_full['close'], length=ma_period)
            data_full[f'EMA{ma_period}'] = ta.ema(data_full['close'], length=ma_period)
            data_full[f'Close_vs_SMA{ma_period}'] = data_full['close'] - data_full[f'SMA{ma_period}']
            data_full[f'Close_vs_EMA{ma_period}'] = data_full['close'] - data_full[f'EMA{ma_period}']

        if 'SMA10' in data_full and 'SMA20' in data_full:
            data_full['SMA10_vs_SMA20'] = data_full['SMA10'] - data_full['SMA20']
        if 'EMA10' in data_full and 'EMA20' in data_full:
            data_full['EMA10_vs_EMA20'] = data_full['EMA10'] - data_full['EMA20']

        # Volatility Indicators
        data_full['ATR14'] = ta.atr(data_full['high'], data_full['low'], data_full['close'], length=14)
        data_full['StdDev20_Return'] = data_full['Return'].rolling(window=20).std()

        # Momentum Indicators
        data_full['RSI14'] = ta.rsi(data_full['close'], length=14)
        
        # MACD calculation with error handling
        try:
            macd_df = ta.macd(data_full['close'], fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                data_full['MACD'] = macd_df['MACD_12_26_9']
                data_full['MACD_signal'] = macd_df['MACDs_12_26_9']
                data_full['MACD_hist'] = macd_df['MACDh_12_26_9']
            else:
                logger.warning("MACD calculation returned None or empty DataFrame. Skipping MACD features.")
        except Exception as e:
            logger.warning(f"Error calculating MACD: {str(e)}. Skipping MACD features.")
        
        data_full['Momentum10'] = data_full['close'] - data_full['close'].shift(10)
        data_full['Williams_%R'] = -100 * (data_full['high'] - data_full['close']) / (data_full['high'] - data_full['low'])
        data_full['Williams%R14'] = ta.willr(data_full['high'], data_full['low'], data_full['close'], length=14)

        # Stochastic Oscillator
        try:
            stoch_df = ta.stoch(data_full['high'], data_full['low'], data_full['close'], k=14, d=3, smooth_k=3, mamode='sma')
            if stoch_df is not None and not stoch_df.empty:
                data_full['Stochastic_K'] = stoch_df['STOCHk_14_3_3']
                data_full['Stochastic_D'] = stoch_df['STOCHd_14_3_3']
            else:
                logger.warning("Stochastic calculation returned None or empty DataFrame. Skipping Stochastic features.")
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {str(e)}. Skipping Stochastic features.")

        # Rate of Change
        data_full['ROC10'] = data_full['close'].pct_change(periods=10)

        # On-Balance Volume
        data_full['OBV'] = ta.obv(data_full['close'], data_full['volume'])

        # Volume-based features
        data_full['Volume_MA5'] = ta.sma(data_full['volume'], length=5)
        data_full['Volume_MA20'] = ta.sma(data_full['volume'], length=20)
        data_full['Volume_Change'] = data_full['volume'].pct_change()
        if 'Volume_MA5' in data_full and 'Volume_MA20' in data_full:
            data_full['Volume_MA5_vs_MA20'] = data_full['Volume_MA5'] - data_full['Volume_MA20']

        # Date/Time Features
        # Ensure index is datetime and handle dayofweek properly
        if not isinstance(data_full.index, pd.DatetimeIndex):
            # Handle timezone-aware datetime conversion properly
            if hasattr(data_full.index, 'tz') and data_full.index.tz is not None:
                # If index is already timezone-aware, convert to UTC first
                data_full.index = data_full.index.tz_convert('UTC').tz_localize(None)
            else:
                data_full.index = pd.to_datetime(data_full.index)
        
        # Use safe datetime conversion
        data_full.index = safe_datetime_conversion(data_full.index)
        
        data_full['DayOfWeek'] = data_full.index.dayofweek  # Monday=0, Sunday=6
        data_full['Month'] = data_full.index.month

        # Ensure all features are numerical
        for col in data_full.columns:
            if data_full[col].dtype == 'object':
                try:
                    data_full[col] = pd.to_numeric(data_full[col])
                except Exception as e:
                    logger.warning(f"Error converting column {col} to numeric: {str(e)}. Dropping it.")
                    data_full = data_full.drop(columns=[col])

        # Drop rows with NaNs created by indicators/lags
        initial_features = data_full.columns.drop(['open', 'high', 'low', 'close', 'volume', 'Return', 'Trend'])
        logger.info(f"NaN counts before dropping:\n{data_full.isnull().sum()}")
        data_full.dropna(subset=initial_features, inplace=True)
        logger.info(f"Data shape after feature engineering and NaN drop: {data_full.shape}")
        logger.info(f"Number of initial features: {len(initial_features)}")
        
        return data_full, initial_features
    except Exception as e:
        logger.error(f"Error in engineer_features: {str(e)}\n{traceback.format_exc()}")
        raise

# %%
@error_handler
def split_data(data_full, initial_features, end_date_train_val):
    """
    Split data into training, validation and test sets.
    
    Args:
        data_full (pd.DataFrame): Full dataset with features and target
        initial_features (list): List of feature column names
        end_date_train_val (str): End date for train/validation split
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, class_mapping)
    """
    # Define split points and ensure timezone consistency
    train_val_end_date = pd.to_datetime(end_date_train_val).tz_localize(data_full.index.tz)
    test_start_date = train_val_end_date + pd.Timedelta(days=1)
    
    # For train/validation split, use approximately the last year of train_val data for validation
    train_end_date = train_val_end_date - pd.DateOffset(years=1)
    
    # Split data using timezone-aware comparisons
    train_df = data_full[data_full.index <= train_end_date]
    val_df = data_full[(data_full.index > train_end_date) & (data_full.index <= train_val_end_date)]
    test_df = data_full[data_full.index >= test_start_date]
    
    # Validate that we have sufficient data
    print("Data split validation:")
    print(f"  Total data: {len(data_full)} records")
    print(f"  Training data: {len(train_df)} records")
    print(f"  Validation data: {len(val_df)} records")
    print(f"  Test data: {len(test_df)} records")
    
    # Check if we have enough data for training
    if len(train_df) < 100:
        print(f"Warning: Insufficient training data ({len(train_df)} < 100). Adjusting split...")
        # Use 70% for training, 15% for validation, 15% for test
        total_len = len(data_full)
        train_end_idx = int(total_len * 0.7)
        val_end_idx = int(total_len * 0.85)
        
        train_df = data_full.iloc[:train_end_idx]
        val_df = data_full.iloc[train_end_idx:val_end_idx]
        test_df = data_full.iloc[val_end_idx:]
        
        print("  Adjusted split:")
        print(f"    Training data: {len(train_df)} records")
        print(f"    Validation data: {len(val_df)} records")
        print(f"    Test data: {len(test_df)} records")
    
    # Final validation
    if len(train_df) == 0:
        raise ValueError("No training data available. Check date ranges and data availability.")
    
    if len(val_df) == 0:
        print("Warning: No validation data. Using training data for validation.")
        val_df = train_df.copy()
    
    if len(test_df) == 0:
        print("Warning: No test data. Using validation data for testing.")
        test_df = val_df.copy()
    
    # Prepare features and target
    X = data_full[initial_features]
    y = data_full['Trend']
    
    # Map class labels for XGBoost compatibility
    y_mapped, class_mapping = map_class_labels(y)
    
    # Split features and target
    X_train, y_train = train_df[initial_features], y_mapped[train_df.index]
    X_val, y_val = val_df[initial_features], y_mapped[val_df.index]
    X_test, y_test = test_df[initial_features], y_mapped[test_df.index]
    
    # Print shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Print sample data
    print(X_train.head())
    print(y_train.head())
    print(X_val.head())
    print(y_val.head())
    print(X_test.head())
    print(y_test.head())
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, class_mapping

# %%
def validate_features(X):
    """Validate features before scaling"""
    # Check for infinite values
    if np.any(np.isinf(X)):
        print("Warning: Infinite values found in features")
        X = np.nan_to_num(X, nan=np.nan, posinf=0, neginf=0)
    
    # Check for NaN values
    if np.any(np.isnan(X)):
        print("Warning: NaN values found in features")
        # Replace NaN with column means
        col_means = np.nanmean(X, axis=0)
        X = np.nan_to_num(X, nan=col_means)
    
    return X

# %%
@error_handler
def scale_features(X_train, X_val, X_test):
    """
    Clean, validate and scale features using StandardScaler.
    Ensures scaler is always fit on DataFrame with feature names to prevent warnings.
    
    Args:
        X_train: Training features DataFrame
        X_val: Validation features DataFrame
        X_test: Test features DataFrame
        
    Returns:
        tuple: (X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler)
    """
    # Ensure all inputs are DataFrames with feature names
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame")
    if not isinstance(X_val, pd.DataFrame):
        raise ValueError("X_val must be a pandas DataFrame")
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame")
    
    # Verify all DataFrames have the same columns
    if not (X_train.columns.equals(X_val.columns) and X_val.columns.equals(X_test.columns)):
        raise ValueError("All feature DataFrames must have the same columns")
    
    # Clean and validate the data
    X_train_clean = validate_features(X_train)
    X_val_clean = validate_features(X_val)
    X_test_clean = validate_features(X_test)

    # Convert back to DataFrames with original column names
    X_train_clean_df = pd.DataFrame(X_train_clean, columns=X_train.columns, index=X_train.index)
    X_val_clean_df = pd.DataFrame(X_val_clean, columns=X_val.columns, index=X_val.index)
    X_test_clean_df = pd.DataFrame(X_test_clean, columns=X_test.columns, index=X_test.index)

    # Scale the features - fit on DataFrame to preserve feature names
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean_df)
    X_val_scaled = scaler.transform(X_val_clean_df)
    X_test_scaled = scaler.transform(X_test_clean_df)

    print(f"Training data shape after scaling: {X_train_scaled.shape}")
    print(f"Validation data shape after scaling: {X_val_scaled.shape}")
    print(f"Test data shape after scaling: {X_test_scaled.shape}")

    # Convert scaled arrays back to DataFrames with original column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Verify scaler has feature names
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Scaler feature names: {scaler.feature_names_in_}")
    else:
        print("Warning: Scaler does not have feature names")
        # Set feature names manually
        scaler.feature_names_in_ = np.array(X_train.columns)

    print(X_train_scaled_df.head())
    print(X_val_scaled_df.head())
    print(X_test_scaled_df.head())
    
    return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler

# %%
@error_handler
def select_features(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train):
    """
    Perform feature selection using a funnel approach with multiple methods.
    
    Args:
        X_train_scaled_df: Scaled training features DataFrame
        X_val_scaled_df: Scaled validation features DataFrame
        X_test_scaled_df: Scaled test features DataFrame
        y_train: Training target variable
        
    Returns:
        tuple: (X_train_final, X_val_final, X_test_final, final_selected_features)
    """
    selected_features = list(X_train_scaled_df.columns)
    print(f"Starting with {len(selected_features)} features")

    # Step 5.1: Correlation Filter
    print("\nStep 5.1: Correlation Filter")
    corr_matrix = X_train_scaled_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > CORRELATION_THRESHOLD)]
    X_train_scaled_df = X_train_scaled_df.drop(columns=to_drop_corr)
    X_val_scaled_df = X_val_scaled_df.drop(columns=to_drop_corr)
    X_test_scaled_df = X_test_scaled_df.drop(columns=to_drop_corr)
    selected_features = list(X_train_scaled_df.columns)
    print(f"Dropped due to high correlation: {to_drop_corr}")
    print(f"Features remaining: {len(selected_features)}")

    # Step 5.2: VIF Filter
    print("\nStep 5.2: VIF Filter")
    features_for_vif = list(X_train_scaled_df.columns)
    final_vif_features = []
    dropped_vif_count = 0
    
    while True:
        if not features_for_vif:
            break
            
        # Calculate VIF for remaining features
        vif = pd.DataFrame()
        vif["feature"] = features_for_vif
        
        # Handle potential division by zero in VIF calculation
        try:
            vif["VIF"] = [variance_inflation_factor(X_train_scaled_df[features_for_vif].values, i) 
                          for i in range(len(features_for_vif))]
        except RuntimeWarning:
            # If division by zero occurs, set VIF to a high value to ensure feature is dropped
            vif["VIF"] = [float('inf') if np.isinf(v) else v 
                          for v in [variance_inflation_factor(X_train_scaled_df[features_for_vif].values, i) 
                                  for i in range(len(features_for_vif))]]

        max_vif = vif['VIF'].max()
        if max_vif > VIF_THRESHOLD:
            feature_to_drop = vif.sort_values('VIF', ascending=False)['feature'].iloc[0]
            features_for_vif.remove(feature_to_drop)
            dropped_vif_count += 1
        else:
            final_vif_features = list(features_for_vif)
            break
            
    if dropped_vif_count > 0:
        print(f"Dropped {dropped_vif_count} features due to VIF > {VIF_THRESHOLD}")
        
    # Update DataFrames with final VIF features
    X_train_scaled_df = X_train_scaled_df[final_vif_features]
    X_val_scaled_df = X_val_scaled_df[final_vif_features]
    X_test_scaled_df = X_test_scaled_df[final_vif_features]
    selected_features = list(X_train_scaled_df.columns)
    print(f"Features remaining after VIF: {len(selected_features)}")

    # Step 5.3: Univariate Filter
    print("\nStep 5.3: Univariate Filter (SelectKBest)")
    if len(selected_features) > N_UNIVARIATE_FEATURES:
        # Use f_classif for multi-class classification
        selector_kbest = SelectKBest(score_func=f_classif, k=N_UNIVARIATE_FEATURES)
        selector_kbest.fit(X_train_scaled_df, y_train)
        kbest_features = X_train_scaled_df.columns[selector_kbest.get_support()]

        X_train_scaled_df = X_train_scaled_df[kbest_features]
        X_val_scaled_df = X_val_scaled_df[kbest_features]
        X_test_scaled_df = X_test_scaled_df[kbest_features]
        selected_features = list(kbest_features)
        print(f"Selected {len(selected_features)} features with SelectKBest.")
    else:
        print("Skipping SelectKBest as current number of features is less than or equal to N_UNIVARIATE_FEATURES.")

    # Step 5.4: XGBoost Feature Importance
    print("\nStep 5.4: Embedded Filter (XGBoost Feature Importance)")
    if len(selected_features) > N_XGB_IMPORTANCE_FEATURES:
        # Use multi-class XGBoost
        temp_xgb = XGBClassifier(
            random_state=RANDOM_SEED, 
            eval_metric='mlogloss',  # Multi-class log loss
            objective='multi:softprob'  # Multi-class objective
        )
        temp_xgb.fit(X_train_scaled_df, y_train)
        importances = pd.Series(temp_xgb.feature_importances_, index=X_train_scaled_df.columns)
        xgb_selected_features = importances.nlargest(N_XGB_IMPORTANCE_FEATURES).index.tolist()

        X_train_scaled_df = X_train_scaled_df[xgb_selected_features]
        X_val_scaled_df = X_val_scaled_df[xgb_selected_features]
        X_test_scaled_df = X_test_scaled_df[xgb_selected_features]
        selected_features = list(xgb_selected_features)
        print(f"Selected {len(selected_features)} features based on XGBoost importance.")
    else:
        print("Skipping XGBoost importance filter as current number of features is less than or equal to N_XGB_IMPORTANCE_FEATURES.")

    # Step 5.5: RFECV
    print("\nStep 5.5: Wrapper Filter (RFECV)")
    # For multi-class, we don't need scale_pos_weight

    if len(selected_features) > RFECV_MIN_FEATURES:
        estimator_rfecv = XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric='mlogloss',  # Multi-class log loss
            objective='multi:softprob'  # Multi-class objective
        )
        cv_splitter = TimeSeriesSplit(n_splits=OPTUNA_CV_SPLITS)
        
        rfecv_selector = RFECV(
            estimator=estimator_rfecv,
            step=1,
            cv=cv_splitter,
            scoring='roc_auc_ovr',  # One-vs-rest ROC AUC for multi-class
            min_features_to_select=RFECV_MIN_FEATURES,
            n_jobs=-1
        )
        print("Fitting RFECV... (this might take a while)")
        rfecv_selector.fit(X_train_scaled_df, y_train)

        final_selected_features_mask = rfecv_selector.support_
        final_selected_features = X_train_scaled_df.columns[final_selected_features_mask].tolist()

        X_train_final = X_train_scaled_df[final_selected_features]
        X_val_final = X_val_scaled_df[final_selected_features]
        X_test_final = X_test_scaled_df[final_selected_features]
        print(f"Selected {len(final_selected_features)} features with RFECV.")
    else:
        X_train_final = X_train_scaled_df.copy()
        X_val_final = X_val_scaled_df.copy()
        X_test_final = X_test_scaled_df.copy()
        final_selected_features = selected_features
        print(f"Skipping RFECV. Using current {len(final_selected_features)} features.")
    print(f"Final selected features: {final_selected_features}")
    return X_train_final, X_val_final, X_test_final, final_selected_features


# %%
@error_handler
def tune_and_train_model(X_train_final, X_val_final, y_train, y_val, random_seed, n_optuna_trials, optuna_cv_splits):
    """
    Tune hyperparameters using Optuna and train final model on combined train+val data.
    
    Args:
        X_train_final (pd.DataFrame): Training features after feature selection
        X_val_final (pd.DataFrame): Validation features after feature selection
        y_train (pd.Series): Training labels
        y_val (pd.Series): Validation labels
        random_seed (int): Random seed for reproducibility
        n_optuna_trials (int): Number of Optuna trials
        optuna_cv_splits (int): Number of CV splits for time series cross-validation
        
    Returns:
        tuple: (tuned_model, best_params)
    """
    def objective(trial):
        params = {
            'objective': 'multi:softprob',  # Multi-class objective
            'eval_metric': 'mlogloss',  # Multi-class log loss
            'random_state': random_seed,
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
        }

        model = XGBClassifier(**params)
        model.early_stopping_rounds = 15
        tscv = TimeSeriesSplit(n_splits=optuna_cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X_train_final):
            X_cv_train, X_cv_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_cv_train, y_cv_train,
                      eval_set=[(X_cv_val, y_cv_val)],
                      verbose=False)
            preds_proba = model.predict_proba(X_cv_val)
            # Use one-vs-rest ROC AUC for multi-class
            scores.append(roc_auc_score(y_cv_val, preds_proba, multi_class='ovr', average='weighted'))
        return np.mean(scores)

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_optuna_trials, n_jobs=1)
    best_params = study.best_params
    print("Best hyperparameters found by Optuna:", best_params)

    # Train final model on combined train+val data
    X_train_val_final = pd.concat([X_train_final, X_val_final])
    y_train_val = pd.concat([y_train, y_val])

    tuned_model = XGBClassifier(
        **best_params,
        objective='multi:softprob',  # Multi-class objective
        eval_metric='mlogloss',  # Multi-class log loss
        random_state=random_seed
    )
    tuned_model.fit(X_train_val_final, y_train_val)
    print("Tuned model trained on X_train_val_final and y_train_val.")
    
    return tuned_model, best_params

# %%
@error_handler
def evaluate_model(tuned_model, X_test_final, y_test, class_mapping, display_plot=True):
    """
    Evaluate the model on test data and optionally display confusion matrix plot.
    
    Args:
        tuned_model: Trained XGBoost model
        X_test_final: Test features
        y_test: Test labels (XGBoost-compatible: 0, 1, 2)
        class_mapping: Dictionary mapping XGBoost labels to original labels
        display_plot: Boolean to control whether to display confusion matrix plot
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred_test_proba = tuned_model.predict_proba(X_test_final)
    y_pred_test_labels = np.argmax(y_pred_test_proba, axis=1)
    
    # Map XGBoost predictions back to original labels
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    y_pred_test_labels_mapped = [reverse_mapping[label] for label in y_pred_test_labels]
    y_test_mapped = [reverse_mapping[label] for label in y_test]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_test_proba, multi_class='ovr', average='weighted')
    accuracy = accuracy_score(y_test_mapped, y_pred_test_labels_mapped)
    
    print(f"Test Set ROC AUC Score: {roc_auc:.4f}")
    print(f"Test Set Accuracy Score: {accuracy:.4f}")
    
    # Confusion matrix
    unique_classes = sorted(class_mapping.keys())
    cm = confusion_matrix(y_test_mapped, y_pred_test_labels_mapped, labels=unique_classes)
    
    print("\nTest Set Confusion Matrix:")
    class_labels = ['Downtrend (-1)', 'Neutral (0)', 'Uptrend (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_labels, 
               yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Set Confusion Matrix (Multi-Class)')
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    if display_plot:
        plt.show()
    else:
        plt.close()
    
    # Classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_test_mapped, y_pred_test_labels_mapped, 
                              target_names=class_labels))
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': y_pred_test_labels_mapped,
        'probabilities': y_pred_test_proba,
        'class_mapping': class_mapping
    }

# %%
@error_handler
def run_backtest(test_df, y_pred_test_proba, symbol, class_mapping, backtest_prob_threshold=0.5, risk_free_rate_annual=0.05, 
                display_plot=True, save_dir="."):
    """
    Run backtesting analysis on trading strategy predictions and compare with buy-and-hold strategy.
    
    Parameters
    ----------
    test_df : pandas.DataFrame
        DataFrame containing test data with columns ['Open', 'Adj Close', 'Return']
    y_pred_test_proba : numpy.ndarray
        Array of prediction probabilities for the test period (multi-class)
    symbol : str
        Trading symbol/stock name for labeling
    class_mapping : dict
        Dictionary mapping XGBoost labels to original labels
    backtest_prob_threshold : float, optional
        Probability threshold for generating trading signals, by default 0.5
    risk_free_rate_annual : float, optional
        Annual risk-free rate for Sharpe ratio calculation, by default 0.05
    display_plot : bool, optional
        Whether to display the plot, by default True
    save_dir : str, optional
        Directory to save output files, by default "."
    
    Returns
    -------
    dict
        Dictionary containing backtest results and metrics
    """
    # Align predictions with test data dates
    backtest_df = test_df[['open', 'close', 'Return']].copy()
    
    # For multi-class, we need to determine the predicted class and confidence
    # Map XGBoost predictions back to original labels
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    predicted_classes = np.argmax(y_pred_test_proba, axis=1)
    predicted_labels = [reverse_mapping[label] for label in predicted_classes]
    max_probabilities = np.max(y_pred_test_proba, axis=1)
    
    backtest_df['Predicted_Trend'] = predicted_labels
    backtest_df['Signal_Confidence'] = max_probabilities
    
    # Create trading signals based on predicted trend and confidence
    # Only trade when confidence is above threshold
    backtest_df['Signal'] = 0  # Default to no position
    backtest_df.loc[(backtest_df['Predicted_Trend'] == 1) & 
                    (backtest_df['Signal_Confidence'] > backtest_prob_threshold), 'Signal'] = 1  # Buy signal (Long)
    backtest_df.loc[(backtest_df['Predicted_Trend'] == -1) & 
                    (backtest_df['Signal_Confidence'] > backtest_prob_threshold), 'Signal'] = -1  # Sell signal (Short)
    # Neutral (0) and low confidence predictions result in no position (0)

    # Calculate strategy returns for intraday trading
    backtest_df['Position'] = backtest_df['Signal'].shift(1).fillna(0)
    # For intraday trading, both long and short positions can be taken
    backtest_df['Strategy_Return'] = backtest_df['Position'] * backtest_df['Return']
    backtest_df['BH_Return'] = backtest_df['Return']

    # Cumulative Returns
    backtest_df['Strategy_Cumulative_Return'] = (1 + backtest_df['Strategy_Return']).cumprod() - 1
    backtest_df['BH_Cumulative_Return'] = (1 + backtest_df['BH_Return']).cumprod() - 1

    # Performance Metrics
    days_in_year = 252
    strategy_total_return = backtest_df['Strategy_Cumulative_Return'].iloc[-1]
    bh_total_return = backtest_df['BH_Cumulative_Return'].iloc[-1]

    strategy_annual_return = (1 + strategy_total_return)**(days_in_year / len(backtest_df)) - 1
    bh_annual_return = (1 + bh_total_return)**(days_in_year / len(backtest_df)) - 1

    strategy_annual_vol = backtest_df['Strategy_Return'].std() * np.sqrt(days_in_year)
    bh_annual_vol = backtest_df['BH_Return'].std() * np.sqrt(days_in_year)

    strategy_sharpe = (strategy_annual_return - risk_free_rate_annual) / strategy_annual_vol if strategy_annual_vol != 0 else 0
    bh_sharpe = (bh_annual_return - risk_free_rate_annual) / bh_annual_vol if bh_annual_vol != 0 else 0

    def calculate_mdd(cumulative_returns_series):
        """Calculate Maximum Drawdown from a series of cumulative returns."""
        wealth_index = 1 + cumulative_returns_series
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    strategy_mdd = calculate_mdd(backtest_df['Strategy_Cumulative_Return'])
    bh_mdd = calculate_mdd(backtest_df['BH_Cumulative_Return'])

    # Count trades (position changes)
    num_trades = backtest_df['Position'].diff().abs().sum() / 2

    # Calculate average trades per day
    trading_days = len(backtest_df)
    avg_trades_per_day = num_trades / trading_days if trading_days > 0 else 0

    # Win rate calculation for both long and short positions
    long_positions = backtest_df[backtest_df['Position'] == 1]
    short_positions = backtest_df[backtest_df['Position'] == -1]
    
    if len(long_positions) > 0:
        long_win_rate = (long_positions['Strategy_Return'] > 0).sum() / len(long_positions)
    else:
        long_win_rate = 0
        
    if len(short_positions) > 0:
        short_win_rate = (short_positions['Strategy_Return'] > 0).sum() / len(short_positions)
    else:
        short_win_rate = 0
    
    # Overall win rate
    all_positions = backtest_df[backtest_df['Position'] != 0]
    if len(all_positions) > 0:
        win_rate_days = (all_positions['Strategy_Return'] > 0).sum() / len(all_positions)
    else:
        win_rate_days = 0

    # Print results
    # Create a polars DataFrame for the results
    results_df = pl.DataFrame({
        "Metric": [
            "Cumulative Return",
            "Annualized Return", 
            "Annualized Volatility",
            f"Sharpe Ratio (Rf={risk_free_rate_annual*100}%)",
            "Maximum Drawdown (MDD)",
            "Number of Trades (approx)",
            "Average Trades per Day",
            "Overall Win Rate",
            "Long Position Win Rate",
            "Short Position Win Rate",
            "Signal Distribution"
        ],
        "Trading Strategy": [
            f"{strategy_total_return:.2%}",
            f"{strategy_annual_return:.2%}",
            f"{strategy_annual_vol:.2%}",
            f"{strategy_sharpe:.2f}",
            f"{strategy_mdd:.2%}",
            f"{num_trades:.0f}",
            f"{avg_trades_per_day:.2f}",
            f"{win_rate_days:.2%}",
            f"{long_win_rate:.2%}",
            f"{short_win_rate:.2%}",
            f"Long: {(backtest_df['Signal'] == 1).sum()}, Short: {(backtest_df['Signal'] == -1).sum()}, Hold: {(backtest_df['Signal'] == 0).sum()}"
        ],
        f"Buy-and-Hold {symbol}": [
            f"{bh_total_return:.2%}",
            f"{bh_annual_return:.2%}",
            f"{bh_annual_vol:.2%}",
            f"{bh_sharpe:.2f}",
            f"{bh_mdd:.2%}",
            "1",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ]
    })

    print(f"\nBacktesting Results (Test Period: {test_df.index.min().date()} to {test_df.index.max().date()}):")
    print(f"Trading Signal Confidence Threshold: {backtest_prob_threshold}")
    
    # Add prominent summary of key metrics
    print(f"\n{'='*60}")
    print("📊 KEY PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"🎯 Annualized Return: {strategy_annual_return:.2%}")
    print(f"📈 Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"📉 Maximum Drawdown: {strategy_mdd:.2%}")
    print(f"🔄 Total Trades: {num_trades:.0f}")
    print(f"📅 Avg Trades/Day: {avg_trades_per_day:.2f}")
    print(f"✅ Win Rate: {win_rate_days:.2%}")
    print(f"{'='*60}")
    
    print(results_df.head(11))
    
    # Print detailed signal distribution
    long_signals = (backtest_df['Signal'] == 1).sum()
    short_signals = (backtest_df['Signal'] == -1).sum()
    hold_signals = (backtest_df['Signal'] == 0).sum()
    total_signals = len(backtest_df)
    
    print("\nDetailed Signal Distribution:")
    print(f"Long Signals: {long_signals} ({long_signals/total_signals*100:.1f}%)")
    print(f"Short Signals: {short_signals} ({short_signals/total_signals*100:.1f}%)")
    print(f"Hold Signals: {hold_signals} ({hold_signals/total_signals*100:.1f}%)")
    print(f"Total Periods: {total_signals}")
    print(f"Average Trades per Day: {avg_trades_per_day:.2f}")

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    backtest_df['Strategy_Cumulative_Return'].plot(label='Trading Strategy')
    backtest_df['BH_Cumulative_Return'].plot(label=f'Buy-and-Hold {symbol}')
    plt.title(f'{symbol} Backtest: Cumulative Returns (Multi-Class)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(save_dir, "cumulative_returns_backtest.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if display_plot:
        plt.show()
    else:
        plt.close()

    # Save detailed results
    backtest_df.to_csv(os.path.join(save_dir, "backtest_results_detailed.csv"))

    # Return results as dictionary
    results = {
        'strategy_total_return': strategy_total_return,
        'bh_total_return': bh_total_return,
        'strategy_annual_return': strategy_annual_return,
        'bh_annual_return': bh_annual_return,
        'strategy_annual_vol': strategy_annual_vol,
        'bh_annual_vol': bh_annual_vol,
        'strategy_sharpe': strategy_sharpe,
        'bh_sharpe': bh_sharpe,
        'strategy_mdd': strategy_mdd,
        'bh_mdd': bh_mdd,
        'num_trades': num_trades,
        'avg_trades_per_day': avg_trades_per_day,
        'overall_win_rate': win_rate_days,
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'backtest_df': backtest_df,
        'plot_path': plot_path,
        'class_mapping': class_mapping
    }

    return results

# %%
def on_data_received(data):
    """WebSocket LTP Handler"""
    global ltp
    print(f"Received WebSocket data: {data}")  # Debug log
    if data.get("type") == "market_data" and data.get("symbol") == DEFAULT_SYMBOL:
        ltp = float(data["data"]["ltp"])
        print(f"LTP Update {EXCHANGE}:{DEFAULT_SYMBOL} => ₹{ltp}")
        
        # Check stop loss and take profit for all active positions
        for position_id, position in list(active_positions.items()):
            if position['action'] == "BUY":  # Long position
                if ltp <= position['stoploss']:
                    print(f"Stop Loss Triggered for LONG position {position_id}:")
                    print(f"Entry: ₹{position['entry_price']} | Current: ₹{ltp}")
                    print(f"Loss: ₹{round((ltp - position['entry_price']) * position['quantity'], 2)}")
                    exit_position(position_id)
                elif ltp >= position['target']:
                    print(f"Target Hit for LONG position {position_id}:")
                    print(f"Entry: ₹{position['entry_price']} | Current: ₹{ltp}")
                    print(f"Profit: ₹{round((ltp - position['entry_price']) * position['quantity'], 2)}")
                    exit_position(position_id)
            elif position['action'] == "SELL":  # Short position
                if ltp >= position['stoploss']:
                    print(f"Stop Loss Triggered for SHORT position {position_id}:")
                    print(f"Entry: ₹{position['entry_price']} | Current: ₹{ltp}")
                    print(f"Loss: ₹{round((position['entry_price'] - ltp) * position['quantity'], 2)}")
                    exit_position(position_id)
                elif ltp <= position['target']:
                    print(f"Target Hit for SHORT position {position_id}:")
                    print(f"Entry: ₹{position['entry_price']} | Current: ₹{ltp}")
                    print(f"Profit: ₹{round((position['entry_price'] - ltp) * position['quantity'], 2)}")
                    exit_position(position_id)

def websocket_thread():
    """WebSocket Thread for live price updates"""
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    reconnect_delay = 5  # seconds
    
    while not stop_event.is_set():
        try:
            print("Connecting to WebSocket...")
            client.connect()
            print("WebSocket connected successfully")
            
            print(f"Subscribing to LTP for {instrument}")
            client.subscribe_ltp(instrument, on_data_received=on_data_received)
            print("WebSocket LTP thread started.")
            
            # Reset reconnect attempts on successful connection
            reconnect_attempts = 0
            
            # Keep the connection alive
            while not stop_event.is_set():
                time.sleep(1)
                
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
            reconnect_attempts += 1
            
            if reconnect_attempts >= max_reconnect_attempts:
                print(f"Failed to connect after {max_reconnect_attempts} attempts. Stopping WebSocket thread.")
                break
                
            print(f"Attempting to reconnect in {reconnect_delay} seconds... (Attempt {reconnect_attempts}/{max_reconnect_attempts})")
            time.sleep(reconnect_delay)
            
    print("Shutting down WebSocket...")
    try:
        client.unsubscribe_ltp(instrument)
        client.disconnect()
    except Exception as e:
        print(f"Error cleaning up WebSocket connection: {e}")
    print("WebSocket connection closed.")

def calculate_position_size():
    """Calculate position size based on risk management parameters"""
    try:
        # Get current balance using funds() method
        funds_response = client.funds()
        if funds_response.get('status') != 'success':
            print(f"Error getting funds: {funds_response.get('message', 'Unknown error')}")
            return 0
            
        available_cash = float(funds_response['data']['availablecash'])
        
        if available_cash < MIN_BALANCE:
            print(f"Insufficient balance: ₹{available_cash} < ₹{MIN_BALANCE}")
            return 0
            
        # Get current market price
        quote = client.quotes(symbol=DEFAULT_SYMBOL, exchange=EXCHANGE)
        if quote.get('status') != 'success':
            print(f"Error getting quote: {quote.get('message', 'Unknown error')}")
            return 0
            
        current_price = float(quote['data']['ltp'])
        
        # Calculate number of shares based on fixed position value
        quantity = int(POSITION_VALUE / current_price)
        
        # Ensure we're not exceeding available cash
        if quantity * current_price > available_cash:
            quantity = int(available_cash / current_price)
            
        print(f"Calculated position size: {quantity} shares at ₹{current_price} (Total: ₹{quantity * current_price})")
        return quantity
        
    except Exception as e:
        print(f"Error calculating position size: {e}")
        return 0

def place_live_order(action, position_id=None):
    """Place a smart order with position management for intraday trading"""
    global active_positions
    
    try:
        # Calculate position size
        quantity = calculate_position_size()
        if quantity == 0:
            print("Cannot place order: Invalid position size")
            return
            
        # Check if we can take more positions
        if len(active_positions) >= MAX_POSITIONS:
            print(f"Cannot take more positions. Current positions: {len(active_positions)}")
            return
            
        print(f"Placing {action} order for {DEFAULT_SYMBOL}")
        resp = client.placesmartorder(
            strategy=STRATEGY_NAME,
            symbol=DEFAULT_SYMBOL,
            exchange=EXCHANGE,
            action=action,
            price_type=PRICE_TYPE,
            product=PRODUCT,
            quantity=quantity,
            position_size=quantity
        )
        print("Order Response:", resp)
        
        if resp.get("status") == "success":
            order_id = resp.get("orderid")
            time.sleep(1)  # Wait for order to be processed
            
            # Get order status and execution details
            status = client.orderstatus(order_id=order_id, strategy=STRATEGY_NAME)
            data = status.get("data", {})
            
            if data.get("order_status", "").lower() == "complete":
                # Get the actual execution price from the order status
                entry_price = float(data.get("average_price", 0))
                if entry_price == 0:
                    # If average_price is not available, try to get the last traded price
                    quote = client.quotes(symbol=DEFAULT_SYMBOL, exchange=EXCHANGE)
                    if quote.get('status') == 'success':
                        entry_price = float(quote['data']['ltp'])
                    else:
                        print("Warning: Could not get execution price, using last traded price")
                        return
                
                # Calculate stop loss and target based on position type
                if action == "BUY":  # Long position
                    stoploss_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
                    target_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)
                else:  # SELL - Short position
                    stoploss_price = round(entry_price * (1 + STOP_LOSS_PCT), 2)
                    target_price = round(entry_price * (1 - TAKE_PROFIT_PCT), 2)
                
                # Store position information
                position_id = position_id or f"pos_{len(active_positions) + 1}"
                active_positions[position_id] = {
                    'action': action,
                    'entry_price': entry_price,
                    'stoploss': stoploss_price,
                    'target': target_price,
                    'quantity': quantity,
                    'order_id': order_id,
                    'total_value': round(entry_price * quantity, 2),
                    'position_type': 'LONG' if action == 'BUY' else 'SHORT'
                }
                
                print(f"Position {position_id} opened ({active_positions[position_id]['position_type']}):")
                print(f"Entry @ ₹{entry_price} | SL ₹{stoploss_price} | Target ₹{target_price}")
                print(f"Quantity: {quantity} shares | Total Value: ₹{active_positions[position_id]['total_value']}")
            else:
                print(f"Order not completed. Status: {data.get('order_status')}")
    except Exception as e:
        print(f"Error placing order: {e}")
        print(traceback.format_exc())

def exit_position(position_id):
    """Exit a specific position for intraday trading with accurate P&L calculation"""
    try:
        if position_id not in active_positions:
            print(f"Position {position_id} not found")
            return
        position = active_positions[position_id]
        # For intraday trading, we need to square off positions
        # BUY positions are squared off with SELL
        # SELL positions are squared off with BUY
        action = "SELL" if position['action'] == "BUY" else "BUY"
        print(f"Exiting {position['position_type']} position {position_id} with {action}")
        resp = client.placesmartorder(
            strategy=STRATEGY_NAME,
            symbol=position.get('symbol', DEFAULT_SYMBOL),
            exchange=EXCHANGE,
            action=action,
            price_type=PRICE_TYPE,
            product=PRODUCT,
            quantity=position['quantity'],
            position_size=position['quantity']
        )
        if resp.get("status") == "success":
            order_id = resp.get("orderid")
            # Fetch the actual exit price from order status
            exit_price = None
            if order_id:
                status = client.orderstatus(order_id=order_id, strategy=STRATEGY_NAME)
                data = status.get("data", {})
                exit_price = float(data.get("average_price", 0))
            if not exit_price or exit_price == 0:
                # fallback to ltp or entry price
                global ltp
                exit_price = ltp if ltp else position['entry_price']
            entry_price = position['entry_price']
            if position['action'] == "BUY":
                pnl = (exit_price - entry_price) * position['quantity']
            else:
                pnl = (entry_price - exit_price) * position['quantity']
            print(f"Position {position_id} closed successfully")
            print(f"Entry Price: ₹{entry_price}")
            print(f"Exit Price: ₹{exit_price}")
            print(f"P&L: ₹{round(pnl, 2)}")
            del active_positions[position_id]
        else:
            print(f"Failed to exit position {position_id}: {resp.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"Error exiting position: {e}")

def exit_all_positions():
    """Exit all active positions"""
    for position_id in list(active_positions.keys()):
        exit_position(position_id)

@error_handler
def get_live_signal():
    """Get live trading signal"""
    try:
        print("\nFetching historical data for signal generation...")
        # Fetch historical data - using simpler approach like ema_crossover_target.py
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Get 7 days of data

        df = client.history(
            symbol=DEFAULT_SYMBOL,
            exchange=EXCHANGE,
            interval=INTERVAL,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if df is None:
            print("Error: No historical data received")
            return None
            
        print(f"Received {len(df)} rows of historical data")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        
        if len(df) < 50:  # Need enough data for indicators
            print(f"Warning: Not enough data points ({len(df)} < 50 required)")
            return None
            
        print("Preparing features for prediction...")
        # Prepare features for prediction
        df = prepare_target_variable(df, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
        df, initial_features = engineer_features(df)
        
        # Log all available features
        logger.info(f"Available features after engineering: {len(df.columns)}")
        logger.info(f"Features: {df.columns.tolist()}")
        
        # Debug: Print the final_selected_features
        logger.info(f"Expected features from model: {len(final_selected_features)}")
        logger.info(f"Expected features list: {final_selected_features}")
        
        # Align features to match the model's expected feature set
        df_aligned = align_features(df, final_selected_features, fill_method=FEATURE_ALIGNMENT_METHOD)
        
        if df_aligned is None:
            print("Error: Failed to align features")
            return None
            
        # Create features DataFrame for the latest data point
        features_df = df_aligned.iloc[-1:].copy()
            
        # Verify the features DataFrame
        logger.info(f"Features DataFrame shape: {features_df.shape}")
        logger.info(f"Features DataFrame columns: {features_df.columns.tolist()}")
        
        # Log feature values for debugging
        logger.info("Feature values for prediction:")
        for feature in features_df.columns:
            print(f"{feature}: {features_df[feature].iloc[0]}")
        
        # Scale features - pass DataFrame directly to preserve feature names
        features_scaled_df = safe_scaler_transform(scaler, features_df, final_selected_features)
        
        # Get prediction (multi-class)
        prediction_proba = tuned_model.predict_proba(features_scaled_df)[0]
        predicted_class = np.argmax(prediction_proba)
        confidence = np.max(prediction_proba)
        
        # Map XGBoost prediction back to original label using class mapping
        reverse_mapping = {v: k for k, v in class_mapping['class_mapping'].items()}
        predicted_label = reverse_mapping[predicted_class]
        
        print(f"Prediction probabilities: {prediction_proba}")
        print(f"Predicted class: {predicted_label} (confidence: {confidence:.3f})")
        
        # Return signal based on prediction and confidence
        if confidence > BACKTEST_PROB_THRESHOLD:
            if predicted_label == 1:
                print("Generated BUY signal (Uptrend)")
                return 1  # Buy signal
            elif predicted_label == -1:
                print("Generated SELL signal (Downtrend)")
                return -1  # Sell signal
            else:  # predicted_label == 0
                print("Generated HOLD signal (Neutral)")
                return 0  # Hold signal
        else:
            print(f"No trading signal generated (confidence {confidence:.3f} < threshold {BACKTEST_PROB_THRESHOLD})")
            return 0  # No signal due to low confidence
            
    except Exception as e:
        logger.error(f"Error getting live trading signal: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def live_trading_thread():
    """Thread for live trading strategy - Supports both single-stock and basket trading"""
    while not stop_event.is_set():
        try:
            # Check market hours
            market_open, market_status = check_market_hours()
            
            if not market_open:
                print(f"Market Status: {market_status}")
                if market_status == "After Market Hours" and active_positions:
                    # Square off all positions at market close
                    square_off_all_positions()
                time.sleep(300)  # Check every 5 minutes
                continue
            
            # Print current positions
            print(f"\nCurrent positions: {len(active_positions)}/{MAX_POSITIONS}")
            for pos_id, pos in active_positions.items():
                symbol_info = f" ({pos.get('symbol', DEFAULT_SYMBOL)})" if TRADING_MODE == "basket" else ""
                print(f"Position {pos_id}{symbol_info}: {pos['position_type']} @ ₹{pos['entry_price']} | SL ₹{pos['stoploss']} | Target ₹{pos['target']}")
            
            # Generate signal based on trading mode
            if TRADING_MODE == "basket":
                # Strict diversified basket trading with auto-reversion
                signals = get_basket_signals()
                opened = 0
                held_symbols = {pos.get('symbol'): (pos_id, pos['action']) for pos_id, pos in active_positions.items()}
                
                # Debug: Print current positions and signals
                print(f"\nCurrent positions: {len(active_positions)}/{MAX_POSITIONS}")
                for pos_id, pos in active_positions.items():
                    print(f"  {pos_id}: {pos.get('symbol', 'UNKNOWN')} - {pos['action']} @ ₹{pos['entry_price']}")
                
                print(f"Generated signals: {len(signals)}")
                for signal_data in signals:
                    print(f"  {signal_data['symbol']}: {signal_data['signal']} (confidence: {signal_data['confidence']:.3f})")
                
                # STEP 1: Check for reversals FIRST (highest priority)
                print("\n=== STEP 1: Checking for reversals ===")
                for signal_data in signals:
                    symbol = signal_data['symbol']
                    signal = signal_data['signal']
                    confidence = signal_data['confidence']
                    
                    # Check if we already have a position in this symbol
                    if symbol in held_symbols:
                        pos_id, held_action = held_symbols[symbol]
                        # Determine if the signal is a reversal
                        if (held_action == "BUY" and signal == -1) or (held_action == "SELL" and signal == 1):
                            print(f"[AUTO-REVERSE] {symbol}: Reverse signal detected. Closing {held_action} and opening {'SELL' if held_action == 'BUY' else 'BUY'}.")
                            exit_position(pos_id)
                            # Wait a moment to ensure position is closed
                            time.sleep(1)
                            # Calculate position size and strictly enforce min trade value
                            quantity = calculate_position_size()
                            if quantity == 0:
                                print(f"[STRICT] Skipping {symbol}: position size is zero after reversal.")
                                continue
                            quote = client.quotes(symbol=symbol, exchange=EXCHANGE)
                            if quote.get('status') == 'success':
                                current_price = float(quote['data']['ltp'])
                            else:
                                print(f"[STRICT] Skipping {symbol}: could not fetch price for min cash check after reversal.")
                                continue
                            total_value = quantity * current_price
                            if total_value < MIN_CASH_PER_TRADE:
                                print(f"[STRICT] Skipping {symbol}: trade value ₹{total_value:.2f} < MIN_CASH_PER_TRADE ₹{MIN_CASH_PER_TRADE} after reversal")
                                continue
                            if signal == 1:
                                print(f"[AUTO-REVERSE] Opening BUY position for {symbol}")
                                place_basket_order("BUY", symbol)
                                opened += 1
                                held_symbols[symbol] = (f"pos_{len(active_positions)}", "BUY")
                            elif signal == -1:
                                print(f"[AUTO-REVERSE] Opening SELL position for {symbol}")
                                place_basket_order("SELL", symbol)
                                opened += 1
                                held_symbols[symbol] = (f"pos_{len(active_positions)}", "SELL")
                        else:
                            print(f"[INFO] {symbol}: Already have {held_action} position, new signal is {signal} (no reversal)")
                
                # STEP 2: Check for new positions (if slots available)
                print("\n=== STEP 2: Checking for new positions ===")
                positions_left = MAX_POSITIONS - len(active_positions)
                print(f"Positions left after reversals: {positions_left}")
                
                if positions_left > 0:
                    for signal_data in signals:
                        symbol = signal_data['symbol']
                        signal = signal_data['signal']
                        confidence = signal_data['confidence']
                        
                        # Skip if we already have a position in this symbol (handled in step 1)
                        if symbol in held_symbols:
                            continue
                            
                        if positions_left <= 0:
                            print(f"[STRICT] Max positions ({MAX_POSITIONS}) reached. Not opening more.")
                            break
                            
                        # Calculate position size and strictly enforce min trade value
                        quantity = calculate_position_size()
                        if quantity == 0:
                            print(f"[STRICT] Skipping {symbol}: position size is zero.")
                            continue
                        quote = client.quotes(symbol=symbol, exchange=EXCHANGE)
                        if quote.get('status') == 'success':
                            current_price = float(quote['data']['ltp'])
                        else:
                            print(f"[STRICT] Skipping {symbol}: could not fetch price for min cash check.")
                            continue
                        total_value = quantity * current_price
                        if total_value < MIN_CASH_PER_TRADE:
                            print(f"[STRICT] Skipping {symbol}: trade value ₹{total_value:.2f} < MIN_CASH_PER_TRADE ₹{MIN_CASH_PER_TRADE}")
                            continue
                        print(f"Basket Signal: {symbol} - {signal} (confidence: {confidence:.3f})")
                        if signal == 1:
                            print(f"Received BUY signal for {symbol} (Uptrend)")
                            place_basket_order("BUY", symbol)
                            opened += 1
                            positions_left -= 1
                            held_symbols[symbol] = (f"pos_{len(active_positions)}", "BUY")
                        elif signal == -1:
                            print(f"Received SELL signal for {symbol} (Downtrend)")
                            place_basket_order("SELL", symbol)
                            opened += 1
                            positions_left -= 1
                            held_symbols[symbol] = (f"pos_{len(active_positions)}", "SELL")
                        elif signal == 0:
                            print(f"Received HOLD signal for {symbol} (Neutral) - No action taken")
                else:
                    print("[INFO] No positions left for new trades after reversals")
                
                if opened == 0:
                    print("No new diversified basket signals generated")
            else:
                # Single stock mode
                signal = get_live_signal()
                if signal is not None:
                    if signal == 1:  # Buy signal (Uptrend)
                        print("Received BUY signal (Uptrend)")
                        if len(active_positions) < MAX_POSITIONS:
                            place_live_order("BUY")
                        else:
                            print(f"Cannot take more positions. Current positions: {len(active_positions)}/{MAX_POSITIONS}")
                    elif signal == -1:  # Sell signal (Downtrend)
                        print("Received SELL signal (Downtrend)")
                        if len(active_positions) < MAX_POSITIONS:
                            place_live_order("SELL")  # Short sell for intraday
                        else:
                            print(f"Cannot take more positions. Current positions: {len(active_positions)}/{MAX_POSITIONS}")
                    elif signal == 0:  # Hold signal (Neutral)
                        print("Received HOLD signal (Neutral) - No action taken")
                    else:
                        print(f"Unknown signal value: {signal}")
                else:
                    print("No single-stock signal generated")
                    
            time.sleep(300)  # Check for signals every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in live trading thread: {str(e)}\n{traceback.format_exc()}")
            time.sleep(60)  # Wait before retrying

def save_model_artifacts(tuned_model: XGBClassifier, scaler: StandardScaler, final_selected_features: list, best_params: dict, class_mapping: dict, save_dir: str = "model_artifacts"):
    """Save model artifacts for live trading"""
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(save_dir, "model.json")
        tuned_model.save_model(model_path)
        
        # Create a new scaler that only works with the selected features
        # This is crucial - we need a scaler that matches the model's feature set
        selected_scaler = StandardScaler()
        
        # Get the original scaler's parameters for the selected features only
        if hasattr(scaler, 'feature_names_in_'):
            # Find indices of selected features in the original scaler
            original_features = scaler.feature_names_in_
            selected_indices = []
            for feature in final_selected_features:
                if feature in original_features:
                    idx = np.where(original_features == feature)[0][0]
                    selected_indices.append(idx)
                else:
                    print(f"Warning: Feature {feature} not found in original scaler")
            
            # Extract parameters for selected features only
            selected_scaler.mean_ = scaler.mean_[selected_indices]
            selected_scaler.scale_ = scaler.scale_[selected_indices]
            selected_scaler.n_features_in_ = len(final_selected_features)
            selected_scaler.feature_names_in_ = np.array(final_selected_features)
        else:
            # Fallback: create a new scaler with the same parameters for selected features
            # This assumes the scaler was trained on features in the same order as final_selected_features
            if len(scaler.mean_) >= len(final_selected_features):
                selected_scaler.mean_ = scaler.mean_[:len(final_selected_features)]
                selected_scaler.scale_ = scaler.scale_[:len(final_selected_features)]
                selected_scaler.n_features_in_ = len(final_selected_features)
                selected_scaler.feature_names_in_ = np.array(final_selected_features)
        
        # Save the selected scaler
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(selected_scaler, f)
        
        # Save selected features with metadata
        features_metadata = {
            'features': final_selected_features,
            'feature_count': len(final_selected_features),
            'feature_order': list(final_selected_features),
            'timestamp': datetime.now().isoformat()
        }
        features_path = os.path.join(save_dir, "selected_features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(features_metadata, f)
        
        # Save best parameters
        params_path = os.path.join(save_dir, "best_params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump(best_params, f)
            
        # Save feature engineering parameters
        feature_params = {
            'CORRELATION_THRESHOLD': CORRELATION_THRESHOLD,
            'VIF_THRESHOLD': VIF_THRESHOLD,
            'N_UNIVARIATE_FEATURES': N_UNIVARIATE_FEATURES,
            'N_XGB_IMPORTANCE_FEATURES': N_XGB_IMPORTANCE_FEATURES,
            'RFECV_MIN_FEATURES': RFECV_MIN_FEATURES
        }
        feature_params_path = os.path.join(save_dir, "feature_params.pkl")
        with open(feature_params_path, 'wb') as f:
            pickle.dump(feature_params, f)
            
        # Save class mapping for multi-class classification
        class_mapping_info = {
            'classes': list(class_mapping.keys()),  # Original labels [-1, 0, 1]
            'xgboost_classes': list(class_mapping.values()),  # XGBoost labels [0, 1, 2]
            'class_names': ['Downtrend', 'Neutral', 'Uptrend'],
            'class_mapping': class_mapping,  # Original to XGBoost mapping
            'reverse_mapping': {v: k for k, v in class_mapping.items()}  # XGBoost to original mapping
        }
        class_mapping_path = os.path.join(save_dir, "class_mapping.pkl")
        with open(class_mapping_path, 'wb') as f:
            pickle.dump(class_mapping_info, f)
            
        logger.info(f"Model artifacts saved to {save_dir}")
        logger.info(f"Number of selected features: {len(final_selected_features)}")
        logger.info(f"Selected features: {final_selected_features}")
        logger.info(f"Scaler feature count: {selected_scaler.n_features_in_}")
        logger.info(f"Scaler feature names: {selected_scaler.feature_names_in_}")
        logger.info(f"Class mapping: {class_mapping}")
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise

def load_model_artifacts(save_dir="model_artifacts"):
    """Load model artifacts for live trading"""
    try:
        # Load the model
        model_path = os.path.join(save_dir, "model.json")
        model = XGBClassifier()
        model.load_model(model_path)
        
        # Load the scaler
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load selected features with metadata
        features_path = os.path.join(save_dir, "selected_features.pkl")
        with open(features_path, 'rb') as f:
            features_metadata = pickle.load(f)
        
        # Load best parameters
        params_path = os.path.join(save_dir, "best_params.pkl")
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)
            
        # Load feature engineering parameters
        feature_params_path = os.path.join(save_dir, "feature_params.pkl")
        with open(feature_params_path, 'rb') as f:
            feature_params = pickle.load(f)
            
        # Load class mapping for multi-class classification
        class_mapping_path = os.path.join(save_dir, "class_mapping.pkl")
        with open(class_mapping_path, 'rb') as f:
            class_mapping_info = pickle.load(f)
            
        # Extract features
        final_selected_features = features_metadata['features']
        
        # Verify feature counts
        logger.info(f"Model artifacts loaded from {save_dir}")
        logger.info(f"Number of selected features: {len(final_selected_features)}")
        logger.info(f"Selected features: {final_selected_features}")
        logger.info(f"Feature engineering parameters: {feature_params}")
        logger.info(f"Scaler feature count: {scaler.n_features_in_}")
        logger.info(f"Class mapping: {class_mapping_info}")
        
        # Verify feature counts match
        if len(final_selected_features) != scaler.n_features_in_:
            logger.error(f"Feature count mismatch: selected features ({len(final_selected_features)}) != scaler features ({scaler.n_features_in_})")
            raise ValueError("Feature count mismatch between selected features and scaler")
        
        return model, scaler, final_selected_features, best_params, class_mapping_info
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise

def square_off_all_positions():
    """Square off all positions at market close for intraday trading"""
    print("Market close detected. Squaring off all positions...")
    for position_id in list(active_positions.keys()):
        print(f"Squaring off position {position_id} at market close")
        exit_position(position_id)
    print("All positions squared off for the day.")

def check_market_hours():
    """Check if market is open for intraday trading"""
    now = datetime.now()
    # Market hours: 9:15 AM to 3:30 PM (Monday to Friday)
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False, "Weekend"
    
    # Check if market is open
    if market_start <= now <= market_end:
        return True, "Market Open"
    elif now < market_start:
        return False, "Before Market Hours"
    else:
        return False, "After Market Hours"

def get_basket_signals():
    """Get all strong trading signals from basket of Nifty50 stocks using stock-specific models"""
    try:
        print("\nScanning Nifty50 basket for trading opportunities...")
        stock_signals = {}
        stock_models = {}
        for symbol in NIFTY50_BASKET[:BASKET_SIZE]:
            try:
                print(f"Analyzing {symbol}...")
                model_data = load_stock_specific_model(symbol)
                if model_data is None:
                    print(f"  No existing model for {symbol}, training new model...")
                    model_data = train_stock_specific_model(symbol)
                if model_data is None:
                    if SKIP_INCOMPATIBLE_STOCKS:
                        print(f"  Failed to train model for {symbol}, skipping...")
                        continue
                    else:
                        print(f"  Failed to train model for {symbol}, stopping basket analysis...")
                        return []
                stock_models[symbol] = model_data
                signal_data = get_stock_specific_signal(symbol, stock_models)
                if signal_data and signal_data['confidence'] > BACKTEST_PROB_THRESHOLD:
                    stock_signals[symbol] = signal_data
                    print(f"  {symbol}: {signal_data['signal']} (confidence: {signal_data['confidence']:.3f})")
                else:
                    confidence = signal_data['confidence'] if signal_data else 0
                    print(f"  {symbol}: No signal (confidence: {confidence:.3f})")
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        print("\nBasket Analysis Complete:")
        print(f"Stocks analyzed: {len(NIFTY50_BASKET[:BASKET_SIZE])}")
        print(f"Models available: {len(stock_models)}")
        print(f"Signals generated: {len(stock_signals)}")
        # Debug: Print signal distribution
        signal_counts = Counter([s['signal'] for s in stock_signals.values()])
        print(f"Signal distribution in basket: {signal_counts}")
        # Return all strong signals sorted by confidence (desc)
        strong_signals = sorted(stock_signals.values(), key=lambda x: x['confidence'], reverse=True)
        return strong_signals
    except Exception as e:
        logger.error(f"Error in basket signal generation: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def place_basket_order(action, symbol, position_id=None):
    """Place a smart order for basket trading with position management for intraday trading"""
    global active_positions
    
    try:
        # Calculate position size
        quantity = calculate_position_size()
        if quantity == 0:
            print("Cannot place order: Invalid position size")
            return
            
        # Check if we can take more positions
        if len(active_positions) >= MAX_POSITIONS:
            print(f"Cannot take more positions. Current positions: {len(active_positions)}")
            return
            
        print(f"Placing {action} order for {symbol}")
        resp = client.placesmartorder(
            strategy=STRATEGY_NAME,
            symbol=symbol,
            exchange=EXCHANGE,
            action=action,
            price_type=PRICE_TYPE,
            product=PRODUCT,
            quantity=quantity,
            position_size=quantity
        )
        print("Order Response:", resp)
        
        if resp.get("status") == "success":
            order_id = resp.get("orderid")
            time.sleep(1)  # Wait for order to be processed
            
            # Get order status and execution details
            status = client.orderstatus(order_id=order_id, strategy=STRATEGY_NAME)
            data = status.get("data", {})
            
            if data.get("order_status", "").lower() == "complete":
                # Get the actual execution price from the order status
                entry_price = float(data.get("average_price", 0))
                if entry_price == 0:
                    # If average_price is not available, try to get the last traded price
                    quote = client.quotes(symbol=symbol, exchange=EXCHANGE)
                    if quote.get('status') == 'success':
                        entry_price = float(quote['data']['ltp'])
                    else:
                        print("Warning: Could not get execution price, using last traded price")
                        return
                
                # Calculate stop loss and target based on position type
                if action == "BUY":  # Long position
                    stoploss_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
                    target_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)
                else:  # SELL - Short position
                    stoploss_price = round(entry_price * (1 + STOP_LOSS_PCT), 2)
                    target_price = round(entry_price * (1 - TAKE_PROFIT_PCT), 2)
                
                # Store position information
                position_id = position_id or f"pos_{len(active_positions) + 1}"
                active_positions[position_id] = {
                    'action': action,
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'stoploss': stoploss_price,
                    'target': target_price,
                    'quantity': quantity,
                    'order_id': order_id,
                    'total_value': round(entry_price * quantity, 2),
                    'position_type': 'LONG' if action == 'BUY' else 'SHORT'
                }
                
                print(f"Position {position_id} opened ({active_positions[position_id]['position_type']}):")
                print(f"Symbol: {symbol} | Entry @ ₹{entry_price} | SL ₹{stoploss_price} | Target ₹{target_price}")
                print(f"Quantity: {quantity} shares | Total Value: ₹{active_positions[position_id]['total_value']}")
            else:
                print(f"Order not completed. Status: {data.get('order_status')}")
    except Exception as e:
        print(f"Error placing basket order: {e}")
        print(traceback.format_exc())

def train_stock_specific_model(symbol, start_date="2022-01-01", end_date="2024-12-31"):
    """Train a model specifically for a given stock"""
    try:
        print(f"\nTraining model for {symbol}...")
        
        # Fetch data for specific stock
        data_full = fetch_historical_data(
            client=client,
            symbol=symbol,
            exchange=EXCHANGE,
            interval=INTERVAL,
            start_date=start_date,
            end_date=end_date
        )
        
        if data_full is None or data_full.empty:
            print(f"Error: Could not fetch data for {symbol}")
            return None
            
        print(f"Fetched {len(data_full)} records for {symbol}")
        
        # Check minimum data requirement
        if len(data_full) < 200:
            print(f"Error: Insufficient data for {symbol} ({len(data_full)} < 200 records)")
            return None
        
        # Prepare target variable
        data_full = prepare_target_variable(data_full, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
        
        # Check data after target preparation
        if len(data_full) < 100:
            print(f"Error: Insufficient data after target preparation for {symbol} ({len(data_full)} < 100 records)")
            return None
        
        # Engineer features
        data_full, initial_features = engineer_features(data_full)
        
        # Check data after feature engineering
        if len(data_full) < 50:
            print(f"Error: Insufficient data after feature engineering for {symbol} ({len(data_full)} < 50 records)")
            return None
        
        print(f"Final dataset size for {symbol}: {len(data_full)} records")
        print(f"Features generated for {symbol}: {len(initial_features)}")
        
        # Ensure we have a minimum number of features
        if len(initial_features) < MIN_FEATURES_REQUIRED:
            print(f"Error: Too few features for {symbol} ({len(initial_features)} < {MIN_FEATURES_REQUIRED})")
            return None
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, class_mapping = split_data(
            data_full, initial_features, END_DATE_TRAIN_VAL
        )
        # Print label distribution for training data
        print_label_distribution(y_train, stock=symbol)
        
        # Validate split results
        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Error: Empty training or validation set for {symbol}")
            return None
        
        # Scale features
        X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, original_scaler = scale_features(X_train, X_val, X_test)
        
        # Select features
        X_train_final, X_val_final, X_test_final, final_selected_features = select_features(
            X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train
        )
        
        # Ensure we have a reasonable number of final features
        if len(final_selected_features) < MIN_FINAL_FEATURES:
            print(f"Error: Too few final features for {symbol} ({len(final_selected_features)} < {MIN_FINAL_FEATURES})")
            return None
        
        print(f"Final selected features for {symbol}: {len(final_selected_features)}")
        
        # Create a new scaler that only works with the selected features
        # This is crucial for consistency between training and prediction
        selected_scaler = StandardScaler()
        
        # Get the original scaler's parameters for the selected features only
        if hasattr(original_scaler, 'feature_names_in_'):
            # Find indices of selected features in the original scaler
            original_features = original_scaler.feature_names_in_
            selected_indices = []
            for feature in final_selected_features:
                if feature in original_features:
                    idx = np.where(original_features == feature)[0][0]
                    selected_indices.append(idx)
                else:
                    print(f"Warning: Feature {feature} not found in original scaler")
            
            # Extract parameters for selected features only
            selected_scaler.mean_ = original_scaler.mean_[selected_indices]
            selected_scaler.scale_ = original_scaler.scale_[selected_indices]
            selected_scaler.n_features_in_ = len(final_selected_features)
            selected_scaler.feature_names_in_ = np.array(final_selected_features)
        else:
            # Fallback: create a new scaler with the same parameters for selected features
            if len(original_scaler.mean_) >= len(final_selected_features):
                selected_scaler.mean_ = original_scaler.mean_[:len(final_selected_features)]
                selected_scaler.scale_ = original_scaler.scale_[:len(final_selected_features)]
                selected_scaler.n_features_in_ = len(final_selected_features)
                selected_scaler.feature_names_in_ = np.array(final_selected_features)
            else:
                print(f"Warning: Original scaler has fewer features ({len(original_scaler.mean_)}) than selected features ({len(final_selected_features)})")
                # Create a minimal scaler
                selected_scaler.mean_ = np.zeros(len(final_selected_features))
                selected_scaler.scale_ = np.ones(len(final_selected_features))
                selected_scaler.n_features_in_ = len(final_selected_features)
                selected_scaler.feature_names_in_ = np.array(final_selected_features)
        
        # Train model
        tuned_model, best_params = tune_and_train_model(
            X_train_final, X_val_final, y_train, y_val, RANDOM_SEED, N_OPTUNA_TRIALS, OPTUNA_CV_SPLITS
        )
        
        # Save model artifacts for this specific stock
        model_dir = f"model_artifacts/{symbol}"
        save_model_artifacts(tuned_model, original_scaler, final_selected_features, best_params, class_mapping, model_dir)
        
        print(f"Model training completed for {symbol}!")
        print(f"Model saved with {len(final_selected_features)} features")
        
        return {
            'model': tuned_model,
            'scaler': selected_scaler,  # Return the selected scaler, not the original
            'features': final_selected_features,
            'params': best_params,
            'class_mapping': class_mapping,
            'model_dir': model_dir
        }
        
    except Exception as e:
        print(f"Error training model for {symbol}: {e}")
        print(traceback.format_exc())
        return None

def load_stock_specific_model(symbol):
    """Load model artifacts for a specific stock"""
    try:
        model_dir = f"model_artifacts/{symbol}"
        model, scaler, final_selected_features, best_params, class_mapping = load_model_artifacts(model_dir)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': final_selected_features,
            'params': best_params,
            'class_mapping': class_mapping,
            'model_dir': model_dir
        }
        
        # Validate the loaded model
        if not validate_model_compatibility(model_data, symbol):
            print(f"Model validation failed for {symbol}, will retrain")
            return None
        
        return model_data
    except Exception as e:
        print(f"Error loading model for {symbol}: {e}")
        return None

def validate_model_compatibility(model_data, symbol):
    """
    Validate that a loaded model is compatible with current data structure.
    
    Args:
        model_data: Dictionary containing model artifacts
        symbol: Stock symbol for logging
        
    Returns:
        bool: True if model is compatible, False otherwise
    """
    try:
        if model_data is None:
            print(f"No model data for {symbol}")
            return False
            
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        # Check if all components exist
        if model is None or scaler is None or features is None:
            print(f"Missing model components for {symbol}")
            return False
        
        # Check feature count consistency
        if len(features) != scaler.n_features_in_:
            print(f"Feature count mismatch for {symbol}: features={len(features)}, scaler={scaler.n_features_in_}")
            return False
        
        # Check if model has the expected number of features
        if hasattr(model, 'n_features_in_'):
            if model.n_features_in_ != len(features):
                print(f"Model feature count mismatch for {symbol}: model={model.n_features_in_}, features={len(features)}")
                return False
        
        print(f"Model validation passed for {symbol}: {len(features)} features")
        return True
        
    except Exception as e:
        print(f"Error validating model for {symbol}: {e}")
        return False

def safe_scaler_transform(scaler, features_df, expected_features):
    """
    Safely transform features using a scaler, handling feature name mismatches and cleaning inf/nan/large values.
    
    Args:
        scaler: Fitted StandardScaler with feature_names_in_ attribute
        features_df: DataFrame with features to transform
        expected_features: List of expected feature names in correct order
        
    Returns:
        Transformed features as DataFrame
    """
    try:
        # Validate inputs
        if scaler is None:
            raise ValueError("Scaler is None")
        if features_df is None or features_df.empty:
            raise ValueError("Features DataFrame is None or empty")
        if not expected_features:
            raise ValueError("Expected features list is empty")
        
        # Ensure features_df is a DataFrame
        if not isinstance(features_df, pd.DataFrame):
            features_df = pd.DataFrame(features_df)
        
        # Clean inf/nan/very large values
        features_df = features_df.replace([np.inf, -np.inf], 0.0)
        features_df = features_df.fillna(0.0)
        features_df = features_df.clip(-1e6, 1e6)
        
        # Check if scaler has feature names
        if not hasattr(scaler, 'feature_names_in_'):
            print("Warning: Scaler does not have feature_names_in_ attribute")
            # Try to set feature names from expected_features
            scaler.feature_names_in_ = np.array(expected_features)
        
        # Create aligned DataFrame with exact feature order
        aligned_df = pd.DataFrame(index=features_df.index)
        
        # Add features in the exact order expected by the scaler
        for feature in expected_features:
            if feature in features_df.columns:
                aligned_df[feature] = features_df[feature]
            else:
                print(f"  Adding missing feature '{feature}' with zeros")
                aligned_df[feature] = 0.0
        
        # Ensure correct order matches expected_features exactly
        aligned_df = aligned_df[expected_features]
        
        # Final cleaning
        aligned_df = aligned_df.replace([np.inf, -np.inf], 0.0)
        aligned_df = aligned_df.fillna(0.0)
        aligned_df = aligned_df.clip(-1e6, 1e6)
        
        # Verify dimensions
        if aligned_df.shape[1] != len(expected_features):
            raise ValueError(f"Feature count mismatch after alignment: {aligned_df.shape[1]} != {len(expected_features)}")
        
        if aligned_df.shape[1] != scaler.n_features_in_:
            raise ValueError(f"Feature count mismatch with scaler: {aligned_df.shape[1]} != {scaler.n_features_in_}")
        
        print(f"  Aligned features: {aligned_df.shape[1]} features")
        print(f"  Scaler expects: {scaler.n_features_in_} features")
        
        # Transform features
        features_scaled = scaler.transform(aligned_df)
        features_scaled_df = pd.DataFrame(features_scaled, columns=aligned_df.columns, index=aligned_df.index)
        
        return features_scaled_df
        
    except Exception as e:
        print(f"Error in safe_scaler_transform: {e}")
        print(f"Features DataFrame shape: {features_df.shape if features_df is not None else 'None'}")
        print(f"Expected features count: {len(expected_features)}")
        print(f"Scaler n_features_in_: {scaler.n_features_in_ if scaler is not None else 'None'}")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"Scaler feature names: {scaler.feature_names_in_}")
        raise

def align_features(df, expected_features, fill_method='zero'):
    """
    Align features in DataFrame to match expected feature set.
    
    Args:
        df: DataFrame with current features
        expected_features: List of expected feature names
        fill_method: How to fill missing features ('zero', 'mean', 'median', 'forward_fill')
        
    Returns:
        DataFrame with aligned features
    """
    try:
        # Validate inputs
        if df is None or df.empty:
            print("Error: Input DataFrame is None or empty")
            return None
            
        if not expected_features or len(expected_features) == 0:
            print("Error: Expected features list is empty")
            return None
        
        # Create a copy to avoid modifying original
        aligned_df = df.copy()
        
        # Check which features are missing
        missing_features = set(expected_features) - set(df.columns)
        extra_features = set(df.columns) - set(expected_features)
        
        if missing_features:
            print(f"Missing features: {len(missing_features)}")
            if len(missing_features) <= 10:
                print(f"Missing: {list(missing_features)}")
            else:
                print(f"Missing: {list(missing_features)[:5]}... and {len(missing_features)-5} more")
            
            # Add missing features with appropriate default values
            for feature in missing_features:
                if fill_method == 'zero':
                    aligned_df[feature] = 0.0
                elif fill_method == 'mean':
                    # Use mean of available features as approximation
                    aligned_df[feature] = aligned_df.mean(axis=1)
                elif fill_method == 'median':
                    aligned_df[feature] = aligned_df.median(axis=1)
                elif fill_method == 'forward_fill':
                    aligned_df[feature] = aligned_df.iloc[:, 0]  # Use first column as proxy
                else:
                    aligned_df[feature] = 0.0
        
        if extra_features:
            print(f"Extra features: {len(extra_features)}")
            if len(extra_features) <= 10:
                print(f"Extra: {list(extra_features)}")
            else:
                print(f"Extra: {list(extra_features)[:5]}... and {len(extra_features)-5} more")
            # Note: We don't remove extra features anymore to avoid scaler issues
            print("Keeping extra features to avoid scaler compatibility issues")
        
        # Ensure correct order of features - only select the expected features
        try:
            aligned_df = aligned_df[expected_features]
        except KeyError as e:
            print(f"Error: Could not reorder features. Missing: {e}")
            return None
        
        print(f"Feature alignment: {len(df.columns)} -> {len(aligned_df.columns)} features")
        
        # Final validation
        if len(aligned_df.columns) != len(expected_features):
            print(f"Error: Final feature count mismatch: {len(aligned_df.columns)} != {len(expected_features)}")
            return None
            
        if not all(col in aligned_df.columns for col in expected_features):
            print("Error: Not all expected features are present after alignment")
            return None
        
        # Handle any remaining NaN values
        if aligned_df.isnull().any().any():
            print("Warning: NaN values found in aligned features, filling with zeros")
            aligned_df = aligned_df.fillna(0.0)
        
        return aligned_df
        
    except Exception as e:
        print(f"Error in feature alignment: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_stock_specific_signal(symbol, stock_models):
    """Get trading signal for a specific stock using its trained model"""
    try:
        if symbol not in stock_models:
            print(f"No model available for {symbol}")
            return None
            
        model_data = stock_models[symbol]
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        class_mapping = model_data['class_mapping']
        
        # Fetch recent data for this stock
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        df = client.history(
            symbol=symbol,
            exchange=EXCHANGE,
            interval=INTERVAL,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if df is None or len(df) < 50:
            return None
            
        # Prepare features for prediction
        df = prepare_target_variable(df, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
        df, initial_features = engineer_features(df)
        
        # Align features to match the model's expected feature set and order
        df_aligned = align_features(df, features, fill_method=FEATURE_ALIGNMENT_METHOD)
        
        if df_aligned is None:
            print(f"Failed to align features for {symbol}")
            return None
        
        # Select the latest row for prediction
        features_df = df_aligned.iloc[-1:].copy()
        
        # Ensure DataFrame columns match scaler's expected feature names and order
        features_df = features_df[features]  # This ensures correct order
        
        # Verify feature dimensions
        print(f"{symbol}: Features shape: {features_df.shape}, Expected: (1, {len(features)})")
        print(f"{symbol}: Scaler expects: {scaler.n_features_in_} features")
        
        if features_df.shape[1] != scaler.n_features_in_:
            print(f"Feature dimension mismatch for {symbol}: {features_df.shape[1]} != {scaler.n_features_in_}")
            return None
        
        # Scale features using stock-specific scaler - pass DataFrame directly
        features_scaled_df = safe_scaler_transform(scaler, features_df, features)
        
        # Get prediction
        prediction_proba = model.predict_proba(features_scaled_df)[0]
        predicted_class = np.argmax(prediction_proba)
        confidence = np.max(prediction_proba)
        
        # Map prediction back to original label
        reverse_mapping = {v: k for k, v in class_mapping['class_mapping'].items()}
        predicted_label = reverse_mapping[predicted_class]
        
        return {
            'symbol': symbol,
            'signal': predicted_label,
            'confidence': confidence,
            'probabilities': prediction_proba
        }
        
    except Exception as e:
        print(f"Error getting signal for {symbol}: {e}")
        return None

def run_basket_backtesting():
    """Run backtesting for basket trading with multiple stocks"""
    logger.info("Starting basket backtesting...")
    
    try:
        # Track results for all stocks
        basket_results = {}
        total_trades = 0
        total_return = 0
        
        print(f"Running basket backtesting on {BASKET_SIZE} stocks...")
        
        # Process each stock in the basket
        for i, symbol in enumerate(NIFTY50_BASKET[:BASKET_SIZE], 1):
            print(f"\n--- Processing Stock {i}/{BASKET_SIZE}: {symbol} ---")
            
            try:
                # Train model for this stock (this already does all the work)
                model_data = train_stock_specific_model(symbol)
                
                if model_data is None:
                    print(f"Skipping {symbol}: Failed to train model")
                    continue
                
                # Extract the trained model and results from model_data
                tuned_model = model_data['model']
                scaler = model_data['scaler']
                final_selected_features = model_data['features']
                class_mapping = model_data['class_mapping']
                
                # Get historical data for backtesting (use the same data that was used for training)
                data_full = fetch_historical_data(
                    client=client,
                    symbol=symbol,
                    exchange=EXCHANGE,
                    interval=INTERVAL,
                    start_date=START_DATE,
                    end_date=END_DATE_TEST
                )
                
                if data_full is None or data_full.empty:
                    print(f"Skipping {symbol}: No data available")
                    continue
                
                # Prepare data for backtesting
                data_full = prepare_target_variable(data_full, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
                data_full, initial_features = engineer_features(data_full)
                
                # Split data
                X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, class_mapping = split_data(
                    data_full, initial_features, END_DATE_TRAIN_VAL
                )
                
                # Align test features to match the model's expected features
                X_test_aligned = align_features(X_test, final_selected_features, fill_method=FEATURE_ALIGNMENT_METHOD)
                
                if X_test_aligned is None:
                    print(f"Failed to align features for {symbol} backtesting")
                    continue
                
                # Verify feature dimensions match the scaler
                if X_test_aligned.shape[1] != scaler.n_features_in_:
                    print(f"Feature dimension mismatch for {symbol}: {X_test_aligned.shape[1]} != {scaler.n_features_in_}")
                    print(f"Expected features: {len(final_selected_features)}")
                    print(f"Scaler expects: {scaler.n_features_in_}")
                    continue
                
                # Scale features using the stock-specific scaler
                X_test_scaled_df = safe_scaler_transform(scaler, X_test_aligned, final_selected_features)
                
                if X_test_scaled_df is None:
                    print(f"Failed to scale features for {symbol} backtesting")
                    continue
                
                # Use the scaled features directly
                X_test_final = X_test_scaled_df
                
                # Evaluate model (the model is already trained)
                eval_result = evaluate_model(tuned_model, X_test_final, y_test, class_mapping, display_plot=False)
                
                # Run backtesting
                backtest_results = run_backtest(
                    test_df, eval_result.get("probabilities"), symbol, class_mapping, 
                    BACKTEST_PROB_THRESHOLD, RISK_FREE_RATE_ANNUAL, display_plot=False
                )
                
                # Store results
                basket_results[symbol] = {
                    'backtest_results': backtest_results,
                    'eval_result': eval_result,
                    'model_data': model_data
                }
                
                total_trades += backtest_results.get('num_trades', 0)
                total_return += backtest_results.get('strategy_total_return', 0)
                
                print(f"✅ {symbol}: {backtest_results.get('num_trades', 0)} trades, "
                      f"Annualized Return: {backtest_results.get('strategy_annual_return', 0):.2%}, "
                      f"Sharpe: {backtest_results.get('strategy_sharpe', 0):.2f}")
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print basket summary
        print("\n=== Basket Backtesting Summary ===")
        print(f"Stocks processed: {len(basket_results)}/{BASKET_SIZE}")
        print(f"Total trades: {total_trades}")
        
        # Calculate annualized returns instead of cumulative
        if basket_results:
            # Calculate annualized returns for each stock
            annualized_returns = []
            sharpe_ratios = []
            for symbol, results in basket_results.items():
                backtest_data = results['backtest_results']
                total_return = backtest_data.get('strategy_total_return', 0)
                num_trades = backtest_data.get('num_trades', 0)
                annualized_return = backtest_data.get('strategy_annual_return', 0)
                sharpe_ratio = backtest_data.get('strategy_sharpe', 0)
                
                annualized_returns.append(annualized_return)
                sharpe_ratios.append(sharpe_ratio)
            
            avg_annualized_return = sum(annualized_returns) / len(annualized_returns)
            avg_sharpe_ratio = sum(sharpe_ratios) / len(sharpe_ratios)
            
            # Calculate average trades per day
            avg_trades_per_day = total_trades / len(basket_results) / 252  # Assuming 252 trading days
            
            # Add prominent summary
            print(f"\n{'='*60}")
            print("📊 BASKET PERFORMANCE SUMMARY")
            print(f"{'='*60}")
            print(f"🎯 Avg Annualized Return: {avg_annualized_return:.2%}")
            print(f"📈 Avg Sharpe Ratio: {avg_sharpe_ratio:.2f}")
            print(f"🔄 Total Trades: {total_trades:.0f}")
            print(f"📅 Avg Trades/Day: {avg_trades_per_day:.2f}")
            print(f"📊 Stocks Analyzed: {len(basket_results)}")
            print(f"{'='*60}")
            
            # Find best performing stocks by annualized return
            stock_performances = []
            for symbol, results in basket_results.items():
                backtest_data = results['backtest_results']
                annualized_return = backtest_data.get('strategy_annual_return', 0)
                num_trades = backtest_data.get('num_trades', 0)
                sharpe_ratio = backtest_data.get('strategy_sharpe', 0)
                
                stock_performances.append((symbol, annualized_return, num_trades, sharpe_ratio))
            
            # Sort by annualized return
            stock_performances.sort(key=lambda x: x[1], reverse=True)
            
            print("\n🏆 Top 5 Performing Stocks:")
            print(f"{'Rank':<4} {'Symbol':<12} {'Annual Return':<15} {'Sharpe':<8} {'Trades':<8}")
            print("-" * 50)
            for i, (symbol, annualized_return, num_trades, sharpe_ratio) in enumerate(stock_performances[:5], 1):
                print(f"{i:<4} {symbol:<12} {annualized_return:>12.2%} {sharpe_ratio:>7.2f} {num_trades:>7.0f}")
        else:
            print("No results to display")
        
        logger.info("Basket backtesting completed successfully")
        return basket_results
        
    except Exception as e:
        logger.error(f"Error in basket backtesting: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def safe_datetime_conversion(index):
    """Safely convert index to timezone-naive datetime index"""
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is not None:
            # Convert timezone-aware to timezone-naive
            return index.tz_convert('UTC').tz_localize(None)
        return index
    else:
        # Convert to datetime, handling timezone-aware if present
        try:
            converted = pd.to_datetime(index)
            if converted.tz is not None:
                return converted.tz_convert('UTC').tz_localize(None)
            return converted
        except ValueError as e:
            if "Tz-aware" in str(e):
                # Handle timezone-aware datetime conversion
                converted = pd.to_datetime(index, utc=True)
                return converted.tz_localize(None)
            else:
                raise e

def cleanup_old_artifacts(symbol=None):
    """
    Clean up old model artifacts when feature engineering parameters change.
    This prevents using incompatible models with new feature sets.
    
    Args:
        symbol: Specific stock symbol to clean up, or None for all stocks
    """
    try:
        if symbol:
            # Clean up specific stock artifacts
            model_dir = f"model_artifacts/{symbol}"
            if os.path.exists(model_dir):
                print(f"Cleaning up old artifacts for {symbol}...")
                import shutil
                shutil.rmtree(model_dir)
                print(f"Removed {model_dir}")
        else:
            # Clean up all artifacts
            model_dir = "model_artifacts"
            if os.path.exists(model_dir):
                print("Cleaning up all old model artifacts...")
                import shutil
                shutil.rmtree(model_dir)
                print(f"Removed {model_dir}")
                
        print("Artifact cleanup completed")
        
    except Exception as e:
        print(f"Error during artifact cleanup: {e}")

def check_feature_engineering_compatibility(save_dir="model_artifacts"):
    """
    Check if existing model artifacts are compatible with current feature engineering parameters.
    
    Args:
        save_dir: Directory containing model artifacts
        
    Returns:
        bool: True if compatible, False if incompatible
    """
    try:
        if not os.path.exists(save_dir):
            print("No existing artifacts found")
            return False
            
        # Load feature engineering parameters
        feature_params_path = os.path.join(save_dir, "feature_params.pkl")
        if not os.path.exists(feature_params_path):
            print("No feature parameters found in artifacts")
            return False
            
        with open(feature_params_path, 'rb') as f:
            saved_params = pickle.load(f)
        
        # Current feature engineering parameters
        current_params = {
            'CORRELATION_THRESHOLD': CORRELATION_THRESHOLD,
            'VIF_THRESHOLD': VIF_THRESHOLD,
            'N_UNIVARIATE_FEATURES': N_UNIVARIATE_FEATURES,
            'N_XGB_IMPORTANCE_FEATURES': N_XGB_IMPORTANCE_FEATURES,
            'RFECV_MIN_FEATURES': RFECV_MIN_FEATURES
        }
        
        # Compare parameters
        if saved_params != current_params:
            print("Feature engineering parameters have changed:")
            print(f"Saved: {saved_params}")
            print(f"Current: {current_params}")
            return False
        
        print("Feature engineering parameters are compatible")
        return True
        
    except Exception as e:
        print(f"Error checking feature engineering compatibility: {e}")
        return False

def print_label_distribution(y, stock=None):
    """Print the label distribution for a given Series y (target labels)."""
    if stock:
        print(f"Label distribution for {stock}:")
    else:
        print("Label distribution:")
    print(y.value_counts(normalize=True))
    print(y.value_counts())
    print()

if __name__ == "__main__":
    try:
        # Check if we want to run backtesting or live trading
        import sys
        
        print("=== Dead Zone Strategy Configuration ===")
        print(f"Trading Mode: {TRADING_MODE}")
        print(f"Basket Size: {BASKET_SIZE}")
        print(f"Model Approach: {MODEL_APPROACH}")
        print(f"Stocks in Basket: {NIFTY50_BASKET[:BASKET_SIZE]}")
        print()
        
        # Check feature engineering compatibility and clean up if needed
        if TRADING_MODE == "single":
            if not check_feature_engineering_compatibility():
                print("Feature engineering parameters have changed. Cleaning up old artifacts...")
                cleanup_old_artifacts()
                print("Old artifacts cleaned up. Will train new models.")
        
        if len(sys.argv) > 1 and sys.argv[1] == "--live":
            logger.info("Starting Dead Zone Strategy in live trading mode...")
            
            if TRADING_MODE == "basket":
                logger.info("Running in basket trading mode...")
                # For basket trading, we don't need to pre-load a single model
                # Models will be trained/loaded on-demand for each stock
            else:
                logger.info("Running in single-stock mode...")
                # Load model artifacts for single-stock live trading
                try:
                    tuned_model, scaler, final_selected_features, best_params, class_mapping = load_model_artifacts()
                    logger.info("Successfully loaded model artifacts for live trading")
                except Exception as e:
                    logger.error(f"Failed to load model artifacts: {str(e)}")
                    logger.info("Running backtesting first to prepare model...")
                
                    # Run backtesting to prepare model for single stock
                print("--- 1. Data Acquisition and Initial Preparation ---")
                data_full = fetch_historical_data(
                    client=client,
                        symbol=DEFAULT_SYMBOL,
                    exchange=EXCHANGE,
                    interval=INTERVAL,
                    start_date=START_DATE,
                    end_date=END_DATE_TEST
                )
                
                data_full = prepare_target_variable(data_full, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
                
                print("\n--- 2. Feature Engineering ---")
                data_full, initial_features = engineer_features(data_full)
                
                print("\n--- 3. Train/Validation/Test Split ---")
                X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, class_mapping = split_data(data_full, initial_features, END_DATE_TRAIN_VAL)
                
                print("\n--- 4. Feature Scaling ---")
                X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler = scale_features(X_train, X_val, X_test)
                
                print("\n--- 5. Feature Selection Funnelling Approach ---")
                X_train_final, X_val_final, X_test_final, final_selected_features = select_features(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train)
                
                print("\n--- 6. Model Building and Hyperparameter Tuning ---")
                tuned_model, best_params = tune_and_train_model(X_train_final, X_val_final, y_train, y_val, RANDOM_SEED, N_OPTUNA_TRIALS, OPTUNA_CV_SPLITS)
                
                # Save model artifacts
                save_model_artifacts(tuned_model, scaler, final_selected_features, best_params, class_mapping)
            
            logger.info("Starting live trading...")
            
            # Start live trading
            ws_thread = threading.Thread(target=websocket_thread)
            trading_thread = threading.Thread(target=live_trading_thread)
            
            ws_thread.start()
            trading_thread.start()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down...")
                stop_event.set()
                exit_all_positions()
                ws_thread.join()
                trading_thread.join()
                logger.info("Strategy shutdown complete.")
        else:
            logger.info("Starting Dead Zone Strategy in backtesting mode...")
            
            if TRADING_MODE == "basket":
                logger.info("Running basket backtesting...")
                # Clean up old basket artifacts if feature engineering changed
                if not check_feature_engineering_compatibility():
                    print("Feature engineering parameters have changed. Cleaning up old basket artifacts...")
                    cleanup_old_artifacts()
                    print("Old basket artifacts cleaned up. Will train new models.")
                
                # Run basket backtesting
                run_basket_backtesting()
            else:
                logger.info("Running single-stock backtesting...")
                # Run the existing single-stock backtesting code
                print("--- 1. Data Acquisition and Initial Preparation ---")
                data_full = fetch_historical_data(
                    client=client,
                    symbol=DEFAULT_SYMBOL,
                    exchange=EXCHANGE,
                    interval=INTERVAL,
                    start_date=START_DATE,
                    end_date=END_DATE_TEST
                )
                
                data_full = prepare_target_variable(data_full, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
                
                print("\n--- 2. Feature Engineering ---")
                data_full, initial_features = engineer_features(data_full)
                
                print("\n--- 3. Train/Validation/Test Split ---")
                X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, class_mapping = split_data(data_full, initial_features, END_DATE_TRAIN_VAL)
                
                print("\n--- 4. Feature Scaling ---")
                X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler = scale_features(X_train, X_val, X_test)
                
                print("\n--- 5. Feature Selection Funnelling Approach ---")
                X_train_final, X_val_final, X_test_final, final_selected_features = select_features(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train)
                
                print("\n--- 6. Model Building and Hyperparameter Tuning ---")
                tuned_model, best_params = tune_and_train_model(X_train_final, X_val_final, y_train, y_val, RANDOM_SEED, N_OPTUNA_TRIALS, OPTUNA_CV_SPLITS)
                
                print("\n--- 7. Model Evaluation on Test Set ---")
                eval_result = evaluate_model(tuned_model, X_test_final, y_test, class_mapping, display_plot=False)
                
                print("\n--- 8. Backtesting the Predicted Signals ---")
                backtest_results = run_backtest(test_df, eval_result.get("probabilities"), DEFAULT_SYMBOL, class_mapping, BACKTEST_PROB_THRESHOLD, RISK_FREE_RATE_ANNUAL, display_plot=False)
                
                # Save model artifacts after successful backtesting
                save_model_artifacts(tuned_model, scaler, final_selected_features, best_params, class_mapping)
                
                logger.info("Single-stock backtesting completed successfully")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        stop_event.set()
        exit_all_positions()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)



