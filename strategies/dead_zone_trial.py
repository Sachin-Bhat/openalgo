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
END_DATE_TRAIN_VAL = "2018-12-31" # Training + Validation data ends here
END_DATE_TEST = pd.Timestamp.now().strftime("%Y-%m-%d")      # Test data uses current date
SYMBOL = "RELIANCE"
EXCHANGE = "NSE"
INTERVAL = "1m"

# Target variable definition
DEAD_ZONE_LOWER = -0.0010  # -0.10%
DEAD_ZONE_UPPER = 0.0010   # +0.10%

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
BACKTEST_PROB_THRESHOLD = 0.60 # Probability threshold to trigger a long signal
RISK_FREE_RATE_ANNUAL = 0.08 # For Sharpe Ratio calculation

# Random Seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Live Trading Configuration
STRATEGY_NAME = "DeadZone_ML_Strategy"
PRODUCT = "CNC"
PRICE_TYPE = "MARKET"

# Position and Risk Management Parameters
MAX_POSITIONS = 5  # Maximum number of concurrent positions
POSITION_SIZE = 1  # Number of shares per position
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit

# Balance and Cash Management Parameters
MIN_BALANCE = 50000  # Minimum balance required to start trading (₹50,000)
MAX_BALANCE = 55000  # Maximum balance to trade with (₹55,000)
MIN_CASH_PER_TRADE = 10000  # Minimum cash required per trade (₹10,000)
MAX_CASH_PER_TRADE = 11000  # Maximum cash to use per trade (₹11,000)
MAX_PORTFOLIO_RISK = 0.02  # Maximum 2% risk per trade

# Live Trading State Variables
ltp = None
active_positions = {}  # Dictionary to track multiple positions
stop_event = threading.Event()
instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]  # Define instrument at top level

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
            
            # Update start_date if we have existing data
            if not existing_data.empty:
                last_date = existing_data.index.max()
                if last_date >= start_dt:
                    start_dt = last_date + pd.Timedelta(minutes=1 if interval == "1m" else 1)
                    print(f"Updating start date to {start_dt} based on existing data")
        except Exception as e:
            print(f"Error reading existing data file: {e}")
            print("Will create new data file")
    
    # If we have all the data already, return it
    if existing_data is not None and start_dt >= end_dt:
        print("All requested data already exists in file")
        return existing_data
    
    # Calculate total days and number of chunks needed for new data
    total_days = (end_dt - start_dt).days
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
        chunk_start = start_dt + pd.Timedelta(days=chunk * chunk_size)
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_size), end_dt)
        
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
                        if chunk_data['timestamp'].dt.tz is None:
                            chunk_data['timestamp'] = chunk_data['timestamp'].dt.tz_localize('UTC+05:30')
                        chunk_data.set_index('timestamp', inplace=True)
                    elif chunk_data.index.name == 'timestamp':
                        if chunk_data.index.tz is None:
                            chunk_data.index = chunk_data.index.tz_localize('UTC+05:30')
                    
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
    
    # Define target variable: 1 if uptrend, 0 otherwise
    data_full.loc[:, 'Uptrend'] = np.nan
    
    # Assign labels
    data_full.loc[data_full['TargetReturn'] > dead_zone_upper, 'Uptrend'] = 1
    data_full.loc[data_full['TargetReturn'] < dead_zone_lower, 'Uptrend'] = 0
    
    # Drop ambiguous (dead zone) samples
    data_full = data_full.dropna(subset=['Uptrend'])
    data_full.loc[:, 'Uptrend'] = data_full['Uptrend'].astype(int)
    
    # Print summary statistics
    print(f"Full data shape: {data_full.shape}")
    print(f"Uptrend distribution:\n{data_full['Uptrend'].value_counts(normalize=True)}")
    
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
        initial_features = data_full.columns.drop(['open', 'high', 'low', 'close', 'volume', 'Return', 'Uptrend'])
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
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
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
    
    # Prepare features and target
    X = data_full[initial_features]
    y = data_full['Uptrend']
    
    # Split features and target
    X_train, y_train = train_df[initial_features], train_df['Uptrend']
    X_val, y_val = val_df[initial_features], val_df['Uptrend']
    X_test, y_test = test_df[initial_features], test_df['Uptrend']
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df

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
    
    Args:
        X_train: Training features DataFrame
        X_val: Validation features DataFrame
        X_test: Test features DataFrame
        
    Returns:
        tuple: (X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler)
    """
    # Clean and validate the data
    X_train_clean = validate_features(X_train)
    X_val_clean = validate_features(X_val)
    X_test_clean = validate_features(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    print(f"Training data shape after scaling: {X_train_scaled.shape}")
    print(f"Validation data shape after scaling: {X_val_scaled.shape}")
    print(f"Test data shape after scaling: {X_test_scaled.shape}")

    # Convert scaled arrays back to DataFrames with original column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

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
        selector_kbest = SelectKBest(score_func=f_classif, k=N_UNIVARIATE_FEATURES)
        X_train_kbest = selector_kbest.fit_transform(X_train_scaled_df, y_train)
        kbest_features_mask = selector_kbest.get_support()
        kbest_features = X_train_scaled_df.columns[kbest_features_mask]

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
        temp_xgb = XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
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
    scale_pos_weight_val = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1

    if len(selected_features) > RFECV_MIN_FEATURES:
        estimator_rfecv = XGBClassifier(
            scale_pos_weight=scale_pos_weight_val,
            random_state=RANDOM_SEED,
            eval_metric='auc'
        )
        cv_splitter = TimeSeriesSplit(n_splits=OPTUNA_CV_SPLITS)
        
        rfecv_selector = RFECV(
            estimator=estimator_rfecv,
            step=1,
            cv=cv_splitter,
            scoring='roc_auc',
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
        scale_pos_weight_obj = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': random_seed,
            'scale_pos_weight': scale_pos_weight_obj,
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
            preds_proba = model.predict_proba(X_cv_val)[:, 1]
            scores.append(roc_auc_score(y_cv_val, preds_proba))
        return np.mean(scores)

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_optuna_trials, n_jobs=1)
    best_params = study.best_params
    print("Best hyperparameters found by Optuna:", best_params)

    # Train final model on combined train+val data
    X_train_val_final = pd.concat([X_train_final, X_val_final])
    y_train_val = pd.concat([y_train, y_val])

    final_scale_pos_weight = (len(y_train_val) - y_train_val.sum()) / y_train_val.sum() if y_train_val.sum() > 0 else 1

    tuned_model = XGBClassifier(
        **best_params,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=random_seed,
        scale_pos_weight=final_scale_pos_weight
    )
    tuned_model.fit(X_train_val_final, y_train_val)
    print("Tuned model trained on X_train_val_final and y_train_val.")
    
    return tuned_model, best_params

# %%
@error_handler
def evaluate_model(tuned_model, X_test_final, y_test, display_plot=True):
    """
    Evaluate the model on test data and optionally display confusion matrix plot.
    
    Args:
        tuned_model: Trained XGBoost model
        X_test_final: Test features
        y_test: Test labels
        display_plot: Boolean to control whether to display confusion matrix plot
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred_test_proba = tuned_model.predict_proba(X_test_final)[:, 1]
    y_pred_test_labels = (y_pred_test_proba > 0.3).astype(int)  # Using 0.3 threshold for classification metrics
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_test_proba)
    accuracy = accuracy_score(y_test, y_pred_test_labels)
    
    print(f"Test Set ROC AUC Score: {roc_auc:.4f}")
    print(f"Test Set Accuracy Score: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test_labels)
    
    print("\nTest Set Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Uptrend', 'Uptrend'], 
               yticklabels=['Non-Uptrend', 'Uptrend'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Set Confusion Matrix')
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    if display_plot:
        plt.show()
    else:
        plt.close()
    
    # Classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test_labels, 
                              target_names=['Non-Uptrend (0)', 'Uptrend (1)']))
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': y_pred_test_labels,
        'probabilities': y_pred_test_proba
    }

# %%
@error_handler
def run_backtest(test_df, y_pred_test_proba, symbol, backtest_prob_threshold=0.5, risk_free_rate_annual=0.05, 
                display_plot=True, save_dir="."):
    """
    Run backtesting analysis on trading strategy predictions and compare with buy-and-hold strategy.
    
    Parameters
    ----------
    test_df : pandas.DataFrame
        DataFrame containing test data with columns ['Open', 'Adj Close', 'Return']
    y_pred_test_proba : numpy.ndarray
        Array of prediction probabilities for the test period
    symbol : str
        Trading symbol/stock name for labeling
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
    backtest_df['Signal_Prob'] = y_pred_test_proba
    backtest_df['Signal'] = (backtest_df['Signal_Prob'] > backtest_prob_threshold).astype(int)

    # Calculate strategy returns
    backtest_df['Position'] = backtest_df['Signal'].shift(1).fillna(0)
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

    num_trades = backtest_df['Position'].diff().abs().sum() / 2

    # Win rate calculation
    positive_strategy_days = backtest_df[backtest_df['Position'] == 1]['Strategy_Return'] > 0
    win_rate_days = positive_strategy_days.sum() / (backtest_df['Position'] == 1).sum() if (backtest_df['Position'] == 1).sum() > 0 else 0

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
            "Win Rate (+ve days in market)"
        ],
        "Trading Strategy": [
            f"{strategy_total_return:.2%}",
            f"{strategy_annual_return:.2%}",
            f"{strategy_annual_vol:.2%}",
            f"{strategy_sharpe:.2f}",
            f"{strategy_mdd:.2%}",
            f"{num_trades:.0f}",
            f"{win_rate_days:.2%}"
        ],
        f"Buy-and-Hold {symbol}": [
            f"{bh_total_return:.2%}",
            f"{bh_annual_return:.2%}",
            f"{bh_annual_vol:.2%}",
            f"{bh_sharpe:.2f}",
            f"{bh_mdd:.2%}",
            "1",
            "N/A"
        ]
    })

    print(f"\nBacktesting Results (Test Period: {test_df.index.min().date()} to {test_df.index.max().date()}):")
    print(f"Trading Signal Probability Threshold: {backtest_prob_threshold}")
    print(results_df.head(7))

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    backtest_df['Strategy_Cumulative_Return'].plot(label='Trading Strategy')
    backtest_df['BH_Cumulative_Return'].plot(label=f'Buy-and-Hold {symbol}')
    plt.title(f'{symbol} Backtest: Cumulative Returns')
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
        'win_rate': win_rate_days,
        'backtest_df': backtest_df,
        'plot_path': plot_path
    }

    return results

# %%
def on_data_received(data):
    """WebSocket LTP Handler"""
    global ltp
    print(f"Received WebSocket data: {data}")  # Debug log
    if data.get("type") == "market_data" and data.get("symbol") == SYMBOL:
        ltp = float(data["data"]["ltp"])
        print(f"LTP Update {EXCHANGE}:{SYMBOL} => ₹{ltp}")
        
        # Check stop loss and take profit for all active positions
        for position_id, position in list(active_positions.items()):
            if ltp <= position['stoploss'] or ltp >= position['target']:
                print(f"Exit Triggered for position {position_id}: LTP ₹{ltp} hit stoploss or target")
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
            
        # Use fixed position size since we're using market orders
        return POSITION_SIZE
        
    except Exception as e:
        print(f"Error calculating position size: {e}")
        return 0

def place_live_order(action, position_id=None):
    """Place a smart order with position management"""
    global active_positions
    
    try:
        # Calculate position size
        quantity = calculate_position_size()
        if quantity == 0:
            print("Cannot place order: Invalid position size")
            return
            
        # Check if we can take more positions
        if action == "BUY" and len(active_positions) >= MAX_POSITIONS:
            print(f"Cannot take more positions. Current positions: {len(active_positions)}")
            return
            
        print(f"Placing {action} order for {SYMBOL}")
        resp = client.placesmartorder(
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
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
            time.sleep(1)
            status = client.orderstatus(order_id=order_id, strategy=STRATEGY_NAME)
            data = status.get("data", {})
            
            if data.get("order_status", "").lower() == "complete":
                entry_price = float(data["price"])
                stoploss_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
                target_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)
                
                # Store position information
                position_id = position_id or f"pos_{len(active_positions) + 1}"
                active_positions[position_id] = {
                    'action': action,
                    'entry_price': entry_price,
                    'stoploss': stoploss_price,
                    'target': target_price,
                    'quantity': quantity,
                    'order_id': order_id
                }
                
                print(f"Position {position_id} opened:")
                print(f"Entry @ ₹{entry_price} | SL ₹{stoploss_price} | Target ₹{target_price}")
    except Exception as e:
        print(f"Error placing order: {e}")

def exit_position(position_id):
    """Exit a specific position"""
    try:
        if position_id not in active_positions:
            print(f"Position {position_id} not found")
            return
            
        position = active_positions[position_id]
        action = "SELL" if position['action'] == "BUY" else "BUY"
        
        print(f"Exiting position {position_id} with {action}")
        resp = client.placesmartorder(
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=action,
            price_type=PRICE_TYPE,
            product=PRODUCT,
            quantity=position['quantity'],
            position_size=position['quantity']
        )
        
        if resp.get("status") == "success":
            del active_positions[position_id]
            print(f"Position {position_id} closed successfully")
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
            symbol=SYMBOL,
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
        
        # Create a DataFrame with all required features in the same order as training
        features = pd.DataFrame(index=df.index[-1:])
        
        # First, check if all required features exist
        missing_features = set(final_selected_features) - set(df.columns)
        if missing_features:
            print(f"Error: Missing required features: {missing_features}")
            return None
            
        print("All required features available, preparing final feature set...")
        # Then, add only the required features in the correct order
        for feature in final_selected_features:
            features[feature] = df[feature].iloc[-1:]
            
        # Verify the features DataFrame
        logger.info(f"Features DataFrame shape: {features.shape}")
        logger.info(f"Features DataFrame columns: {features.columns.tolist()}")
        
        # Log feature values for debugging
        logger.info("Feature values for prediction:")
        for feature in features.columns:
            print(f"{feature}: {features[feature].iloc[0]}")
        
        # Create a new scaler with only the selected features
        selected_scaler = StandardScaler()
        selected_scaler.fit(features)
        
        # Scale features
        features_scaled = selected_scaler.transform(features)
        
        # Get prediction
        prediction = tuned_model.predict_proba(features_scaled)[0]
        print(f"Prediction probabilities: {prediction}")
        
        # Return signal based on prediction
        if prediction[1] > BACKTEST_PROB_THRESHOLD:
            print("Generated BUY signal")
            return 1  # Buy signal
        elif prediction[0] > BACKTEST_PROB_THRESHOLD:
            print("Generated SELL signal")
            return -1  # Sell signal
        else:
            print("No trading signal generated")
            return 0  # No signal
            
    except Exception as e:
        logger.error(f"Error getting live trading signal: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def live_trading_thread():
    """Thread for live trading strategy"""
    while not stop_event.is_set():
        try:
            # Generate signal regardless of LTP
            signal = get_live_signal()
            if signal is not None:  # Only proceed if we got a valid signal
                if signal == 1:  # Buy signal
                    print("Received BUY signal")
                    place_live_order("BUY")
                elif signal == -1:  # Sell signal
                    print("Received SELL signal")
                    place_live_order("SELL")
                else:
                    print("No trading signal")
                    
            time.sleep(300)  # Check for signals every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in live trading thread: {str(e)}\n{traceback.format_exc()}")
            time.sleep(60)  # Wait before retrying

def save_model_artifacts(tuned_model: XGBClassifier, scaler: StandardScaler, final_selected_features: list, best_params: dict, save_dir: str = "model_artifacts"):
    """Save model artifacts for live trading"""
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(save_dir, "model.json")
        tuned_model.save_model(model_path)
        
        # Create a new scaler with only the selected features
        selected_scaler = StandardScaler()
        
        # Get the original scaler's parameters
        original_means = scaler.mean_
        original_scales = scaler.scale_
        
        # Create a new scaler with the same parameters for selected features
        selected_scaler.mean_ = original_means
        selected_scaler.scale_ = original_scales
        selected_scaler.n_features_in_ = len(final_selected_features)
        
        # Save the scaler
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
            
        logger.info(f"Model artifacts saved to {save_dir}")
        logger.info(f"Number of selected features: {len(final_selected_features)}")
        logger.info(f"Selected features: {final_selected_features}")
        logger.info(f"Scaler feature count: {selected_scaler.n_features_in_}")
        
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
            
        # Extract features
        final_selected_features = features_metadata['features']
        
        # Verify feature counts
        logger.info(f"Model artifacts loaded from {save_dir}")
        logger.info(f"Number of selected features: {len(final_selected_features)}")
        logger.info(f"Selected features: {final_selected_features}")
        logger.info(f"Feature engineering parameters: {feature_params}")
        logger.info(f"Scaler feature count: {scaler.n_features_in_}")
        
        # Verify feature counts match
        if len(final_selected_features) != scaler.n_features_in_:
            logger.error(f"Feature count mismatch: selected features ({len(final_selected_features)}) != scaler features ({scaler.n_features_in_})")
            raise ValueError("Feature count mismatch between selected features and scaler")
        
        return model, scaler, final_selected_features, best_params
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Check if we want to run backtesting or live trading
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--live":
            logger.info("Starting Dead Zone Strategy in live trading mode...")
            
            # Load model artifacts for live trading
            try:
                tuned_model, scaler, final_selected_features, best_params = load_model_artifacts()
                logger.info("Successfully loaded model artifacts for live trading")
            except Exception as e:
                logger.error(f"Failed to load model artifacts: {str(e)}")
                logger.info("Running backtesting first to prepare model...")
                
                # Run backtesting to prepare model
                print("--- 1. Data Acquisition and Initial Preparation ---")
                data_full = fetch_historical_data(
                    client=client,
                    symbol=SYMBOL,
                    exchange=EXCHANGE,
                    interval=INTERVAL,
                    start_date=START_DATE,
                    end_date=END_DATE_TEST
                )
                
                data_full = prepare_target_variable(data_full, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
                
                print("\n--- 2. Feature Engineering ---")
                data_full, initial_features = engineer_features(data_full)
                
                print("\n--- 3. Train/Validation/Test Split ---")
                X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = split_data(data_full, initial_features, END_DATE_TRAIN_VAL)
                
                print("\n--- 4. Feature Scaling ---")
                X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler = scale_features(X_train, X_val, X_test)
                
                print("\n--- 5. Feature Selection Funnelling Approach ---")
                X_train_final, X_val_final, X_test_final, final_selected_features = select_features(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train)
                
                print("\n--- 6. Model Building and Hyperparameter Tuning ---")
                tuned_model, best_params = tune_and_train_model(X_train_final, X_val_final, y_train, y_val, RANDOM_SEED, N_OPTUNA_TRIALS, OPTUNA_CV_SPLITS)
                
                # Save model artifacts
                save_model_artifacts(tuned_model, scaler, final_selected_features, best_params)
            
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
            # Run the existing backtesting code
            print("--- 1. Data Acquisition and Initial Preparation ---")
            data_full = fetch_historical_data(
                client=client,
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=INTERVAL,
                start_date=START_DATE,
                end_date=END_DATE_TEST
            )
            
            data_full = prepare_target_variable(data_full, DEAD_ZONE_UPPER, DEAD_ZONE_LOWER)
            
            print("\n--- 2. Feature Engineering ---")
            data_full, initial_features = engineer_features(data_full)
            
            print("\n--- 3. Train/Validation/Test Split ---")
            X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = split_data(data_full, initial_features, END_DATE_TRAIN_VAL)
            
            print("\n--- 4. Feature Scaling ---")
            X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, scaler = scale_features(X_train, X_val, X_test)
            
            print("\n--- 5. Feature Selection Funnelling Approach ---")
            X_train_final, X_val_final, X_test_final, final_selected_features = select_features(X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train)
            
            print("\n--- 6. Model Building and Hyperparameter Tuning ---")
            tuned_model, best_params = tune_and_train_model(X_train_final, X_val_final, y_train, y_val, RANDOM_SEED, N_OPTUNA_TRIALS, OPTUNA_CV_SPLITS)
            
            print("\n--- 7. Model Evaluation on Test Set ---")
            eval_result = evaluate_model(tuned_model, X_test_final, y_test, display_plot=False)
            
            print("\n--- 8. Backtesting the Predicted Signals ---")
            backtest_results = run_backtest(test_df, eval_result.get("probabilities"), SYMBOL, BACKTEST_PROB_THRESHOLD, RISK_FREE_RATE_ANNUAL, display_plot=False)
            
            # Save model artifacts after successful backtesting
            save_model_artifacts(tuned_model, scaler, final_selected_features, best_params)
            
            logger.info("Backtesting completed successfully")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        stop_event.set()
        exit_all_positions()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)



