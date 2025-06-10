"""
Enhanced Dead Zone Live Trading Strategy for OpenAlgo

This strategy uses machine learning (XGBoost) to predict market movements based on technical indicators.
It implements a "dead zone" approach where trades are only executed when the model has high confidence
in predicting moves beyond a certain threshold.

Features:
- Supports any NIFTY 50 or BANKNIFTY symbol
- Uses CNC product type for delivery trading
- Automatic model training and retraining
- Comprehensive technical indicators
- Risk management with position tracking
- WebSocket real-time price updates
- Configurable parameters
"""

import threading
import time
import pandas as pd
import numpy as np
import pandas_ta_remake as ta
from quantmod.indicators import BBands
from datetime import datetime, timedelta
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from openalgo import api
from nifty_symbols import NIFTY50_SYMBOLS, BANKNIFTY_SYMBOLS, get_symbol_info, validate_symbol

# Initialize OpenAlgo client
client = api(
    api_key="917dd42d55f63ae8f00117abfbe5b05465fc3bd76a3efbee3be7c085df0be579",
    host="http://127.0.0.1:5000",
    ws_url="ws://127.0.0.1:8765"
)

# Strategy Configuration
class StrategyConfig:
    """Configuration class for the Dead Zone strategy"""
    
    def __init__(self, symbol="RELIANCE"):
        # Validate symbol
        if not validate_symbol(symbol):
            print(f"Warning: {symbol} not in supported symbols list")
        
        # Basic Configuration
        self.STRATEGY_NAME = f"DeadZone_ML_{symbol}"
        self.SYMBOL = symbol
        symbol_info = get_symbol_info(symbol)
        self.EXCHANGE = symbol_info["exchange"]
        self.QUANTITY = 1
        self.PRODUCT = "CNC"  # Cash and Carry for equity delivery
        self.PRICE_TYPE = "MARKET"
        
        # Dead Zone Strategy Parameters
        self.DEAD_ZONE_LOWER = -0.0010  # -0.10%
        self.DEAD_ZONE_UPPER = 0.0010   # +0.10%
        self.SIGNAL_PROBABILITY_THRESHOLD = 0.60  # Probability threshold to trigger trades
        self.DAYS_OF_HISTORY = 100  # Days of historical data for feature calculation
        self.RETRAIN_FREQUENCY_DAYS = 7  # Retrain model every 7 days
        
        # Model Configuration
        self.MODEL_DIR = "models"
        self.MODEL_FILE = f"{self.MODEL_DIR}/{symbol}_dead_zone_model.pkl"
        self.SCALER_FILE = f"{self.MODEL_DIR}/{symbol}_dead_zone_scaler.pkl"
        self.FEATURES_FILE = f"{self.MODEL_DIR}/{symbol}_dead_zone_features.pkl"
        self.CONFIG_FILE = f"{self.MODEL_DIR}/{symbol}_dead_zone_config.json"
        
        # XGBoost Model Parameters
        self.XGB_PARAMS = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
        
        # Risk Management
        self.MAX_POSITION_SIZE = 1  # Maximum position size
        self.ENABLE_POSITION_TRACKING = True
        
        # Timing Configuration
        self.SIGNAL_CHECK_INTERVAL = 1800  # Check for signals every 30 minutes
        self.MARKET_START_HOUR = 9
        self.MARKET_START_MINUTE = 15
        self.MARKET_END_HOUR = 15
        self.MARKET_END_MINUTE = 30
    
    def save_config(self):
        """Save configuration to file"""
        config_dict = {
            'symbol': self.SYMBOL,
            'dead_zone_lower': self.DEAD_ZONE_LOWER,
            'dead_zone_upper': self.DEAD_ZONE_UPPER,
            'signal_threshold': self.SIGNAL_PROBABILITY_THRESHOLD,
            'days_of_history': self.DAYS_OF_HISTORY,
            'retrain_frequency': self.RETRAIN_FREQUENCY_DAYS,
            'xgb_params': self.XGB_PARAMS,
            'last_updated': datetime.now().isoformat()
        }
        
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r') as f:
                config_dict = json.load(f)
                
                # Update parameters from saved config
                self.DEAD_ZONE_LOWER = config_dict.get('dead_zone_lower', self.DEAD_ZONE_LOWER)
                self.DEAD_ZONE_UPPER = config_dict.get('dead_zone_upper', self.DEAD_ZONE_UPPER)
                self.SIGNAL_PROBABILITY_THRESHOLD = config_dict.get('signal_threshold', self.SIGNAL_PROBABILITY_THRESHOLD)
                self.DAYS_OF_HISTORY = config_dict.get('days_of_history', self.DAYS_OF_HISTORY)
                self.RETRAIN_FREQUENCY_DAYS = config_dict.get('retrain_frequency', self.RETRAIN_FREQUENCY_DAYS)
                self.XGB_PARAMS.update(config_dict.get('xgb_params', {}))
                
                print(f"Loaded configuration for {self.SYMBOL}")


class DeadZoneStrategy:
    """Dead Zone Machine Learning Trading Strategy"""
    
    def __init__(self, config):
        self.config = config
        self.ltp = None
        self.in_position = False
        self.current_position = None
        self.current_quantity = 0
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.last_model_update = None
        self.stop_event = threading.Event()
        
        # Load existing configuration
        self.config.load_config()
        
        print(f"Dead Zone Strategy initialized for {config.SYMBOL}")
        print(f"Exchange: {config.EXCHANGE}, Product: {config.PRODUCT}")
        print(f"Dead Zone: {config.DEAD_ZONE_LOWER:.3f} to {config.DEAD_ZONE_UPPER:.3f}")
        print(f"Signal Threshold: {config.SIGNAL_PROBABILITY_THRESHOLD:.3f}")
    
    def is_market_hours(self):
        """Check if current time is within market hours"""
        now = datetime.now()
        market_start = now.replace(hour=self.config.MARKET_START_HOUR, 
                                 minute=self.config.MARKET_START_MINUTE, 
                                 second=0, microsecond=0)
        market_end = now.replace(hour=self.config.MARKET_END_HOUR, 
                               minute=self.config.MARKET_END_MINUTE, 
                               second=0, microsecond=0)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        is_weekday = now.weekday() < 5
        is_market_time = market_start <= now <= market_end
        
        return is_weekday and is_market_time
    
    def on_data_received(self, data):
        """WebSocket LTP handler"""
        if data.get("type") == "market_data" and data.get("symbol") == self.config.SYMBOL:
            self.ltp = float(data["data"]["ltp"])
            print(f"LTP Update {self.config.EXCHANGE}:{self.config.SYMBOL} => ₹{self.ltp}")
    
    def websocket_thread(self):
        """WebSocket thread for real-time price updates"""
        try:
            instrument = [{"exchange": self.config.EXCHANGE, "symbol": self.config.SYMBOL}]
            client.connect()
            client.subscribe_ltp(instrument, on_data_received=self.on_data_received)
            print("WebSocket LTP thread started.")
            while not self.stop_event.is_set():
                time.sleep(1)
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            print("Shutting down WebSocket...")
            try:
                instrument = [{"exchange": self.config.EXCHANGE, "symbol": self.config.SYMBOL}]
                client.unsubscribe_ltp(instrument)
                client.disconnect()
            except:
                pass
            print("WebSocket connection closed.")
    
    def validate_features(self, X):
        """Validate and clean features before processing"""
        # Handle infinite values
        if np.any(np.isinf(X)):
            print("Warning: Infinite values found in features")
            X = np.nan_to_num(X, nan=np.nan, posinf=0, neginf=0)
        
        # Handle NaN values
        if np.any(np.isnan(X)):
            print("Warning: NaN values found in features")
            col_means = np.nanmean(X, axis=0)
            col_means = np.nan_to_num(col_means, nan=0)  # Replace any remaining NaN with 0
            X = np.nan_to_num(X, nan=col_means)
        
        return X
    
    def engineer_features(self, df):
        """Engineer comprehensive technical features"""
        print("Engineering features...")
        
        try:
            # Basic price features
            df['Return'] = df['close'].pct_change()
            df['H-L'] = df['high'] - df['low']
            df['C-O'] = df['close'] - df['open']
            df['Amplitude'] = (df['high'] - df['low']) / df['close'].shift(1).clip(lower=0.01)
            df['Difference'] = (df['close'] - df['open']) / df['close'].shift(1).clip(lower=0.01)
            
            # Bollinger Bands with error handling
            try:
                bb_lower, bb_middle, bb_upper = BBands(df['close'], lookback=20)
                df['BB_lower'] = bb_lower
                df['BB_middle'] = bb_middle
                df['BB_upper'] = bb_upper
                df['BB_width'] = df['BB_upper'] - df['BB_lower']
                df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']).clip(lower=0.01)
            except Exception as e:
                print(f"Error with BBands, using SMA fallback: {e}")
                sma20 = df['close'].rolling(20).mean()
                std20 = df['close'].rolling(20).std()
                df['BB_lower'] = sma20 - (2 * std20)
                df['BB_middle'] = sma20
                df['BB_upper'] = sma20 + (2 * std20)
                df['BB_width'] = df['BB_upper'] - df['BB_lower']
                df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']).clip(lower=0.01)
            
            # Lagged returns
            for lag in [1, 2, 3, 5, 10]:
                df[f'Return_lag{lag}'] = df['Return'].shift(lag)
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'SMA{period}'] = ta.sma(df['close'], length=period)
                df[f'EMA{period}'] = ta.ema(df['close'], length=period)
                df[f'Close_vs_SMA{period}'] = (df['close'] - df[f'SMA{period}']) / df[f'SMA{period}'].clip(lower=0.01)
                df[f'Close_vs_EMA{period}'] = (df['close'] - df[f'EMA{period}']) / df[f'EMA{period}'].clip(lower=0.01)
            
            # MA crossovers
            df['SMA10_vs_SMA20'] = (df['SMA10'] - df['SMA20']) / df['SMA20'].clip(lower=0.01)
            df['EMA10_vs_EMA20'] = (df['EMA10'] - df['EMA20']) / df['EMA20'].clip(lower=0.01)
            
            # Volatility indicators
            df['ATR14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['ATR_normalized'] = df['ATR14'] / df['close'].clip(lower=0.01)
            df['StdDev20_Return'] = df['Return'].rolling(window=20).std()
            
            # Momentum indicators
            df['RSI14'] = ta.rsi(df['close'], length=14)
            df['RSI_normalized'] = (df['RSI14'] - 50) / 50  # Normalize RSI around 0
            
            # MACD with error handling
            try:
                macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if isinstance(macd_result, tuple) and len(macd_result) == 3:
                    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd_result
                else:
                    # Handle single DataFrame return
                    df['MACD'] = macd_result['MACD_12_26_9']
                    df['MACD_signal'] = macd_result['MACDs_12_26_9']
                    df['MACD_hist'] = macd_result['MACDh_12_26_9']
            except Exception as e:
                print(f"Error calculating MACD: {e}")
                df['MACD'] = 0
                df['MACD_signal'] = 0
                df['MACD_hist'] = 0
            
            # Stochastic
            try:
                stoch_result = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
                if isinstance(stoch_result, tuple):
                    df['Stochastic_K'], df['Stochastic_D'] = stoch_result
                else:
                    df['Stochastic_K'] = stoch_result['STOCHk_14_3_3']
                    df['Stochastic_D'] = stoch_result['STOCHd_14_3_3']
            except Exception as e:
                print(f"Error calculating Stochastic: {e}")
                df['Stochastic_K'] = 50
                df['Stochastic_D'] = 50
            
            # Williams %R
            df['Williams%R14'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # Rate of Change
            df['ROC10'] = df['close'].pct_change(periods=10)
            
            # Volume indicators
            df['OBV'] = ta.obv(df['close'], df['volume'])
            df['Volume_SMA5'] = ta.sma(df['volume'], length=5)
            df['Volume_ratio'] = df['volume'] / df['Volume_SMA5'].clip(lower=1)
            
            # Price position indicators
            df['High_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).clip(lower=0.01)
            
            # Momentum
            df['Momentum10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10).clip(lower=0.01)
            
            # Time-based features
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['DayOfMonth'] = df.index.day
            
            print(f"Feature engineering completed. Features: {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            raise
    
    def get_historical_data(self, days=None):
        """Get historical data for analysis"""
        if days is None:
            days = self.config.DAYS_OF_HISTORY
        
        print(f"Fetching {days} days of historical data for {self.config.SYMBOL}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer for indicators
        
        try:
            df = client.history(
                symbol=self.config.SYMBOL,
                exchange=self.config.EXCHANGE,
                interval="D",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            if isinstance(df, dict):
                print(f"Error fetching data: {df}")
                return None
            
            if df.empty:
                print("No historical data available")
                return None
            
            print(f"Retrieved {len(df)} days of historical data")
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def prepare_training_data(self, df):
        """Prepare training data with labels"""
        print("Preparing training data...")
        
        # Calculate target return (next day's return)
        df['TargetReturn'] = df['Return'].shift(-1)
        
        # Define target variable based on dead zone
        df['Uptrend'] = np.nan
        df.loc[df['TargetReturn'] > self.config.DEAD_ZONE_UPPER, 'Uptrend'] = 1
        df.loc[df['TargetReturn'] < self.config.DEAD_ZONE_LOWER, 'Uptrend'] = 0
        
        # Remove dead zone samples
        df = df.dropna(subset=['Uptrend'])
        df['Uptrend'] = df['Uptrend'].astype(int)
        
        print(f"Training data shape: {df.shape}")
        if len(df) > 0:
            print(f"Uptrend distribution:\n{df['Uptrend'].value_counts(normalize=True)}")
        
        return df
    
    def train_model(self):
        """Train the machine learning model"""
        print("Training/retraining the machine learning model...")
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        # Get historical data
        df = self.get_historical_data()
        if df is None:
            print("Failed to get historical data for training")
            return False
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare training data
        df = self.prepare_training_data(df)
        
        if len(df) < 50:
            print("Insufficient data for model training")
            return False
        
        # Select feature columns
        exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'Return', 'TargetReturn', 'Uptrend']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Clean data
        df = df.dropna(subset=feature_columns)
        
        if len(df) < 30:
            print("Insufficient clean data for model training")
            return False
        
        X = df[feature_columns]
        y = df['Uptrend']
        
        # Validate and clean features
        X_clean = self.validate_features(X.values)
        X_clean_df = pd.DataFrame(X_clean, columns=feature_columns, index=X.index)
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean_df)
        
        # Simple feature selection - remove highly correlated features
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
        corr_matrix = X_scaled_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        self.selected_features = [col for col in feature_columns if col not in to_drop]
        print(f"Selected {len(self.selected_features)} features after correlation filtering")
        
        X_final = X_scaled_df[self.selected_features]
        
        # Handle class imbalance
        class_counts = y.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1
        
        # Update XGB parameters with calculated scale_pos_weight
        xgb_params = self.config.XGB_PARAMS.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        
        # Train model
        self.model = XGBClassifier(**xgb_params)
        self.model.fit(X_final, y)
        
        # Save model components
        try:
            with open(self.config.MODEL_FILE, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.config.SCALER_FILE, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(self.config.FEATURES_FILE, 'wb') as f:
                pickle.dump(self.selected_features, f)
            
            # Save configuration
            self.config.save_config()
            
            self.last_model_update = datetime.now()
            print(f"Model trained and saved successfully at {self.last_model_update}")
            
            # Print model performance summary
            train_score = self.model.score(X_final, y)
            print(f"Training accuracy: {train_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_files = [self.config.MODEL_FILE, self.config.SCALER_FILE, self.config.FEATURES_FILE]
            if all(os.path.exists(f) for f in model_files):
                with open(self.config.MODEL_FILE, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.config.SCALER_FILE, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(self.config.FEATURES_FILE, 'rb') as f:
                    self.selected_features = pickle.load(f)
                
                self.last_model_update = datetime.fromtimestamp(os.path.getmtime(self.config.MODEL_FILE))
                print(f"Model loaded successfully. Last updated: {self.last_model_update}")
                return True
            else:
                print("Model files not found. Training new model...")
                return self.train_model()
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            return self.train_model()
    
    def should_retrain_model(self):
        """Check if model should be retrained"""
        if self.last_model_update is None:
            return True
        
        days_since_update = (datetime.now() - self.last_model_update).days
        return days_since_update >= self.config.RETRAIN_FREQUENCY_DAYS
    
    def get_prediction_signal(self):
        """Get trading signal from ML model"""
        if self.model is None or self.scaler is None or self.selected_features is None:
            print("Model not loaded or trained")
            return None
        
        try:
            # Get recent data for prediction
            df = self.get_historical_data(days=70)  # Enough for indicators
            if df is None or df.empty:
                print("Error getting data for prediction")
                return None
            
            # Engineer features
            df = self.engineer_features(df)
            
            # Get latest data point
            latest_data = df.iloc[-1:][self.selected_features]
            
            # Validate and clean
            X_clean = self.validate_features(latest_data.values)
            
            # Scale features
            X_scaled = self.scaler.transform(X_clean)
            
            # Get prediction probability
            prob = self.model.predict_proba(X_scaled)[0][1]  # Probability of uptrend
            
            print(f"ML Prediction Probability (Uptrend): {prob:.3f}")
            print(f"Signal Threshold: {self.config.SIGNAL_PROBABILITY_THRESHOLD:.3f}")
            
            # Generate signal
            if prob >= self.config.SIGNAL_PROBABILITY_THRESHOLD:
                return "BUY", prob
            elif prob <= (1 - self.config.SIGNAL_PROBABILITY_THRESHOLD):
                return "SELL", prob
            else:
                return None, prob  # Dead zone
                
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return None, 0
    
    def get_current_position(self):
        """Get current position from broker"""
        try:
            response = client.openposition(
                strategy=self.config.STRATEGY_NAME,
                symbol=self.config.SYMBOL,
                exchange=self.config.EXCHANGE,
                product=self.config.PRODUCT
            )
            
            if response.get("status") == "success":
                quantity = int(response.get("quantity", 0))
                self.current_quantity = quantity
                return quantity
            else:
                self.current_quantity = 0
                return 0
                
        except Exception as e:
            print(f"Error getting position: {e}")
            self.current_quantity = 0
            return 0
    
    def place_order(self, action):
        """Place trading order"""
        print(f"Placing {action} order for {self.config.SYMBOL}")
        
        try:
            response = client.placeorder(
                strategy=self.config.STRATEGY_NAME,
                symbol=self.config.SYMBOL,
                exchange=self.config.EXCHANGE,
                action=action,
                price_type=self.config.PRICE_TYPE,
                product=self.config.PRODUCT,
                quantity=self.config.QUANTITY
            )
            
            print(f"Order Response: {response}")
            
            if response.get("status") == "success":
                order_id = response.get("orderid")
                print(f"Order placed successfully. Order ID: {order_id}")
                
                # Update position state
                if action == "BUY":
                    self.current_quantity += self.config.QUANTITY
                elif action == "SELL":
                    self.current_quantity -= self.config.QUANTITY
                
                self.in_position = self.current_quantity != 0
                return True
            else:
                print(f"Order failed: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"Error placing order: {e}")
            return False
    
    def close_position(self):
        """Close all positions"""
        try:
            response = client.closeposition(strategy=self.config.STRATEGY_NAME)
            print(f"Close Position Response: {response}")
            
            if response.get("status") == "success":
                print("Position closed successfully")
                self.in_position = False
                self.current_position = None
                self.current_quantity = 0
                return True
            else:
                print(f"Failed to close position: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"Error closing position: {e}")
            return False
    
    def strategy_thread(self):
        """Main strategy logic thread"""
        print("Strategy thread started...")
        
        # Load or train model
        if not self.load_model():
            print("Failed to load/train model. Exiting strategy.")
            return
        
        while not self.stop_event.is_set():
            try:
                # Only trade during market hours
                if not self.is_market_hours():
                    print("Outside market hours. Waiting...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Check if model needs retraining
                if self.should_retrain_model():
                    print("Retraining model...")
                    self.train_model()
                
                # Get current position
                current_qty = self.get_current_position()
                self.in_position = current_qty != 0
                
                if current_qty > 0:
                    self.current_position = "LONG"
                elif current_qty < 0:
                    self.current_position = "SHORT"
                else:
                    self.current_position = None
                
                print(f"Current position: {self.current_position} (Qty: {current_qty})")
                
                # Get ML prediction
                signal_result = self.get_prediction_signal()
                if len(signal_result) == 2:
                    signal, prob = signal_result
                else:
                    signal, prob = None, 0
                
                if signal:
                    print(f"ML Signal: {signal} (Confidence: {prob:.3f})")
                    
                    # Position management logic
                    if not self.in_position:
                        # No position, can open new one
                        self.place_order(signal)
                    else:
                        # Have position, check if need to close/reverse
                        if (self.current_position == "LONG" and signal == "SELL") or \
                           (self.current_position == "SHORT" and signal == "BUY"):
                            print("Signal opposite to current position. Closing first.")
                            if self.close_position():
                                time.sleep(5)  # Wait for position to close
                                self.place_order(signal)
                else:
                    print(f"No signal (prob: {prob:.3f}) - in dead zone")
                
            except Exception as e:
                print(f"Error in strategy thread: {e}")
            
            # Wait before next check
            time.sleep(self.config.SIGNAL_CHECK_INTERVAL)
    
    def start(self):
        """Start the strategy"""
        print(f"Starting Dead Zone ML Strategy for {self.config.SYMBOL}...")
        print(f"Product: {self.config.PRODUCT}, Price Type: {self.config.PRICE_TYPE}")
        print(f"Dead Zone: [{self.config.DEAD_ZONE_LOWER:.3f}, {self.config.DEAD_ZONE_UPPER:.3f}]")
        print(f"Signal Threshold: {self.config.SIGNAL_PROBABILITY_THRESHOLD:.3f}")
        
        # Start threads
        ws_thread = threading.Thread(target=self.websocket_thread)
        strat_thread = threading.Thread(target=self.strategy_thread)
        
        ws_thread.start()
        strat_thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Shutting down...")
            self.stop_event.set()
            ws_thread.join()
            strat_thread.join()
            print("Strategy shutdown complete.")


def main():
    """Main function to run the strategy"""
    # Configuration
    SYMBOL = "RELIANCE"  # Change this to any NIFTY 50 or BANKNIFTY symbol
    
    # Validate symbol
    if not validate_symbol(SYMBOL):
        print(f"Error: {SYMBOL} is not in the supported symbols list.")
        print("Supported symbols:")
        print("NIFTY 50:", NIFTY50_SYMBOLS[:10], "...")
        print("BANKNIFTY:", BANKNIFTY_SYMBOLS[:5], "...")
        return
    
    # Create configuration
    config = StrategyConfig(symbol=SYMBOL)
    
    # Create and start strategy
    strategy = DeadZoneStrategy(config)
    strategy.start()


if __name__ == "__main__":
    main() 