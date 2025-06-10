import yfinance as yf
import pandas as pd
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

# %%
# --- Configuration ---
TICKER = "^NSEI"
START_DATE = "2018-01-01"
END_DATE_TRAIN_VAL = "2022-12-31" # Training + Validation data ends here
END_DATE_TEST = pd.Timestamp.now().strftime("%Y-%m-%d")      # Test data uses current date

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
RISK_FREE_RATE_ANNUAL = 0.02 # For Sharpe Ratio calculation

# Random Seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
# --- 1. Data Acquisition and Initial Preparation ---
print("--- 1. Data Acquisition and Initial Preparation ---")
# Download data for the entire period
data_full = yf.download(TICKER, start=START_DATE, end=END_DATE_TEST, progress=False, auto_adjust=False)
data_full.dropna(inplace=True)

# Calculate daily returns
data_full['Return'] = data_full['Adj Close'].pct_change()

# Target Return is the next day's return
data_full['TargetReturn'] = data_full['Return'].shift(-1)

# Define target variable: 1 if uptrend, 0 otherwise
data_full['Uptrend'] = np.nan

# Assign labels
data_full.loc[data_full['TargetReturn'] > DEAD_ZONE_UPPER, 'Uptrend'] = 1
data_full.loc[data_full['TargetReturn'] < DEAD_ZONE_LOWER, 'Uptrend'] = 0

print(data_full.columns)

# Drop ambiguous (dead zone) samples
data_full = data_full.dropna(subset=[('Uptrend', '')])
data_full['Uptrend'] = data_full['Uptrend'].astype(int)

print(f"Full data shape: {data_full.shape}")
print(f"Uptrend distribution:\n{data_full['Uptrend'].value_counts(normalize=True)}")


# %%
# --- 2. Feature Engineering ---
print("\n--- 2. Feature Engineering ---")

# Price-based features
data_full['H-L'] = data_full['High'] - data_full['Low']
data_full['C-O'] = data_full['Close'] - data_full['Open']
data_full['Amplitude'] = (data_full['High'] - data_full['Low']) / data_full['Adj Close'].shift(1)
data_full['Difference'] = (data_full['Close'] - data_full['Open']) / data_full['Adj Close'].shift(1)
data_full['High_Low_Range'] = data_full['High'] - data_full['Low']
data_full['Open_Close_Range'] = data_full['Open'] - data_full['Close']

# Bollinger Bands
data_full['BB_lower'], data_full['BB_middle'], data_full['BB_upper'] = BBands(data_full['Adj Close'], lookback=20)
data_full['BB_width'] = data_full['BB_upper'] - data_full['BB_lower']

# Lagged Returns
for lag in [1, 2, 3, 5, 10]:
    data_full[f'Return_lag{lag}'] = data_full['Return'].shift(lag)

# Moving Averages & Differentials
for ma_period in [10, 20, 50]:
    data_full[f'SMA{ma_period}'] = ta.sma(data_full[('Adj Close', TICKER)], length=ma_period)
    data_full[f'EMA{ma_period}'] = ta.ema(data_full[('Adj Close', TICKER)], length=ma_period)
    data_full[f'Close_vs_SMA{ma_period}'] = data_full[('Adj Close', TICKER)] - data_full[f'SMA{ma_period}']
    data_full[f'Close_vs_EMA{ma_period}'] = data_full[('Adj Close', TICKER)] - data_full[f'EMA{ma_period}']

if 'SMA10' in data_full and 'SMA20' in data_full:
    data_full['SMA10_vs_SMA20'] = data_full['SMA10'] - data_full['SMA20']
if 'EMA10' in data_full and 'EMA20' in data_full:
    data_full['EMA10_vs_EMA20'] = data_full['EMA10'] - data_full['EMA20']


# Volatility Indicators
data_full['ATR14'] = ta.atr(data_full[('High', TICKER)], data_full[('Low', TICKER)], data_full[('Close', TICKER)], length=14)
data_full['StdDev20_Return'] = data_full['Return'].rolling(window=20).std()

# Momentum Indicators
data_full['RSI14'] = ta.rsi(data_full[('Adj Close', TICKER)], length=14)
macd, macdsignal, macdhist = ta.macd(data_full[('Adj Close', TICKER)], fast=12, slow=26, signal=9)
data_full['MACD'] = macd
data_full['MACD_signal'] = macdsignal
data_full['MACD_hist'] = macdhist
data_full['Momentum10'] = data_full[('Adj Close', TICKER)] - data_full[('Adj Close', TICKER)].shift(10)
data_full['Williams_%R'] = -100 * (data_full[('High', TICKER)] - data_full[('Close', TICKER)]) / (data_full[('High', TICKER)] - data_full[('Low', TICKER)])
data_full['Williams%R14'] = ta.willr(data_full[('High', TICKER)], data_full[('Low', TICKER)], data_full[('Close', TICKER)], length=14)

# Stochastic Oscillator
data_full['Stochastic_K'], data_full['Stochastic_D'] = ta.stoch(
    high=data_full[('High', TICKER)],
    low=data_full[('Low', TICKER)],
    close=data_full[('Close', TICKER)],
    k=14,  # Fast %K period
    d=3,   # Slow %K period
    smooth_k=3,  # Slow %D period
    mamode='sma'  # Moving average mode
)

# Rate of Change
data_full['ROC10'] = data_full[('Adj Close', TICKER)].pct_change(periods=10)

# On-Balance Volume
data_full['OBV'] = ta.obv(data_full[('Adj Close', TICKER)], data_full[('Volume', TICKER)])

# Volume-based features
data_full['Volume_MA5'] = ta.sma(data_full[('Volume', TICKER)], length=5)
data_full['Volume_MA20'] = ta.sma(data_full[('Volume', TICKER)], length=20)
data_full['Volume_Change'] = data_full[('Volume', TICKER)].pct_change()
if 'Volume_MA5' in data_full and 'Volume_MA20' in data_full:
    data_full['Volume_MA5_vs_MA20'] = data_full['Volume_MA5'] - data_full['Volume_MA20']

# Date/Time Features (example, often less predictive for daily returns)
data_full['DayOfWeek'] = data_full.index.dayofweek # Monday=0, Sunday=6
data_full['Month'] = data_full.index.month

# Ensure all features are numerical
for col in data_full.columns:
    if data_full[col].dtype == 'object':
        try:
            data_full[col] = pd.to_numeric(data_full[col])
        except Exception as e:
            print(f"Error converting column {col}: {e}")
            print(f"Warning: Could not convert column {col} to numeric. Dropping it.")
            data_full = data_full.drop(columns=[col])

# Drop rows with NaNs created by indicators/lags
initial_features = data_full.columns.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return', 'Uptrend'])
print(data_full.isnull().sum())
data_full.dropna(subset=initial_features, inplace=True)
print(f"Data shape after feature engineering and NaN drop: {data_full.shape}")
print(f"Number of initial features: {len(initial_features)}")


# %%
# --- 3. Train/Validation/Test Split (Time-Series Aware) ---
print("\n--- 3. Train/Validation/Test Split ---")

# Define split points
train_val_end_date = pd.to_datetime(END_DATE_TRAIN_VAL)
test_start_date = train_val_end_date + pd.Timedelta(days=1)

# For train/validation split, use approximately the last year of train_val data for validation
train_end_date = train_val_end_date - pd.DateOffset(years=1)

train_df = data_full.loc[data_full.index <= train_end_date]
val_df = data_full.loc[(data_full.index > train_end_date) & (data_full.index <= train_val_end_date)]
test_df = data_full.loc[data_full.index >= test_start_date]

X = data_full[initial_features]
y = data_full['Uptrend']

X_train, y_train = train_df[initial_features], train_df['Uptrend']
X_val, y_val = val_df[initial_features], val_df['Uptrend']
X_test, y_test = test_df[initial_features], test_df['Uptrend']

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

print(X_train.head())
print(y_train.head())
print(X_val.head())
print(y_val.head())
print(X_test.head())
print(y_test.head())

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


print("\n--- 4. Feature Scaling ---")

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

# %%
# --- 5. Feature Selection Funnelling Approach ---
print("\n--- 5. Feature Selection Funnelling Approach ---")
selected_features = list(X_train.columns) # Start with all initial features
print(selected_features)

# %%
# Step 5.1: Filter - Remove High Correlation
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

# %%
# Step 5.2: Filter - Variance Inflation Factor (VIF)
print("\nStep 5.2: VIF Filter")
features_for_vif = list(X_train_scaled_df.columns)
final_vif_features = []
dropped_vif_count = 0
while True:
    if not features_for_vif: 
        break
    vif = pd.DataFrame()
    vif["feature"] = features_for_vif
    # Add constant for VIF calculation if not already standardized around 0
    # data_for_vif_calc = sm.add_constant(X_train_scaled_df[features_for_vif], prepend=False)
    # vif["VIF"] = [variance_inflation_factor(data_for_vif_calc.values, i) for i in range(data_for_vif_calc.shape[1]-1)] # Exclude constant
    vif["VIF"] = [variance_inflation_factor(X_train_scaled_df[features_for_vif].values, i) for i in range(len(features_for_vif))]

    max_vif = vif['VIF'].max()
    if max_vif > VIF_THRESHOLD:
        feature_to_drop = vif.sort_values('VIF', ascending=False)['feature'].iloc[0]
        features_for_vif.remove(feature_to_drop)
        dropped_vif_count +=1
    else:
        final_vif_features = list(features_for_vif) # Use deep copy
        break
if dropped_vif_count > 0 :
    print(f"Dropped {dropped_vif_count} features due to VIF > {VIF_THRESHOLD}")
X_train_scaled_df = X_train_scaled_df[final_vif_features]
X_val_scaled_df = X_val_scaled_df[final_vif_features]
X_test_scaled_df = X_test_scaled_df[final_vif_features]
selected_features = list(X_train_scaled_df.columns)
print(f"Features remaining after VIF: {len(selected_features)}")
print(selected_features)

# %%
# Step 5.3: Filter - Univariate Feature Selection (ANOVA F-test)
print("\nStep 5.3: Univariate Filter (SelectKBest)")
if len(selected_features) > N_UNIVARIATE_FEATURES :
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

# %%
# Step 5.4: Embedded - Feature Importance from a base XGBoost model
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

# %%
# Step 5.5: Wrapper - Recursive Feature Elimination with Cross-Validation (RFECV)
print("\nStep 5.5: Wrapper Filter (RFECV)")
# Calculate scale_pos_weight for handling imbalance
scale_pos_weight_val = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1

estimator_rfecv = XGBClassifier(
    scale_pos_weight=scale_pos_weight_val,
    random_state=RANDOM_SEED,
    eval_metric='auc'
)
cv_splitter = TimeSeriesSplit(n_splits=OPTUNA_CV_SPLITS) # Using same splits as Optuna for consistency
# RFECV can be slow; ensure selected_features is not too large
# Or reduce cv_splitter n_splits for RFECV if it's too slow.

if len(selected_features) > RFECV_MIN_FEATURES:
    rfecv_selector = RFECV(
        estimator=estimator_rfecv,
        step=1, # remove 1 feature at a time
        cv=cv_splitter,
        scoring='roc_auc', # Optimize for AUC
        min_features_to_select=RFECV_MIN_FEATURES,
        n_jobs=-1 # Use all available cores
    )
    print("Fitting RFECV... (this might take a while)")
    rfecv_selector.fit(X_train_scaled_df, y_train)

    final_selected_features_mask = rfecv_selector.support_
    final_selected_features = X_train_scaled_df.columns[final_selected_features_mask].tolist()

    X_train_final = X_train_scaled_df[final_selected_features]
    X_val_final = X_val_scaled_df[final_selected_features]
    X_test_final = X_test_scaled_df[final_selected_features]
    print(f"Selected {len(final_selected_features)} features with RFECV.")
    print("Final selected features:", final_selected_features)
else:
    X_train_final = X_train_scaled_df.copy()
    X_val_final = X_val_scaled_df.copy()
    X_test_final = X_test_scaled_df.copy()
    final_selected_features = selected_features
    print(f"Skipping RFECV. Using current {len(final_selected_features)} features:", final_selected_features)

# %%
# --- 6. Model Building and Hyperparameter Tuning (XGBoost with Optuna) ---
print("\n--- 6. Model Building and Hyperparameter Tuning ---")

def objective(trial):
    # Recalculate scale_pos_weight inside objective if y_train can change (though it doesn't here)
    scale_pos_weight_obj = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': RANDOM_SEED,
        'scale_pos_weight': scale_pos_weight_obj,
        'n_estimators': trial.suggest_int('n_estimators', 50, 300), # Reduced upper for speed
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7), # Reduced upper for speed
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1.0), # Adjusted range
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0), # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)  # L2
    }

    model = XGBClassifier(**params)
    model.early_stopping_rounds = 15
    tscv = TimeSeriesSplit(n_splits=OPTUNA_CV_SPLITS)
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

# %%
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1) # n_jobs=1 for XGB early stopping issue with Optuna

best_params = study.best_params
print("Best hyperparameters found by Optuna:", best_params)

# %%
# Retrain model with best_params on full training data (+validation data)
# For final model evaluation, it's often good to train on train+val
X_train_val_final = pd.concat([X_train_final, X_val_final])
y_train_val = pd.concat([y_train, y_val])

final_scale_pos_weight = (len(y_train_val) - y_train_val.sum()) / y_train_val.sum() if y_train_val.sum() > 0 else 1

tuned_model = XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=RANDOM_SEED,
    scale_pos_weight=final_scale_pos_weight
)
tuned_model.fit(X_train_val_final, y_train_val) # Train on combined train+validation
print("Tuned model trained on X_train_val_final and y_train_val.")


# %%
# --- 7. Model Evaluation on Test Set ---
print("\n--- 7. Model Evaluation on Test Set ---")
y_pred_test_proba = tuned_model.predict_proba(X_test_final)[:, 1]
y_pred_test_labels = (y_pred_test_proba > 0.3).astype(int) # Using 0.3 threshold for classification metrics

print(f"Test Set ROC AUC Score: {roc_auc_score(y_test, y_pred_test_proba):.4f}")
print(f"Test Set Accuracy Score: {accuracy_score(y_test, y_pred_test_labels):.4f}")

print("\nTest Set Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Uptrend', 'Uptrend'], yticklabels=['Non-Uptrend', 'Uptrend'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Set Confusion Matrix')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()


# %%
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred_test_labels, target_names=['Non-Uptrend (0)', 'Uptrend (1)']))

# %%
# --- 8. Backtesting the Predicted Signals ---
print("\n--- 8. Backtesting the Predicted Signals ---")

# Align predictions with test data dates
backtest_df = test_df[['Open', 'Adj Close', 'Return']].copy() # Use Adj Close for returns, Open for entry/exit
backtest_df['Signal_Prob'] = y_pred_test_proba # These are probs for t+1, predicted at end of t
backtest_df['Signal'] = (backtest_df['Signal_Prob'] > BACKTEST_PROB_THRESHOLD).astype(int)

# Calculate strategy returns
# Assume trade on next day's open based on signal from previous day's close
# The 'Return' column is (Close_t / Close_t-1) -1. We need Open_t+1 to Open_t+2 returns or similar.
# For simplicity, let's assume we act on signal_t for return_t+1 (which is derived from Close_t+1)
# This is an approximation; a more precise backtest would use Open_t+1 for entry and Open_t+2 (or Close_t+1) for exit.

backtest_df['Strategy_Return'] = 0.0
# If signal is 1 (Long), capture the next day's market return.
# The 'Return' in backtest_df is for day t based on Close_t vs Close_t-1
# The 'Signal' is for day t+1 based on data from day t.
# So, if Signal_t=1, we are interested in Return_t+1
# We need to shift the signal to align with the day the return is realized.
# The y_test and y_pred_proba are for the NEXT day's uptrend.
# So, the probability at index `d` is for `d+1`'s uptrend.
# The `Return` at index `d` is the return realized on day `d`.

# Let's re-align properly:
# Signal generated at close of day t (index t) is for day t+1.
# If Signal[t] == 1, we buy at Open[t+1] and sell at Close[t+1] or Open[t+2].
# Daily return if holding from Open[t+1] to Close[t+1] is (Close[t+1]/Open[t+1]) - 1

# Shift signals by 1 to represent action taken on the day the signal applies to.
# Signal[t] is prediction for t+1. So, action based on Signal[t] occurs on day t+1.
# The Return[t+1] is (AdjClose[t+1]/AdjClose[t]) - 1.

# A common way:
# Position at day t+1 is determined by signal from day t.
# Return for strategy on day t+1 is Position[t+1] * Market_Return[t+1]
backtest_df['Position'] = backtest_df['Signal'].shift(1).fillna(0) # Position for today based on yesterday's signal
backtest_df['Strategy_Return'] = backtest_df['Position'] * backtest_df['Return'] # 'Return' is daily market return

# Buy and Hold returns
backtest_df['BH_Return'] = backtest_df['Return']

# Cumulative Returns
backtest_df['Strategy_Cumulative_Return'] = (1 + backtest_df['Strategy_Return']).cumprod() - 1
backtest_df['BH_Cumulative_Return'] = (1 + backtest_df['BH_Return']).cumprod() - 1

# Performance Metrics
strategy_total_return = backtest_df['Strategy_Cumulative_Return'].iloc[-1]
bh_total_return = backtest_df['BH_Cumulative_Return'].iloc[-1]

days_in_year = 252
strategy_annual_return = (1 + strategy_total_return)**(days_in_year / len(backtest_df)) - 1
bh_annual_return = (1 + bh_total_return)**(days_in_year / len(backtest_df)) - 1

strategy_annual_vol = backtest_df['Strategy_Return'].std() * np.sqrt(days_in_year)
bh_annual_vol = backtest_df['BH_Return'].std() * np.sqrt(days_in_year)

strategy_sharpe = (strategy_annual_return - RISK_FREE_RATE_ANNUAL) / strategy_annual_vol if strategy_annual_vol != 0 else 0
bh_sharpe = (bh_annual_return - RISK_FREE_RATE_ANNUAL) / bh_annual_vol if bh_annual_vol != 0 else 0

# Max Drawdown
def calculate_mdd(cumulative_returns_series):
    # Add 1 to make it a series of wealth index
    wealth_index = 1 + cumulative_returns_series
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()

strategy_mdd = calculate_mdd(backtest_df['Strategy_Cumulative_Return'])
bh_mdd = calculate_mdd(backtest_df['BH_Cumulative_Return'])

num_trades = backtest_df['Position'].diff().abs().sum() / 2 # Each trade is an entry and exit

# Win rate (simplified: days the strategy made positive return when in market)
positive_strategy_days = backtest_df[backtest_df['Position'] == 1]['Strategy_Return'] > 0
win_rate_days = positive_strategy_days.sum() / (backtest_df['Position'] == 1).sum() if (backtest_df['Position'] == 1).sum() > 0 else 0


print(f"\nBacktesting Results (Test Period: {test_df.index.min().date()} to {test_df.index.max().date()}):")
print(f"Trading Signal Probability Threshold: {BACKTEST_PROB_THRESHOLD}")
print(f"{'Metric':<30} | {'Trading Strategy':<20} | {f'Buy-and-Hold {TICKER}':<20}")
print("-" * 75)
print(f"{'Cumulative Return':<30} | {strategy_total_return:>19.2%} | {bh_total_return:>19.2%}")
print(f"{'Annualized Return':<30} | {strategy_annual_return:>19.2%} | {bh_annual_return:>19.2%}")
print(f"{'Annualized Volatility':<30} | {strategy_annual_vol:>19.2%} | {bh_annual_vol:>19.2%}")
print(f"{f'Sharpe Ratio (Rf={RISK_FREE_RATE_ANNUAL*100}%)':<30} | {strategy_sharpe:>19.2f} | {bh_sharpe:>19.2f}")
print(f"{'Maximum Drawdown (MDD)':<30} | {strategy_mdd:>19.2%} | {bh_mdd:>19.2%}")
print(f"{'Number of Trades (approx)':<30} | {num_trades:>19.0f} | {'1':>19}")
print(f"{'Win Rate (positive days in market)':<30} | {win_rate_days:>19.2%} | {'N/A':>19}")

# Plot cumulative returns
plt.figure(figsize=(12, 6))
backtest_df['Strategy_Cumulative_Return'].plot(label='Trading Strategy')
backtest_df['BH_Cumulative_Return'].plot(label=f'Buy-and-Hold {TICKER}')
plt.title(f'{TICKER} Backtest: Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.savefig("cumulative_returns_backtest.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n--- Script Finished ---")
print("Figures have been saved as:")
print("- cumulative_returns_backtest.png")
print("- confusion_matrix.png")

# %%
# To save dataframes for the report:
X_train_final.to_csv("X_train_final_features.csv")
y_train.to_csv("y_train.csv")
X_test_final.to_csv("X_test_final_features.csv")
y_test.to_csv("y_test.csv")
backtest_df.to_csv("backtest_results_detailed.csv")


