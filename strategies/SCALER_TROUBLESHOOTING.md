# Scaler Troubleshooting Guide

This guide explains how to handle scaler warnings and feature count mismatches in the Dead Zone ML Strategy.

## Common Issues and Solutions

### 1. Scaler Warnings: "X has feature names that are all strings"

**Problem**: The scaler is being fit on numpy arrays instead of pandas DataFrames with feature names.

**Solution**: Always fit the scaler on pandas DataFrames with feature names:

```python
# ✅ Correct way
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)  # X_train_df is a DataFrame

# ❌ Incorrect way
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_array)  # X_train_array is a numpy array
```

**Updated Code**: The `scale_features()` function now ensures scalers are always fit on DataFrames with feature names.

### 2. Feature Count Mismatches

**Problem**: The number of features in your data doesn't match what the scaler expects.

**Causes**:
- Feature engineering parameters changed
- Different stocks have different available features
- Missing features in new data

**Solutions**:

#### A. Automatic Feature Alignment
The `safe_scaler_transform()` function automatically handles feature mismatches:

```python
# Automatically aligns features and fills missing ones with zeros
features_scaled_df = safe_scaler_transform(scaler, features_df, expected_features)
```

#### B. Manual Feature Alignment
Use the `align_features()` function:

```python
# Align features to match expected set
aligned_df = align_features(df, expected_features, fill_method='zero')
```

#### C. Clean Up Old Artifacts
When feature engineering parameters change, clean up old models:

```python
# Check compatibility
if not check_feature_engineering_compatibility():
    # Clean up old artifacts
    cleanup_old_artifacts()
```

### 3. Feature Engineering Parameter Changes

**Problem**: You changed feature engineering parameters (correlation threshold, VIF threshold, etc.) but old models are incompatible.

**Solution**: The strategy now automatically detects parameter changes and cleans up old artifacts:

```python
# This happens automatically in main execution
if not check_feature_engineering_compatibility():
    print("Feature engineering parameters have changed. Cleaning up old artifacts...")
    cleanup_old_artifacts()
    print("Old artifacts cleaned up. Will train new models.")
```

## Best Practices

### 1. Always Use DataFrames with Feature Names

```python
# ✅ Good
scaler.fit(X_train_df)  # DataFrame with column names

# ❌ Bad
scaler.fit(X_train_array)  # Numpy array without feature names
```

### 2. Validate Feature Counts

```python
# Check feature counts match
if len(features) != scaler.n_features_in_:
    raise ValueError(f"Feature count mismatch: {len(features)} != {scaler.n_features_in_}")
```

### 3. Use Safe Transform Functions

```python
# Use safe transform that handles mismatches
features_scaled = safe_scaler_transform(scaler, features_df, expected_features)
```

### 4. Clean Up When Parameters Change

```python
# Always check compatibility before loading models
if not check_feature_engineering_compatibility():
    cleanup_old_artifacts()
```

## Debugging Steps

### Step 1: Check Feature Engineering Parameters

```python
current_params = {
    'CORRELATION_THRESHOLD': CORRELATION_THRESHOLD,
    'VIF_THRESHOLD': VIF_THRESHOLD,
    'N_UNIVARIATE_FEATURES': N_UNIVARIATE_FEATURES,
    'N_XGB_IMPORTANCE_FEATURES': N_XGB_IMPORTANCE_FEATURES,
    'RFECV_MIN_FEATURES': RFECV_MIN_FEATURES
}
print("Current parameters:", current_params)
```

### Step 2: Check Scaler Properties

```python
print(f"Scaler n_features_in_: {scaler.n_features_in_}")
print(f"Scaler feature_names_in_: {scaler.feature_names_in_}")
```

### Step 3: Check Data Features

```python
print(f"Data features count: {len(features_df.columns)}")
print(f"Data features: {features_df.columns.tolist()}")
```

### Step 4: Use Test Script

Run the test script to verify functionality:

```bash
python test_artifact_cleanup.py
```

## Common Error Messages and Solutions

### Error: "X has 10 features, but StandardScaler is expecting 15 features"

**Solution**: Use feature alignment:
```python
aligned_df = align_features(df, expected_features, fill_method='zero')
```

### Warning: "X has feature names that are all strings"

**Solution**: Ensure scaler is fit on DataFrame:
```python
scaler.fit(X_train_df)  # Use DataFrame, not numpy array
```

### Error: "Feature count mismatch between selected features and scaler"

**Solution**: Clean up old artifacts and retrain:
```python
cleanup_old_artifacts()
# Retrain model
```

## Testing Your Setup

Run the comprehensive test script:

```bash
python test_artifact_cleanup.py
```

This will test:
- ✅ Artifact cleanup functionality
- ✅ Feature engineering compatibility checking
- ✅ Feature alignment
- ✅ Safe scaler transform

## Summary

The updated strategy now includes:

1. **Automatic scaler fitting on DataFrames** - Prevents feature name warnings
2. **Robust feature alignment** - Handles missing/extra features automatically
3. **Parameter compatibility checking** - Detects when feature engineering changes
4. **Automatic artifact cleanup** - Removes incompatible models
5. **Safe transform functions** - Handle edge cases gracefully

These improvements ensure the strategy works reliably across different stocks and parameter configurations. 