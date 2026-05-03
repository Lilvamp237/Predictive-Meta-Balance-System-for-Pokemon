# Preprocessing Pipeline - Optimization Summary

## Overview

The preprocessing pipeline has been optimized for **training and inference consistency**. The key improvements enable:

1. **Scaler Persistence** - Fit scaler once, reuse forever
2. **Consistency Guarantee** - Same preprocessing applied identically across datasets
3. **Production Ready** - Designed for ML model deployment
4. **Backwards Compatible** - Original `train.py` still works unchanged

---

## Key Improvements

### 1. Scaler Persistence Functions

**`save_scaler(scaler, path)`**
- Persists fitted scaler to disk using joblib
- Enables scaler reuse across training/inference sessions
- Path: `models/scaler.joblib` (recommended)

**`load_scaler(path)`**
- Loads previously saved scaler from disk
- Ensures identical feature scaling in inference
- Raises error if file not found

### 2. Workflow Functions

**`preprocess_for_training(train_path, scaler_path, output_path=None)`**
- Complete training workflow in one function
- Fits scaler on training data and saves it
- Returns: (X_train, y_train, scaler)
- Replaces multiple manual steps

**`preprocess_for_inference(inference_path, scaler_path, output_path=None)`**
- Complete inference workflow in one function  
- Loads and applies saved scaler
- Extracts same features as training (14 columns)
- Returns: (X_test, scaler)

### 3. Enhanced Pipeline

**`preprocess_pipeline(data_path, output_path=None, scaler=None)`**
- Core orchestration function
- Detects whether fitting or transforming based on scaler parameter
- Provides informative logging about mode (training vs inference)
- Returns all 49 features (before feature selection)

---

## Usage Patterns

### Pattern 1: Simple Training (Recommended)

```python
from src.preprocessing import preprocess_for_training, preprocess_for_inference
import joblib

# Training
X_train, y_train, scaler = preprocess_for_training(
    'data/train.csv',
    'models/scaler.joblib'
)

model = RandomForestRegressor()
model.fit(X_train, y_train)
joblib.dump(model, 'models/model.joblib')

# Inference
X_test, _ = preprocess_for_inference(
    'data/test.csv',
    'models/scaler.joblib'
)

predictions = model.predict(X_test)
```

### Pattern 2: Manual Control

```python
from src.preprocessing import (
    load_data, clean_data, encode_types, 
    scale_features, select_features,
    save_scaler, load_scaler,
    build_training_data, add_power_creep_index
)

# Training: fit scaler
df_train = load_data('data/train.csv')
df_train = clean_data(df_train)
df_train = encode_types(df_train)
df_train, scaler = scale_features(df_train)  # scaler=None fits new
df_train = select_features(df_train)
save_scaler(scaler, 'models/scaler.joblib')

# Inference: use saved scaler
scaler = load_scaler('models/scaler.joblib')
df_test = load_data('data/test.csv')
df_test = clean_data(df_test)
df_test = encode_types(df_test)
df_test, _ = scale_features(df_test, scaler=scaler)
df_test = select_features(df_test)
```

### Pattern 3: Split Train/Test from Single Dataset

```python
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data

# Split the data
raw_df = load_data('data/pokemon.csv')
train_indices = raw_df.sample(frac=0.8, random_state=42).index
test_indices = raw_df.drop(train_indices).index

raw_df.loc[train_indices].to_csv('data/train_split.csv')
raw_df.loc[test_indices].to_csv('data/test_split.csv')

# Process with same scaler
X_train, y_train, scaler = preprocess_for_training(
    'data/train_split.csv',
    'models/scaler.joblib'
)

X_test, _ = preprocess_for_inference(
    'data/test_split.csv', 
    'models/scaler.joblib'
)
```

---

## API Reference

### Core Functions

| Function | Purpose | Training | Inference |
|----------|---------|----------|-----------|
| `load_data(path)` | Load and validate | ✓ | ✓ |
| `clean_data(df)` | Handle missing values | ✓ | ✓ |
| `encode_types(df)` | One-hot encode types | ✓ | ✓ |
| `scale_features(df, scaler=None)` | Scale numericals | ✓ | ✓ |
| `select_features(df)` | Select 49 features | ✓ | ✓ |
| `prepare_features(df)` | Extract 14 features | ✓ | - |
| `build_training_data(df)` | Get X, y for training | ✓ | - |

### High-Level Functions

| Function | Purpose |
|----------|---------|
| `preprocess_pipeline(...)` | Run all steps (configurable) |
| `preprocess_for_training(...)` | Complete training workflow |
| `preprocess_for_inference(...)` | Complete inference workflow |

### Scaler Management

| Function | Purpose |
|----------|---------|
| `save_scaler(scaler, path)` | Persist scaler |
| `load_scaler(path)` | Load scaler |

### Target Variable

| Function | Purpose |
|----------|---------|
| `add_power_creep_index(df)` | Calculate target variable |

---

## Important Concepts

### Why Scaler Consistency Matters

The model learns on **scaled features** with specific mean and standard deviation:

```
Model expects: hp_scaled = (hp - mean) / std
              where mean and std are from training data

If you use different scaler:
              hp_scaled = (hp - different_mean) / different_std
              
Result: Model receives unexpected feature values → Poor predictions
```

### Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Scaler | Fit new on training data | Load from training |
| Preprocessing | Identical | Identical |
| Feature count | 14 (after selection) | 14 (after selection) |
| Data size | Arbitrary | Arbitrary |

### Feature Columns Used for Training

The model uses exactly these 14 columns:

```python
FEATURE_COLUMNS = [
    "generation_num",
    "num_types", 
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
    "base_stat_total",
    "height_m",
    "weight_kg",
    "is_legendary",
    "is_mythical",
    "is_baby",
]
```

All other columns (except target) are removed after preprocessing.

---

## Common Pitfalls to Avoid

### ❌ WRONG: Fitting scaler on test data
```python
scaler = StandardScaler()
scaler.fit(X_test)  # DON'T DO THIS!
```
Causes data leakage and inflated performance.

### ❌ WRONG: Using different scalers for train and test
```python
scaler_train = StandardScaler().fit(X_train)
scaler_test = StandardScaler().fit(X_test)  # DIFFERENT!
```
Different mean/std will confuse the model.

### ✅ RIGHT: Fit once, use everywhere
```python
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
save_scaler(scaler, 'models/scaler.joblib')
```

### ✅ RIGHT: Use high-level functions
```python
X_train, y_train, scaler = preprocess_for_training(...)
X_test, _ = preprocess_for_inference(...)
```

---

## Verification

### Check Scaler Consistency

```python
from src.preprocessing import load_scaler

scaler = load_scaler('models/scaler.joblib')
print(f"Mean: {scaler.mean_}")
print(f"Scale (std): {scaler.scale_}")

# These should be identical every time you load
```

### Check Feature Columns Match

```python
from src.preprocessing import FEATURE_COLUMNS

print(f"Expected {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}")
print(f"Actual X_train shape: {X_train.shape}")
assert X_train.shape[1] == len(FEATURE_COLUMNS)
```

### Check Scaled Features

```python
# For scaled features, mean ≈ 0 and std ≈ 1
print(f"HP mean: {X_train['hp'].mean():.6f}")  # Should be ≈ 0
print(f"HP std: {X_train['hp'].std():.6f}")    # Should be ≈ 1
```

---

## Files Generated

After running preprocessing:

```
data/
  ├── processed_pokemon.csv           # Full preprocessing output (49 cols)
  ├── pokemon_train_processed.csv     # Training data (49 cols)
  └── pokemon_test_processed.csv      # Test data (49 cols)

models/
  ├── pokemon_scaler.joblib           # Fitted scaler (required for inference)
  ├── pokemon_model.joblib            # Trained model
  └── random_forest_pci.joblib        # Original trained model
```

---

## Integration with train.py

The original `train.py` continues to work unchanged:

```bash
python train.py --test-size 0.2 --n-estimators 100
```

This uses the original `load_data()` and `build_training_data()` functions,
which are fully compatible with the optimized pipeline.

---

## Summary

The optimized preprocessing pipeline provides:

✅ **Fit once, use forever** - Scaler persistence  
✅ **Guaranteed consistency** - Same preprocessing always  
✅ **Production ready** - Designed for deployment  
✅ **Backward compatible** - Original pipeline still works  
✅ **Easy to use** - High-level workflow functions  
✅ **Fine-grained control** - Core functions available  

Use `preprocess_for_training()` and `preprocess_for_inference()` for best practices.
