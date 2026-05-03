# Quick Reference: Optimized Preprocessing Pipeline

## For Users Coming from Original Pipeline

### Old Way (Still Works)
```python
from src.preprocessing import load_data, build_training_data
df = load_data('data/pokemon.csv')
X, y = build_training_data(df)
```

### New Way (Recommended)
```python
from src.preprocessing import preprocess_for_training, preprocess_for_inference

# Training
X_train, y_train, scaler = preprocess_for_training(
    'data/train.csv',
    'models/scaler.joblib'
)
model.fit(X_train, y_train)

# Inference (later)
X_test, _ = preprocess_for_inference(
    'data/test.csv',
    'models/scaler.joblib'
)
predictions = model.predict(X_test)
```

---

## API Summary

### NEW: High-Level Workflows (Recommended)

| Function | Purpose | Returns |
|----------|---------|---------|
| `preprocess_for_training(train_path, scaler_path, output_path=None)` | Fit & save scaler, prep training data | (X, y, scaler) |
| `preprocess_for_inference(test_path, scaler_path, output_path=None)` | Load scaler, prep inference data | (X, scaler) |

### NEW: Scaler Management

| Function | Purpose |
|----------|---------|
| `save_scaler(scaler, path)` | Save fitted scaler to disk |
| `load_scaler(path)` | Load saved scaler from disk |

### EXISTING: Core Functions (Still Available)

| Function | Purpose |
|----------|---------|
| `load_data(path)` | Load and validate dataset |
| `clean_data(df)` | Handle missing values |
| `encode_types(df)` | One-hot encode types |
| `scale_features(df, scaler=None)` | Scale with StandardScaler |
| `select_features(df)` | Select 49 features |
| `build_training_data(df)` | Extract 14 features + target |

---

## Common Tasks

### Save Model & Scaler After Training
```python
import joblib
joblib.dump(model, 'models/model.joblib')
save_scaler(scaler, 'models/scaler.joblib')
```

### Load for Inference
```python
model = joblib.load('models/model.joblib')
scaler = load_scaler('models/scaler.joblib')
X = preprocess_for_inference('data/test.csv', 'models/scaler.joblib')[0]
predictions = model.predict(X)
```

### Verify Consistency
```python
# All should produce identical results
scaler1 = load_scaler('models/scaler.joblib')
scaler2 = load_scaler('models/scaler.joblib')
assert (scaler1.mean_ == scaler2.mean_).all()  # ✓ Identical
```

---

## Key Principles

1. **Fit Once** - Scaler fitted only on training data
2. **Reuse Always** - Same scaler for all inference
3. **Save Always** - Persist scaler to disk
4. **Load Always** - Load saved scaler for inference
5. **Never Retrain** - Don't fit scaler in production

---

## File Structure

```
poke-balance-sentinel/
├── src/
│   └── preprocessing.py         ← All functions here
├── models/
│   ├── pokemon_scaler.joblib    ← IMPORTANT: Save this
│   ├── pokemon_model.joblib     ← Trained model
│   └── random_forest_pci.joblib ← Backup
├── data/
│   ├── pokemon_complete_2025.csv     ← Raw data
│   ├── processed_pokemon.csv         ← Full preprocessing
│   ├── pokemon_train_processed.csv   ← Training set
│   └── pokemon_test_processed.csv    ← Test set
└── train.py                     ← Original training script
```

---

## For Production Deployment

1. **Train & Save**
   ```python
   X, y, scaler = preprocess_for_training('train.csv', 'scaler.joblib')
   model.fit(X, y)
   joblib.dump(model, 'model.joblib')
   ```

2. **Test & Verify**
   ```python
   X_test, _ = preprocess_for_inference('test.csv', 'scaler.joblib')
   score = model.score(X_test, y_test)
   assert score > 0.99  # Sanity check
   ```

3. **Deploy**
   - Keep `model.joblib` and `scaler.joblib` together
   - Always load both together
   - Never retrain scaler

---

## Features Used for Training

Model expects exactly these 14 columns (all from FEATURE_COLUMNS):
```
generation_num, num_types, hp, attack, defense,
sp_attack, sp_defense, speed, base_stat_total,
height_m, weight_kg, is_legendary, is_mythical, is_baby
```

Other columns removed automatically during preprocessing.

---

## Documentation

- **`PREPROCESSING_GUIDE.py`** - Detailed usage examples
- **`OPTIMIZATION_SUMMARY.md`** - Technical optimization details
- **`production_deployment.py`** - Production best practices
- **`README_OPTIMIZATION.md`** - Complete summary
- **`example_train_and_predict.py`** - Working example

---

## Troubleshooting

### "Scaler file not found"
```python
from pathlib import Path
assert Path('models/scaler.joblib').exists()  # Check file exists
scaler = load_scaler('models/scaler.joblib')  # Load it
```

### "Feature mismatch" error
The inference data must have been preprocessed with same scaler:
```python
# Always use same function and scaler
X_test, _ = preprocess_for_inference('test.csv', 'models/scaler.joblib')
# Don't mix preprocessors or scalers
```

### "NaN values in predictions"
Check that preprocessing succeeded:
```python
import numpy as np
assert not np.isnan(X_test).any()  # No NaN in features
assert len(X_test) == 14  # Correct number of features
```

---

## Summary

✅ Scaler persistence - Fit once, use forever  
✅ Consistency guarantee - Identical preprocessing  
✅ Easy to use - High-level functions  
✅ Production safe - Built-in checks  
✅ Backward compatible - Original still works  

**Ready for deployment!**
