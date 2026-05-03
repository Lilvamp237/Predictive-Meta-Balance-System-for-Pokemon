"""
PREPROCESSING PIPELINE USAGE GUIDE
===================================

This guide demonstrates how to use the optimized preprocessing pipeline
for training, inference, and model deployment scenarios.
"""

# =============================================================================
# SCENARIO 1: TRAINING A NEW MODEL (Fit Scaler)
# =============================================================================

"""
Use preprocess_for_training() when:
- Building a new ML model
- Need to fit a new scaler on training data
- Want to save the scaler for later inference

This workflow:
1. Loads and preprocesses training data
2. Fits a new scaler on the training features
3. Saves the scaler to disk for reuse
4. Returns features and target for model training
"""

from src.preprocessing import preprocess_for_training, load_data, build_training_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Option A: Using the convenience function
X_train, y_train, scaler = preprocess_for_training(
    train_data_path='data/pokemon_train.csv',
    scaler_output_path='models/scaler.joblib',
    processed_data_output_path='data/pokemon_train_processed.csv'
)

# Now train your model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
joblib.dump(model, 'models/model.joblib')

# =============================================================================
# SCENARIO 2: INFERENCE ON NEW DATA (Use Saved Scaler)
# =============================================================================

"""
Use preprocess_for_inference() when:
- Making predictions on new/test data
- Have already trained a model
- Have a saved scaler from training

This workflow:
1. Loads the saved scaler
2. Preprocesses new data using the saved scaler
3. Returns features for prediction
"""

from src.preprocessing import preprocess_for_inference

# Load saved model and scaler
model = joblib.load('models/model.joblib')
X_test, _ = preprocess_for_inference(
    inference_data_path='data/pokemon_test.csv',
    scaler_path='models/scaler.joblib',
    processed_data_output_path='data/pokemon_test_processed.csv'
)

# Make predictions
predictions = model.predict(X_test)

# =============================================================================
# SCENARIO 3: MANUAL CONTROL WITH CORE FUNCTIONS
# =============================================================================

"""
Use the core functions directly when you need more control over
individual preprocessing steps or custom workflows.
"""

from src.preprocessing import (
    load_data,
    clean_data,
    encode_types,
    scale_features,
    select_features,
    save_scaler,
    load_scaler,
)
import pandas as pd

# TRAINING: Fit scaler
print("Training preprocessing...")
df = load_data('data/pokemon_train.csv')
df = clean_data(df)
df = encode_types(df)
df_scaled, scaler = scale_features(df)  # scaler = None means FIT new scaler
df_final = select_features(df_scaled)
save_scaler(scaler, 'models/scaler.joblib')

# INFERENCE: Use saved scaler
print("Inference preprocessing...")
scaler = load_scaler('models/scaler.joblib')
df = load_data('data/pokemon_test.csv')
df = clean_data(df)
df = encode_types(df)
df_scaled, _ = scale_features(df, scaler=scaler)  # Use fitted scaler
df_final = select_features(df_scaled)

# =============================================================================
# SCENARIO 4: SPLIT TRAIN/TEST AND PROCESS BOTH
# =============================================================================

"""
When starting with a single dataset, split it first, then:
1. Process training set (fit scaler)
2. Save scaler
3. Process test set (use same scaler)
"""

from sklearn.model_selection import train_test_split

# Load raw data
raw_df = load_data('data/pokemon_complete_2025.csv')

# Split into train and test
train_indices = raw_df.sample(frac=0.8, random_state=42).index
test_indices = raw_df.drop(train_indices).index

train_df = raw_df.loc[train_indices]
test_df = raw_df.loc[test_indices]

# Save splits temporarily
train_df.to_csv('data/train_split.csv', index=False)
test_df.to_csv('data/test_split.csv', index=False)

# Process with consistent scaler
X_train, y_train, scaler = preprocess_for_training(
    'data/train_split.csv',
    'models/scaler.joblib'
)

X_test_df, _ = preprocess_for_inference(
    'data/test_split.csv',
    'models/scaler.joblib'
)

# =============================================================================
# SCENARIO 5: BATCH PREDICTION / DEPLOYMENT
# =============================================================================

"""
In production, load the model and scaler once, then use them for
predictions on multiple batches without refitting.
"""

import joblib
from src.preprocessing import preprocess_for_inference

# Load production artifacts (once)
model = joblib.load('models/model.joblib')
scaler_path = 'models/scaler.joblib'

def batch_predict(data_path):
    """Process a batch of Pokemon data and make predictions."""
    # Always use the same scaler path for consistency
    X_batch, _ = preprocess_for_inference(data_path, scaler_path)
    predictions = model.predict(X_batch)
    return predictions

# Use for multiple batches
predictions_batch1 = batch_predict('data/batch1.csv')
predictions_batch2 = batch_predict('data/batch2.csv')

# =============================================================================
# KEY CONCEPTS
# =============================================================================

"""
SCALER PERSISTENCE:
- The scaler learns mean and standard deviation from training data
- Must be saved after training and loaded during inference
- Ensures consistent feature scaling across all datasets

WHY CONSISTENCY MATTERS:
- Features scaled with different scaler mean/std will mislead the model
- The model expects features in the specific scale it was trained on
- Using different scalers for test data leads to poor predictions

WORKFLOW SUMMARY:
1. TRAINING:
   - Fit scaler on training data
   - Train model on scaled training data
   - Save both model and scaler

2. INFERENCE:
   - Load saved scaler
   - Apply saved scaler to new data
   - Load saved model
   - Make predictions

3. TESTING:
   - Use same scaler as training
   - Preprocess test data identically
   - Evaluate model on consistently scaled data
"""

# =============================================================================
# COMMON PITFALLS TO AVOID
# =============================================================================

"""
❌ WRONG: Fitting scaler on test data
  - scaler = StandardScaler()
  - scaler.fit(X_test)  # DON'T DO THIS
  - Causes data leakage and inflated performance metrics

❌ WRONG: Using different scaler for train and test
  - Different mean/std will confuse the model
  - Model expects specific scale from training

✅ RIGHT: Fit once on training, use everywhere
  - scaler = StandardScaler()
  - scaler.fit(X_train)
  - X_train_scaled = scaler.transform(X_train)
  - X_test_scaled = scaler.transform(X_test)
  - Save scaler for production

✅ RIGHT: Use preprocess_for_training() and preprocess_for_inference()
  - Handles all the details automatically
  - Enforces best practices
  - Ensures consistency
"""

# =============================================================================
# VERIFYING CONSISTENCY
# =============================================================================

"""
How to verify that training and inference use the same scaler:
"""

from src.preprocessing import load_scaler

# Load the scaler
scaler = load_scaler('models/scaler.joblib')

# Check scaler parameters
print(f"Scaler mean: {scaler.mean_}")
print(f"Scaler scale (std): {scaler.scale_}")

# These values should be identical every time you load the scaler
# If they're different, you're using a different scaler

# =============================================================================
# SUMMARY OF AVAILABLE FUNCTIONS
# =============================================================================

"""
HIGH-LEVEL (Recommended for most use cases):
  - preprocess_for_training(train_path, scaler_path, output_path)
    Fit scaler and prepare training data
  
  - preprocess_for_inference(inference_path, scaler_path, output_path)
    Use saved scaler for inference

CORE FUNCTIONS (Fine-grained control):
  - load_data(path): Load and validate dataset
  - clean_data(df): Handle missing values and duplicates
  - encode_types(df): One-hot encode Pokemon types
  - scale_features(df, scaler=None): Scale numerical features
  - select_features(df): Select relevant columns
  - preprocess_pipeline(path, scaler=None): Run all steps

SCALER MANAGEMENT:
  - save_scaler(scaler, path): Persist scaler to disk
  - load_scaler(path): Load saved scaler

TRAINING PIPELINE:
  - build_training_data(df): Extract features and target for training
  - add_power_creep_index(df): Calculate target variable
"""
