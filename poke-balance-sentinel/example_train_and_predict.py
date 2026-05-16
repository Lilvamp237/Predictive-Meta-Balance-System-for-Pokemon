"""
Example: Complete ML Pipeline with Scaler Persistence
=======================================================

Demonstrates best practices for:
1. Training a model with preprocessing
2. Saving scaler for inference
3. Making predictions on new data
4. Ensuring consistent preprocessing
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

from src.preprocessing import (
    preprocess_for_training,
    preprocess_for_inference,
    load_data,
    build_training_data,
)


def train_pipeline_with_scaler():
    """
    Complete training pipeline that saves scaler for inference.
    """
    print("=" * 80)
    print("TRAINING PIPELINE WITH SCALER PERSISTENCE")
    print("=" * 80)

    # Paths for artifacts
    scaler_path = "models/pokemon_scaler.joblib"
    model_path = "models/pokemon_model.joblib"
    train_processed_path = "data/pokemon_train_processed.csv"

    # Step 1: Preprocess training data (fit scaler)
    print("\n[1] Preprocessing training data...")
    X_train, y_train, scaler = preprocess_for_training(
        train_data_path="data/pokemon_complete_2025.csv",
        scaler_output_path=scaler_path,
        processed_data_output_path=train_processed_path,
    )

    print(f"\n✓ Training data prepared:")
    print(f"  Features: {X_train.shape}")
    print(f"  Target: {y_train.shape}")
    print(f"  Scaler saved to: {scaler_path}")

    # Step 2: Train model
    print("\n[2] Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on training data
    train_score = model.score(X_train, y_train)
    print(f"  Training R² score: {train_score:.4f}")

    # Step 3: Save model
    print("\n[3] Saving model...")
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")

    return scaler_path, model_path


def inference_pipeline_with_scaler(scaler_path, model_path, test_data_path):
    """
    Complete inference pipeline using saved scaler.
    """
    print("\n" + "=" * 80)
    print("INFERENCE PIPELINE WITH SAVED SCALER")
    print("=" * 80)

    # Step 1: Load model and scaler
    print("\n[1] Loading model and scaler...")
    model = joblib.load(model_path)
    test_processed_path = "data/pokemon_test_processed.csv"

    print(f"  Model loaded from: {model_path}")
    print(f"  Scaler will be loaded from: {scaler_path}")

    # Step 2: Preprocess test data (use saved scaler)
    print("\n[2] Preprocessing test data with saved scaler...")
    X_test, _ = preprocess_for_inference(
        inference_data_path=test_data_path,
        scaler_path=scaler_path,
        processed_data_output_path=test_processed_path,
    )

    print(f"\n✓ Test data prepared:")
    print(f"  Features: {X_test.shape}")

    # Step 3: Make predictions
    print("\n[3] Making predictions...")
    predictions = model.predict(X_test)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Mean prediction: {predictions.mean():.4f}")

    return predictions


def demonstrate_consistency():
    """
    Demonstrate that preprocessing is consistent across train/test.
    """
    print("\n" + "=" * 80)
    print("CONSISTENCY VERIFICATION")
    print("=" * 80)

    from src.preprocessing import load_scaler
    import pandas as pd

    # Load preprocessed data
    train_df = pd.read_csv("data/pokemon_train_processed.csv")
    test_df = pd.read_csv("data/pokemon_test_processed.csv")

    print("\n[1] Checking preprocessed data shapes...")
    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  ✓ Same columns: {train_df.shape[1] == test_df.shape[1]}")

    print("\n[2] Checking numerical feature ranges...")
    print("  Feature statistics:")
    print(f"  {'Feature':<15} {'Train Mean':>12} {'Test Mean':>12} {'Train Std':>10} {'Test Std':>10}")
    print("  " + "-" * 60)

    numerical_cols = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    for col in numerical_cols:
        if col in train_df.columns and col in test_df.columns:
            train_mean = train_df[col].mean()
            train_std = train_df[col].std()
            test_mean = test_df[col].mean()
            test_std = test_df[col].std()
            print(f"  {col:<15} {train_mean:>12.3f} {test_mean:>12.3f} {train_std:>10.3f} {test_std:>10.3f}")

    print("\n[3] Verifying scaler...")
    scaler = load_scaler("models/pokemon_scaler.joblib")
    print(f"  Scaler mean (first 3): {scaler.mean_[:3]}")
    print(f"  Scaler scale (first 3): {scaler.scale_[:3]}")


def main():
    """
    Run complete example.
    """
    # Create models directory if needed
    Path("models").mkdir(exist_ok=True)

    # Step 1: Train with scaler
    print("\n📊 STEP 1: MODEL TRAINING")
    scaler_path, model_path = train_pipeline_with_scaler()

    # Step 2: Inference with saved scaler
    print("\n\n🔮 STEP 2: MODEL INFERENCE")
    predictions = inference_pipeline_with_scaler(
        scaler_path, model_path, "data/pokemon_complete_2025.csv"
    )

    # Step 3: Verify consistency
    print("\n\n✅ STEP 3: CONSISTENCY CHECK")
    demonstrate_consistency()

    print("\n" + "=" * 80)
    print("✅ COMPLETE EXAMPLE FINISHED")
    print("=" * 80)
    print("\nKey artifacts created:")
    print(f"  • Model: {model_path}")
    print(f"  • Scaler: {scaler_path}")
    print(f"  • Processed data: data/pokemon_*_processed.csv")
    print("\nNext steps:")
    print("  1. Use the saved scaler for any new predictions")
    print("  2. Share the model and scaler for deployment")
    print("  3. Always use the same scaler for consistency")


if __name__ == "__main__":
    main()
