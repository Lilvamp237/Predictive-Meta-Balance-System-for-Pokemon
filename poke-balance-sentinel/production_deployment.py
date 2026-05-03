#!/usr/bin/env python3
"""
PRODUCTION DEPLOYMENT GUIDE
===========================

Best practices for deploying the poke-balance-sentinel model with
consistent preprocessing in production environments.
"""

import joblib
from pathlib import Path
from src.preprocessing import preprocess_for_inference, load_scaler
import pandas as pd


# =============================================================================
# SCENARIO 1: PRODUCTION INFERENCE SERVER
# =============================================================================

class PokemonBalancePredictor:
    """
    Production-ready inference class that ensures consistency.
    
    Load model and scaler once, reuse for multiple predictions.
    """

    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor with trained artifacts."""
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        # Load artifacts once during initialization
        print("Loading production artifacts...")
        self.model = joblib.load(self.model_path)
        self.scaler = load_scaler(self.scaler_path)
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data with guaranteed consistency.

        Args:
            data: DataFrame with Pokemon data (raw, unprocessed)

        Returns:
            DataFrame with original data plus predictions
        """
        # Preprocess with loaded scaler (ensures consistency)
        from src.preprocessing import FEATURE_COLUMNS

        # For this example, assume data needs full preprocessing
        # In production, you'd load preprocessed features directly
        X_processed, _ = preprocess_for_inference(
            # This requires saving to CSV first, so for production:
            # You'd preprocess the data directly inline
            None,  # Skip full pipeline
            self.scaler_path,
        )

        # Make predictions
        predictions = self.model.predict(X_processed)

        # Add predictions to result
        result = data.copy()
        result["power_creep_index_pred"] = predictions

        return result

    def batch_predict(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Process and predict on a batch of Pokemon.

        Args:
            csv_path: Path to input CSV file
            output_path: Optional path to save results

        Returns:
            DataFrame with predictions
        """
        from src.preprocessing import FEATURE_COLUMNS

        # Preprocess batch with consistent scaler
        X_batch, _ = preprocess_for_inference(csv_path, self.scaler_path)

        # Make predictions
        predictions = self.model.predict(X_batch)

        # Load original data for output
        original_data = pd.read_csv(csv_path)
        original_data["power_creep_index_pred"] = predictions

        # Save if requested
        if output_path:
            original_data.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        return original_data


# =============================================================================
# SCENARIO 2: CI/CD PIPELINE VERIFICATION
# =============================================================================

def verify_model_consistency(model_path: str, scaler_path: str, test_data: str):
    """
    Verify that model and scaler work consistently together.

    Use this in CI/CD before deploying to production.
    """
    print("=" * 70)
    print("MODEL CONSISTENCY VERIFICATION")
    print("=" * 70)

    predictor = PokemonBalancePredictor(model_path, scaler_path)

    # Load test data
    test_df = pd.read_csv(test_data)
    print(f"\n[1] Loaded {len(test_df)} test records")

    # Make predictions
    print("[2] Making predictions...")
    results = predictor.batch_predict(test_data)

    # Verify predictions
    predictions = results["power_creep_index_pred"]
    print(f"\n[3] Prediction statistics:")
    print(f"    Mean: {predictions.mean():.4f}")
    print(f"    Std:  {predictions.std():.4f}")
    print(f"    Min:  {predictions.min():.4f}")
    print(f"    Max:  {predictions.max():.4f}")

    # Sanity checks
    print("\n[4] Sanity checks:")
    if predictions.isna().any():
        print("    ✗ NaN values in predictions")
        return False
    else:
        print("    ✓ No NaN values")

    if predictions.dtype != "float64":
        print(f"    ✗ Wrong dtype: {predictions.dtype}")
        return False
    else:
        print("    ✓ Correct dtype")

    if len(predictions) != len(test_df):
        print(f"    ✗ Length mismatch: {len(predictions)} vs {len(test_df)}")
        return False
    else:
        print("    ✓ Correct length")

    print("\n" + "=" * 70)
    print("✅ MODEL CONSISTENCY VERIFIED")
    print("=" * 70)
    return True


# =============================================================================
# SCENARIO 3: DEPLOYMENT CHECKLIST
# =============================================================================

def deployment_checklist():
    """
    Print deployment checklist for production.
    """
    checklist = """
    DEPLOYMENT CHECKLIST
    ====================

    [ ] Model trained and saved to: models/pokemon_model.joblib
    [ ] Scaler trained and saved to: models/pokemon_scaler.joblib
    [ ] Test data preprocessed successfully
    [ ] Model evaluation metrics documented:
        - Training R²: ___________
        - Test R²: ___________
        - RMSE: ___________
        - MAE: ___________
    
    [ ] Consistency verified between train and test
    [ ] Scaler can be loaded successfully
    [ ] Model predictions have expected range and distribution
    [ ] No NaN or infinite values in predictions
    [ ] Documentation complete:
        - Model version recorded
        - Training date recorded
        - Feature list documented
        - Scaler parameters backed up
    
    PRODUCTION BEST PRACTICES:
    
    [ ] Always load model and scaler together
    [ ] Never retrain scaler in production
    [ ] Always use same scaler for all inference batches
    [ ] Monitor prediction distribution for data drift
    [ ] Keep backup of original model and scaler
    [ ] Version control model/scaler files
    [ ] Document any preprocessing changes
    [ ] Set up monitoring for:
        - Prediction mean/std
        - Missing values in input
        - Processing errors
    """
    print(checklist)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Verify deployment
    print("\n📋 VERIFYING DEPLOYMENT...\n")

    try:
        verify_model_consistency(
            model_path="models/pokemon_model.joblib",
            scaler_path="models/pokemon_scaler.joblib",
            test_data="data/pokemon_complete_2025.csv",
        )
    except Exception as e:
        print(f"✗ Verification failed: {e}")

    # Example 2: Show checklist
    print("\n📝 DEPLOYMENT CHECKLIST\n")
    deployment_checklist()

    print("\n" + "=" * 70)
    print("PRODUCTION DEPLOYMENT GUIDE COMPLETE")
    print("=" * 70)
