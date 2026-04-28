from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.models import evaluate_regressor, save_model, train_random_forest
from src.preprocessing import build_training_data, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PCI Random Forest model.")
    parser.add_argument(
        "--data",
        type=str,
        default="data\\pokemon_complete_2025.csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models\\random_forest_pci.joblib",
        help="Path to save the trained model.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    df = load_data(data_path)
    X, y = build_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    metrics = evaluate_regressor(model, X_test, y_test)
    save_model(model, args.model_out)

    print("Training complete.")
    print(
        f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}"
    )
    print(f"Saved model to: {args.model_out}")


if __name__ == "__main__":
    main()
