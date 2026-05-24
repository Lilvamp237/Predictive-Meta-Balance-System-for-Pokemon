from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def save_model(model, model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(model_path: str | Path):
    return joblib.load(model_path)



# Balance Risk: Unsupervised

def train_balance_risk_model(
    X_train,
    *,
    n_clusters: int = 3,
    random_state: int = 42,
) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    model.fit(X_train)
    return model


def map_balance_clusters(model: KMeans, X: pd.DataFrame) -> dict[int, str]:
    centers = pd.DataFrame(model.cluster_centers_, columns=X.columns)
    center_strength = centers.mean(axis=1).sort_values()

    ordered_clusters = center_strength.index.tolist()
    label_map = {
        ordered_clusters[0]: "underpowered",
        ordered_clusters[1]: "normal",
        ordered_clusters[2]: "overpowered",
    }
    return label_map


def predict_balance_risk(model: KMeans, X: pd.DataFrame, label_map: dict[int, str]) -> np.ndarray:
    clusters = model.predict(X)
    return np.array([label_map[c] for c in clusters])



# Longevity: Regression

def train_random_forest(
    X_train,
    y_train,
    *,
    n_estimators: int = 300,
    random_state: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}