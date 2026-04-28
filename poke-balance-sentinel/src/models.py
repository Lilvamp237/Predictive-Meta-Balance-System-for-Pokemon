from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def evaluate_regressor(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_model(model, model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(model_path: str | Path):
    return joblib.load(model_path)
