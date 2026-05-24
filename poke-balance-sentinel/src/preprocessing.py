from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
TYPE_COLS = ["type_1", "type_2"]


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        if df[col].isna().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return df


def _validate_base_stats(df: pd.DataFrame) -> None:
    missing = [c for c in BASE_STATS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base stat columns: {missing}")


def _normalize_type_value(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "none": "", "Unknown": "", "unknown": ""})
    )


def fit_type_columns(df: pd.DataFrame) -> list[str]:
    if "type_1" not in df.columns:
        raise ValueError("type_1 column is required.")
    temp = df.copy()
    temp["type_1"] = _normalize_type_value(temp["type_1"])
    temp["type_2"] = _normalize_type_value(temp["type_2"]) if "type_2" in temp.columns else ""

    all_types = sorted(set(temp["type_1"].unique()) | set(temp["type_2"].unique()))
    all_types = [t for t in all_types if t]
    return [f"type_{t.lower()}" for t in all_types]


def encode_types(df: pd.DataFrame, expected_type_cols: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    if "type_1" not in df.columns:
        raise ValueError("type_1 column is required.")

    df["type_1"] = _normalize_type_value(df["type_1"])
    df["type_2"] = _normalize_type_value(df["type_2"]) if "type_2" in df.columns else ""

    if expected_type_cols is None:
        expected_type_cols = fit_type_columns(df)

    raw_types = [c.replace("type_", "") for c in expected_type_cols]

    for t in raw_types:
        col = f"type_{t}"
        df[col] = (
            (df["type_1"].str.lower() == t) |
            (df["type_2"].str.lower() == t)
        ).astype(int)

    df["num_types"] = ((df["type_1"] != "").astype(int) + (df["type_2"] != "").astype(int)).clip(upper=2)

    return df.drop(columns=[c for c in TYPE_COLS if c in df.columns], errors="ignore")


def _fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[BASE_STATS])
    return scaler


def _apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[BASE_STATS] = scaler.transform(df[BASE_STATS])
    return df


def save_artifact(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path):
    return joblib.load(path)


def preprocess_for_training(
    input_path: str | Path,
    scaler_path: str | Path = "models/scaler.joblib",
    feature_mode: Literal["balance", "longevity"] = "balance",
) -> tuple[pd.DataFrame, StandardScaler, list[str] | None]:
    df = load_data(input_path)
    df = clean_data(df)
    _validate_base_stats(df)

    scaler = _fit_scaler(df)
    save_artifact(scaler, scaler_path)

    df = _apply_scaler(df, scaler)

    if feature_mode == "balance":
        X = df[BASE_STATS].copy()
        return X, scaler, None

    if feature_mode == "longevity":
        expected_type_cols = fit_type_columns(df)
        df = encode_types(df, expected_type_cols=expected_type_cols)
        X = df[BASE_STATS + ["num_types"] + expected_type_cols].copy()
        return X, scaler, expected_type_cols

    raise ValueError("feature_mode must be 'balance' or 'longevity'")


def preprocess_for_inference(
    input_data: str | Path | pd.DataFrame,
    scaler_path: str | Path = "models/scaler.joblib",
    feature_mode: Literal["balance", "longevity"] = "balance",
    type_cols_path: str | Path = "models/type_columns.joblib",
) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = load_data(input_data)

    df = clean_data(df)
    _validate_base_stats(df)

    scaler = load_artifact(scaler_path)
    df = _apply_scaler(df, scaler)

    if feature_mode == "balance":
        return df[BASE_STATS].copy()

    if feature_mode == "longevity":
        expected_type_cols = load_artifact(type_cols_path)
        df = encode_types(df, expected_type_cols=expected_type_cols)
        return df[BASE_STATS + ["num_types"] + expected_type_cols].copy()

    raise ValueError("feature_mode must be 'balance' or 'longevity'")