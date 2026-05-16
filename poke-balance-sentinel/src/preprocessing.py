from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _to_snake_case(name: str) -> str:
    name = str(name)
    name = name.strip()
    name = name.replace("\n", " ")
    # replace non-alphanumeric with underscore
    name = "".join([c if c.isalnum() else "_" for c in name])
    # collapse underscores
    name = "_".join([p for p in name.split("_") if p != ""])
    return name.lower()


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values for numerical columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill missing for object/categorical columns with mode or 'Unknown'
    obj_cols = df.select_dtypes(include=[object]).columns.tolist()
    for col in obj_cols:
        if df[col].isna().any():
            mode = df[col].mode()
            if len(mode) > 0 and pd.notna(mode.iloc[0]):
                df[col] = df[col].fillna(mode.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")

    return df


def _ensure_base_stat_total(df: pd.DataFrame) -> None:
    required = {"hp", "attack", "defense", "sp_attack", "sp_defense", "speed"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing base stat columns: {missing}")
    if "base_stat_total" not in df.columns:
        df["base_stat_total"] = df["hp"] + df["attack"] + df["defense"] + df["sp_attack"] + df["sp_defense"] + df["speed"]


def encode_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure type columns exist
    if "type_1" not in df.columns:
        raise ValueError("type_1 column required for type encoding")
    if "type_2" not in df.columns:
        df["type_2"] = ""

    # Normalize and fill
    df["type_1"] = df["type_1"].astype(str).fillna("Unknown").str.strip()
    df["type_2"] = df["type_2"].astype(str).fillna("").str.strip()

    all_types = set(df["type_1"].unique()) | set(df["type_2"].unique())
    # remove empty/unknown
    all_types = {t for t in all_types if t and str(t).lower() not in {"nan", "unknown", "none"}}

    for t in sorted(all_types):
        col = f"type_{str(t).lower()}"
        df[col] = ((df["type_1"] == t) | (df["type_2"] == t)).astype(int)

    # Drop original type columns
    df = df.drop(columns=[c for c in ["type_1", "type_2"] if c in df.columns])
    return df


def scale_base_stats(df: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    _ensure_base_stat_total(df)
    stat_cols = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    for c in stat_cols:
        if c not in df.columns:
            raise ValueError(f"Missing stat column: {c}")

    if scaler is None:
        scaler = StandardScaler()
        df[stat_cols] = scaler.fit_transform(df[stat_cols])
    else:
        df[stat_cols] = scaler.transform(df[stat_cols])

    return df, scaler


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def generate_final_dataset(
    input_path: str | Path = "data/pokemon_complete_2025.csv",
    output_path: str | Path = "data/final_processed_dataset.csv",
    scaler_path: str | Path = "models/scaler.joblib",
) -> pd.DataFrame:
    """
    Generate one final processed dataset for the whole team.

    Steps:
    - Load raw data
    - Clean missing values and duplicates
    - Feature engineer types into one-hot
    - Scale base stats (hp, attack, defense, sp_attack, sp_defense, speed)
    - Keep only numeric columns and the one-hot type columns
    - Ensure no missing values and numeric-only columns
    - Save final CSV and scaler
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    scaler_path = Path(scaler_path)

    # Load
    df = load_data(input_path)

    # Clean
    df = clean_data(df)

    # Ensure base stat total exists
    _ensure_base_stat_total(df)

    # Encode types
    df = encode_types(df)

    # Scale base stats (fit scaler)
    df, scaler = scale_base_stats(df, scaler=None)

    # Select only numeric columns (keep one-hot ints and numeric features)
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Ensure no missing values remain (fill with median as final safety)
    for col in numeric_df.columns:
        if numeric_df[col].isna().any():
            numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

    # Clean column names to snake_case
    numeric_df.columns = [_to_snake_case(c) for c in numeric_df.columns]

    # Save final CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_df.to_csv(output_path, index=False)

    # Save scaler
    save_scaler(scaler, scaler_path)

    return numeric_df


if __name__ == "__main__":
    generate_final_dataset()
