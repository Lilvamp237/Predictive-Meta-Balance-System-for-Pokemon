from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROMAN_TO_INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
}

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

TARGET_COLUMN = "power_creep_index"


def normalize_generation(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    value_str = str(value).strip().upper()
    if value_str.isdigit():
        return int(value_str)
    return ROMAN_TO_INT.get(value_str)


def _coerce_bool_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("bool").astype("int64")


def _ensure_base_stat_total(df: pd.DataFrame) -> None:
    if "base_stat_total" in df.columns:
        return
    required = {"hp", "attack", "defense", "sp_attack", "sp_defense", "speed"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing columns for base_stat_total: {missing}")
    df["base_stat_total"] = (
        df["hp"]
        + df["attack"]
        + df["defense"]
        + df["sp_attack"]
        + df["sp_defense"]
        + df["speed"]
    )


def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _ensure_base_stat_total(df)

    if "generation_num" not in df.columns and "generation" in df.columns:
        df["generation_num"] = df["generation"].apply(normalize_generation)

    _coerce_bool_columns(df, ["is_legendary", "is_mythical", "is_baby"])
    return df


def add_power_creep_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _ensure_base_stat_total(df)

    if "generation_num" not in df.columns and "generation" in df.columns:
        df["generation_num"] = df["generation"].apply(normalize_generation)

    gen_mean = df.groupby("generation_num")["base_stat_total"].transform("mean")
    fallback = df["base_stat_total"].mean()
    df[TARGET_COLUMN] = df["base_stat_total"] / gen_mean.fillna(fallback)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    features = df[FEATURE_COLUMNS].copy()
    for col in features.columns:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())
    return features


def build_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_with_pci = add_power_creep_index(df)
    X = prepare_features(df_with_pci)
    y = df_with_pci[TARGET_COLUMN]
    return X, y
