from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models import load_model
from src.preprocessing import FEATURE_COLUMNS


@st.cache_resource
def load_model_cached(model_path: Path):
    return load_model(model_path)


st.set_page_config(page_title="Poke Balance Sentinel", layout="centered")
st.title("Poke Balance Sentinel")
st.write("Estimate the Power Creep Index (PCI) from base stats and metadata.")

model_path = ROOT / "models" / "random_forest_pci.joblib"
if not model_path.exists():
    st.error("Model not found. Train it with `python train.py` first.")
    st.stop()

model = load_model_cached(model_path)

st.subheader("Pokemon Stats")
hp = st.number_input("HP", min_value=1, max_value=255, value=60)
attack = st.number_input("Attack", min_value=1, max_value=255, value=70)
defense = st.number_input("Defense", min_value=1, max_value=255, value=70)
sp_attack = st.number_input("Sp. Attack", min_value=1, max_value=255, value=70)
sp_defense = st.number_input("Sp. Defense", min_value=1, max_value=255, value=70)
speed = st.number_input("Speed", min_value=1, max_value=255, value=60)

generation_num = st.selectbox("Generation", [1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
num_types = st.selectbox("Number of Types", [1, 2], index=1)
height_m = st.number_input("Height (m)", min_value=0.1, max_value=20.0, value=1.0)
weight_kg = st.number_input("Weight (kg)", min_value=0.1, max_value=1000.0, value=30.0)

is_legendary = st.checkbox("Legendary", value=False)
is_mythical = st.checkbox("Mythical", value=False)
is_baby = st.checkbox("Baby Pokemon", value=False)

base_stat_total = hp + attack + defense + sp_attack + sp_defense + speed

input_data = {
    "generation_num": generation_num,
    "num_types": num_types,
    "hp": hp,
    "attack": attack,
    "defense": defense,
    "sp_attack": sp_attack,
    "sp_defense": sp_defense,
    "speed": speed,
    "base_stat_total": base_stat_total,
    "height_m": height_m,
    "weight_kg": weight_kg,
    "is_legendary": int(is_legendary),
    "is_mythical": int(is_mythical),
    "is_baby": int(is_baby),
}

features = pd.DataFrame([input_data])[FEATURE_COLUMNS]

if st.button("Predict PCI"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated PCI: {prediction:.3f}")
