import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pokémon Meta-Balance Sentinel", layout="wide", page_icon="🛡️")

st.title("🛡️ Predictive Meta-Balance System")
st.markdown("Decision-support tool for **Game Developers** and **Competitive Players**.")

# -----------------------------------------------------------------------------
# 2. Load Models
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    designer_model = joblib.load('../models/unsupervised_balance_risk_pipeline.joblib')
    gamer_model = joblib.load('../models/longevity_RandomForest.joblib')
    return designer_model, gamer_model

try:
    designer_model, gamer_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please check your file paths.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Data Scaling Constants 
# -----------------------------------------------------------------------------
MEDIAN_BST = 450.0
MEAN_BST = 427.686

# Means and Standard Deviations from the raw dataset (to match Member 1's prep)
SCALERS = {
    'hp': (70.184, 26.631),
    'attack': (77.522, 29.783),
    'defense': (72.507, 29.287),
    'sp_attack': (70.081, 29.658),
    'sp_defense': (70.206, 26.639),
    'speed': (67.186, 28.717)
}

# -----------------------------------------------------------------------------
# 4. Sidebar: User Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("🛠️ Design a Pokémon")
st.sidebar.markdown("Adjust the base stats below. Both models will adapt dynamically.")

hp_raw = st.sidebar.slider("HP", 1, 255, 80)
attack_raw = st.sidebar.slider("Attack", 5, 255, 80)
defense_raw = st.sidebar.slider("Defense", 5, 255, 80)
sp_attack_raw = st.sidebar.slider("Sp. Attack", 10, 255, 80)
sp_defense_raw = st.sidebar.slider("Sp. Defense", 10, 255, 80)
speed_raw = st.sidebar.slider("Speed", 5, 255, 80)

st.sidebar.divider()
st.sidebar.header("🧬 Types")
num_types = st.sidebar.radio("Number of Types", [1, 2], index=0)

# -----------------------------------------------------------------------------
# 5. Data Engineering: Auto-calculate remaining features
# -----------------------------------------------------------------------------
# Derived Raw Stats
bst = hp_raw + attack_raw + defense_raw + sp_attack_raw + sp_defense_raw + speed_raw
physical_total = hp_raw + attack_raw + defense_raw
special_total = hp_raw + sp_attack_raw + sp_defense_raw
offensive_total = speed_raw + attack_raw + sp_attack_raw
defensive_total = hp_raw + defense_raw + sp_defense_raw
atk_def_ratio = attack_raw / defense_raw if defense_raw > 0 else attack_raw

# Standardize the Core 6 Stats to match the training data
hp_scaled = (hp_raw - SCALERS['hp'][0]) / SCALERS['hp'][1]
attack_scaled = (attack_raw - SCALERS['attack'][0]) / SCALERS['attack'][1]
defense_scaled = (defense_raw - SCALERS['defense'][0]) / SCALERS['defense'][1]
sp_attack_scaled = (sp_attack_raw - SCALERS['sp_attack'][0]) / SCALERS['sp_attack'][1]
sp_defense_scaled = (sp_defense_raw - SCALERS['sp_defense'][0]) / SCALERS['sp_defense'][1]
speed_scaled = (speed_raw - SCALERS['speed'][0]) / SCALERS['speed'][1]

# Base dictionary shared by both models
base_dict = {
    'num_types': num_types, 
    'hp': hp_scaled, 'attack': attack_scaled, 'defense': defense_scaled, 
    'sp_attack': sp_attack_scaled, 'sp_defense': sp_defense_scaled, 'speed': speed_scaled, 
    'height_m': 1.0, 'weight_kg': 30.0, 'base_experience': 150, 
    'capture_rate': 45, 'base_happiness': 70, 'hatch_counter': 20, 
    'gender_rate': 1, 'bmi': 30.0, 'attack_defense_ratio': atk_def_ratio, 
    'physical_total': physical_total, 'special_total': special_total, 
    'offensive_total': offensive_total, 'defensive_total': defensive_total,
    'type_bug': 0, 'type_dark': 0, 'type_dragon': 0, 'type_electric': 0, 
    'type_fairy': 0, 'type_fighting': 0, 'type_fire': 0, 'type_flying': 0, 
    'type_ghost': 0, 'type_grass': 0, 'type_ground': 0, 'type_ice': 0, 
    'type_normal': 1, 'type_poison': 0, 'type_psychic': 0, 'type_rock': 0, 
    'type_steel': 0, 'type_water': 0
}

# --- CREATE DESIGNER DATAFRAME ---
designer_dict = base_dict.copy()
designer_dict['base_stat_total'] = bst
designer_df = pd.DataFrame([designer_dict])
if hasattr(designer_model, "feature_names_in_"):
    designer_df = designer_df[designer_model.feature_names_in_]

# --- CREATE GAMER DATAFRAME ---
gamer_dict = base_dict.copy()
# Automate Member 4's logic so the Gamers don't have to!
gamer_dict['stat_efficiency'] = bst / MEAN_BST
gamer_dict['type_coverage'] = num_types
gamer_df = pd.DataFrame([gamer_dict])
if hasattr(gamer_model, "feature_names_in_"):
    gamer_df = gamer_df[gamer_model.feature_names_in_]

# -----------------------------------------------------------------------------
# 6. Display the Current Build
# -----------------------------------------------------------------------------
st.subheader("Current Stat Spread")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Base Stat Total", bst)
col2.metric("Offensive Total", offensive_total)
col3.metric("Defensive Total", defensive_total)
col4.metric("Sweep Efficiency", f"{(speed_raw/defense_raw if defense_raw > 0 else 0):.2f}")

st.divider()

# -----------------------------------------------------------------------------
# 7. Predictions Layout (Using Tabs!)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🎮 Developer View (Balance Risk)", "🏆 Gamer View (Longevity)"])

with tab1:
    st.header("🎮 Balance Risk Assessment")
    
    # Predict using the Isolation Forest
    designer_pred = designer_model.predict(designer_df)[0]
    designer_score = designer_model.decision_function(designer_df)[0]
    
    # Color-coded to match the Scatter Plot in your Presentation!
    if designer_pred == 1:
        st.success("## 🔵 BALANCED (NORMAL) DESIGN")
        st.write("This stat spread fits perfectly within established mathematical norms. It is **not** an anomaly.")
    elif bst > MEDIAN_BST:
        st.error("## 🔴 OVERPOWERED RISK (ANOMALY)")
        st.write("This design is mathematically anomalous on the high-end. High risk of breaking the meta.")
    else:
        st.warning("## 🟡 UNDERPOWERED RISK (ANOMALY)")
        st.write("This design is mathematically anomalous on the low-end. It may be unviable in competitive play without buffs.")

    st.caption(f"Raw Anomaly Score: {designer_score:.3f} (Scores below 0 are anomalies)")


with tab2:
    st.header("🏆 Competitive Longevity")
    
    # Predict using the Random Forest
    longevity_pred = gamer_model.predict(gamer_df)[0]
    
    st.metric(label="Predicted Viability", value=f"{longevity_pred:.1f} Generations")
    st.caption("How long this Pokémon is expected to stay in the standard competitive meta.")
    
    if longevity_pred >= 4.0:
        st.success("Safe Investment: This stat spread shows high resilience against historical power creep.")
    elif longevity_pred >= 2.0:
        st.warning("Moderate Investment: Viable now, but may fall off as power creep continues.")
    else:
        st.error("Risky Investment: This stat spread is highly vulnerable to shifting meta trends.")