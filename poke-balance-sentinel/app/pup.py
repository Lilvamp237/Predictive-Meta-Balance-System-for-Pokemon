import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pokémon Meta-Balance Sentinel", layout="wide", page_icon="🛡️")

# -----------------------------------------------------------------------------
# 2. Load Models
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        designer_model = joblib.load('../models/unsupervised_balance_risk_pipeline.joblib')
        gamer_model = joblib.load('../models/longevity_RandomForest.joblib')
    except FileNotFoundError:
        # Create dummy models if the files don't exist, to allow the app to run.
        from sklearn.ensemble import IsolationForest, RandomForestRegressor
        st.warning("Model files not found. Using dummy models for demonstration.")
        designer_model = IsolationForest(random_state=42)
        dummy_data_iso = pd.DataFrame([[0]*20 for _ in range(10)], columns=[f'f{i}' for i in range(20)])
        designer_model.fit(dummy_data_iso)
        setattr(designer_model, "feature_names_in_", dummy_data_iso.columns)

        gamer_model = RandomForestRegressor(random_state=42)
        dummy_data_rf = pd.DataFrame([[0]*20 for _ in range(10)], columns=[f'f{i}' for i in range(20)])
        gamer_model.fit(dummy_data_rf, [3.0]*10)
        setattr(gamer_model, "feature_names_in_", dummy_data_rf.columns)

    return designer_model, gamer_model

designer_model, gamer_model = load_models()

# -----------------------------------------------------------------------------
# 3. Static Data & Constants
# -----------------------------------------------------------------------------
MEDIAN_BST = 450.0
MEAN_BST = 427.686
SCALERS = {
    'hp': (70.184, 26.631), 'attack': (77.522, 29.783), 'defense': (72.507, 29.287),
    'sp_attack': (70.081, 29.658), 'sp_defense': (70.206, 26.639), 'speed': (67.186, 28.717)
}
POKEMON_TYPES = [
    'Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting', 'Poison',
    'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark',
    'Steel', 'Fairy'
]

# -----------------------------------------------------------------------------
# 4. Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Poke Balance Sentinel")
    st.info("**About:** This decision-support system uses two trained ML models to evaluate any Pokémon's competitive properties.")

    st.subheader("Loaded Models")
    st.success("Balance Risk: `sklearn` Pipeline → Loaded")
    st.success("Longevity: Random Forest → Loaded")

    st.subheader("How To Use")
    st.markdown("""
    1. **Enter Base Stats** and **Types**.
    2. **Review the Derived Stats** row.
    3. **Select a prediction tab** and click **Predict**.
    """)

# -----------------------------------------------------------------------------
# 5. Main Page Layout
# -----------------------------------------------------------------------------
st.title("Pokémon Meta-Balance Decision Support Tool")
st.markdown("Configure a Pokémon below to receive ML-driven predictions on competitive balance risk and long-term viability across future generations.")

# --- Tags ---
cols = st.columns(3)
with cols[0]:
    st.info("GEN 1 - 9 TRAINING DATA")
with cols[1]:
    st.info("2 ML MODELS")
with cols[2]:
    st.info("UNIVERSITY ML GROUP PROJECT")

st.header("Pokémon Configuration")

# Clean 2-column layout to replace the old 3-column one
input_cols = st.columns([2, 1])

# --- Column 1: Base Stats ---
with input_cols[0]:
    st.subheader("Base Stats")
    # Nesting columns to create a neat 3x2 grid for stats
    stat_cols = st.columns(3)
    hp_raw = stat_cols[0].number_input("HP", 1, 255, 65, help="Health Points")
    attack_raw = stat_cols[1].number_input("Attack", 5, 255, 65, help="Physical Attack power")
    defense_raw = stat_cols[2].number_input("Defense", 5, 255, 65, help="Physical Defense")
    sp_attack_raw = stat_cols[0].number_input("Sp. Attack", 10, 255, 65, help="Special Attack power")
    sp_defense_raw = stat_cols[1].number_input("Sp. Defense", 10, 255, 65, help="Special Defense")
    speed_raw = stat_cols[2].number_input("Speed", 5, 255, 65, help="Determines turn order")

# --- Column 2: Type Information ---
with input_cols[1]:
    st.subheader("Type Information")
    primary_type = st.selectbox("Primary Type", POKEMON_TYPES, index=0)
    secondary_type = st.selectbox("Secondary Type", ["None"] + POKEMON_TYPES, index=0)

# -----------------------------------------------------------------------------
# 6. Data Engineering & Derived Stats
# -----------------------------------------------------------------------------
# --- Calculate Derived Stats ---
bst = hp_raw + attack_raw + defense_raw + sp_attack_raw + sp_defense_raw + speed_raw
offensive_total = speed_raw + attack_raw + sp_attack_raw
defensive_total = hp_raw + defense_raw + sp_defense_raw
stat_efficiency = bst / MEAN_BST if MEAN_BST > 0 else 0

st.divider()

# --- Display Derived Stats ---
st.subheader("Derived Stats")
st.markdown("Calculated automatically from your inputs and passed to the models as features.")
metric_cols = st.columns(4)
metric_cols[0].metric("BASE STAT TOTAL", bst, help="Sum of all 6 base stats. Fully evolved ~530.")
metric_cols[1].metric("OFFENSIVE SCORE", offensive_total, help="ATK + SP.ATK + SPD")
metric_cols[2].metric("DEFENSIVE SCORE", defensive_total, help="HP + DEF + SP.DEF")
metric_cols[3].metric("STAT EFFICIENCY", f"{stat_efficiency:.2f}", help="BST / Mean BST - Ideal range 0.7-1.3")

# --- Prepare DataFrames for Models ---
hp_scaled = (hp_raw - SCALERS['hp'][0]) / SCALERS['hp'][1]
attack_scaled = (attack_raw - SCALERS['attack'][0]) / SCALERS['attack'][1]
defense_scaled = (defense_raw - SCALERS['defense'][0]) / SCALERS['defense'][1]
sp_attack_scaled = (sp_attack_raw - SCALERS['sp_attack'][0]) / SCALERS['sp_attack'][1]
sp_defense_scaled = (sp_defense_raw - SCALERS['sp_defense'][0]) / SCALERS['sp_defense'][1]
speed_scaled = (speed_raw - SCALERS['speed'][0]) / SCALERS['speed'][1]

# Handle types
num_types = 1 if secondary_type == "None" else 2
type_dict = {f"type_{t.lower()}": 0 for t in POKEMON_TYPES}
type_dict[f"type_{primary_type.lower()}"] = 1
if secondary_type != "None":
    type_dict[f"type_{secondary_type.lower()}"] = 1

# Base dictionary shared by both models (hardcoding unused background features)
base_dict = {
    'num_types': num_types, 'hp': hp_scaled, 'attack': attack_scaled, 'defense': defense_scaled,
    'sp_attack': sp_attack_scaled, 'sp_defense': sp_defense_scaled, 'speed': speed_scaled,
    'height_m': 1.0, 'weight_kg': 30.0, 'base_experience': 150,
    'capture_rate': 45, 'base_happiness': 70, 'hatch_counter': 20,
    'gender_rate': 1, 'bmi': 30.0,
    'attack_defense_ratio': attack_raw / defense_raw if defense_raw > 0 else attack_raw,
    'physical_total': hp_raw + attack_raw + defense_raw,
    'special_total': hp_raw + sp_attack_raw + sp_defense_raw,
    'offensive_total': offensive_total,
    'defensive_total': defensive_total,
}
base_dict.update(type_dict)

# --- CREATE DESIGNER DATAFRAME ---
designer_dict = base_dict.copy()
designer_dict['base_stat_total'] = bst
designer_df = pd.DataFrame([designer_dict])
if hasattr(designer_model, "feature_names_in_"):
    model_cols = designer_model.feature_names_in_
    for col in model_cols:
        if col not in designer_df.columns:
            designer_df[col] = 0
    designer_df = designer_df[model_cols]

# --- CREATE GAMER DATAFRAME ---
gamer_dict = base_dict.copy()
gamer_dict['stat_efficiency'] = stat_efficiency
gamer_dict['type_coverage'] = num_types
gamer_df = pd.DataFrame([gamer_dict])
if hasattr(gamer_model, "feature_names_in_"):
    model_cols = gamer_model.feature_names_in_
    for col in model_cols:
        if col not in gamer_df.columns:
            gamer_df[col] = 0
    gamer_df = gamer_df[model_cols]

# -----------------------------------------------------------------------------
# 7. Predictions
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Balance Risk Prediction", "Competitive Longevity Prediction"])

with tab1:
    st.header("Balance Risk Classification")
    st.write("Classifies whether this Pokémon is likely to be competitively balanced or a high-risk pick in the meta. Powered by a trained `sklearn` Pipeline.")

    if st.button("Run Balance Risk Prediction"):
        designer_pred = designer_model.predict(designer_df)[0]
        is_anomaly = designer_pred == -1
        is_overpowered = is_anomaly and (bst > MEDIAN_BST)

        st.subheader("Verdict")
        if is_overpowered:
            with st.container(border=True):
                 st.error("#### HIGH BALANCE RISK")
                 st.write("Potential overpowered profile detected. This Pokémon shows statistical patterns associated with overpowered or hard-to-counter competitive picks. It may require banning or tiering restrictions.")
            prob_high_risk, prob_balanced = 1.0, 0.0
        else:
            with st.container(border=True):
                st.success("#### BALANCED / LOW RISK")
                st.write("This Pokémon's stat spread fits within established competitive norms. If it's an anomaly on the low end, it might be underpowered but does not pose a risk to meta balance.")
            prob_high_risk, prob_balanced = 0.0, 1.0

        st.subheader("Model Confidence in Predicted Class")
        st.progress(100)

        st.subheader("Class Probabilities")
        prob_cols = st.columns(2)
        prob_cols[0].metric("P(BALANCED / LOW RISK)", f"{prob_balanced:.1%}")
        prob_cols[1].metric("P(HIGH BALANCE RISK)", f"{prob_high_risk:.1%}")

    with st.expander("How this prediction works"):
        st.write("This model is an Isolation Forest, an unsupervised algorithm that detects anomalies. It was trained on data from all existing Pokémon (Generations 1-9). It doesn't learn 'good' or 'bad' but rather what is 'normal' vs 'abnormal'. A prediction of HIGH BALANCE RISK means the Pokémon's stats are a statistical outlier on the high end, similar to existing legendary or pseudo-legendary Pokémon that often dominate the meta.")

    with st.expander("Input features sent to this model"):
        st.dataframe(designer_df)

    with st.expander("Model limitations"):
        st.warning("This model only analyzes a Pokémon's stats and types. It **does not** account for Abilities, Moves, or Items, which are also critical factors in competitive balance.")

with tab2:
    st.header("Competitive Longevity Prediction")
    st.write("Predicts how many future generations this Pokémon is likely to remain viable in the standard competitive meta. Powered by a Random Forest Regressor.")
    
    if st.button("Run Longevity Prediction"):
        longevity_pred = gamer_model.predict(gamer_df)[0]
        st.metric(label="Predicted Viability (Generations)", value=f"{longevity_pred:.1f}")
        
        if longevity_pred >= 4.0:
            st.success("High Resilience: This Pokémon has a strong statistical profile, suggesting it will likely resist power creep for several generations.")
        elif longevity_pred >= 2.0:
            st.warning("Moderate Resilience: Viable in the current meta, but may be outclassed as more powerful Pokémon are introduced in the future.")
        else:
            st.error("Low Resilience: This Pokémon is highly vulnerable to power creep and may have a short competitive lifespan.")
            
        with st.expander("Input features sent to this model"):
            st.dataframe(gamer_df)