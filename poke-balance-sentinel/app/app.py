import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# 1. Page Config & CSS Design System
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pokémon Meta-Balance Sentinel", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    :root { --bg-base: #0D1117; --bg-surface: #161B27; --accent: #4F83FF; --text-primary: #F1F5F9; --border: #2A3347; }
    .stApp { background-color: var(--bg-base); }
    
    /* Hero Header Styling */
    .hero-header { background: linear-gradient(135deg, #161B27 0%, #1a2340 60%, #1E2537 100%); 
                   border: 1px solid var(--border); border-radius: 14px; padding: 1.6rem; margin-bottom: 1.5rem; }
    .hero-title { font-size: 1.65rem; font-weight: 700; color: var(--text-primary); }
    
    /* Stats Row Styling */
    .stat-cards-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }
    .stat-card { background: var(--bg-surface); border-radius: 8px; padding: 1rem; border: 1px solid var(--border); text-align: center; }
    .stat-card-value { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); }
    .stat-card-label { font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; }
    
    /* Result Card Styling */
    .result-card { border-radius: 10px; padding: 1.25rem; margin: 0.75rem 0; border: 1px solid var(--border); background: #0D1B2B; color: var(--text-primary); }
    
    /* Custom Slider Accent */
    div[data-baseweb="slider"] { margin-bottom: 1rem; }
    
    /* Expander styling */
    div[data-testid="stExpander"] { background-color: var(--bg-surface); border: 1px solid var(--border); border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Load Models
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        designer_model = joblib.load('../models/unsupervised_balance_risk_pipeline.joblib')
        gamer_model = joblib.load('../models/longevity_RandomForest.joblib')
    except FileNotFoundError:
        # Dummy models for fail-safe demonstration
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
# 4. Sidebar Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Poke Balance Sentinel")
    st.info("**About:** This decision-support system uses two trained ML models to evaluate any Pokémon's competitive properties.")
    
    st.subheader("Loaded Models")
    st.success("Balance Risk: Unsupervised Pipeline → Loaded")
    st.success("Longevity: Random Forest → Loaded")

    # How To Use section
    st.subheader("How To Use")
    st.markdown("""
    1. **Adjust Base Stats** using the sliders.
    2. **Select Pokémon Types** from the dropdowns.
    3. **Review Derived Stats** calculated automatically.
    4. **Select a View Tab** (Developer or Gamer).
    5. Click the **Run Prediction** button.
    """)

# -----------------------------------------------------------------------------
# 5. Main Page Layout
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-header"><div class="hero-title">Predictive Meta-Balance System</div></div>', unsafe_allow_html=True)

st.header("Pokémon Configuration")

input_cols = st.columns([2, 1])

# --- Column 1: Base Stats ---
with input_cols[0]:
    st.subheader("Base Stats")
    stat_cols = st.columns(3)
    
    hp_raw = stat_cols[0].slider("HP", min_value=1, max_value=255, value=80, help="Min: 1 (Shedinja) | Max: 255 (Blissey)")
    sp_attack_raw = stat_cols[0].slider("Sp. Attack", min_value=10, max_value=194, value=80, help="Min: 10 (Shuckle) | Max: 194 (Mega Mewtwo Y)")
    
    attack_raw = stat_cols[1].slider("Attack", min_value=5, max_value=190, value=80, help="Min: 5 (Happiny) | Max: 190 (Mega Mewtwo X)")
    sp_defense_raw = stat_cols[1].slider("Sp. Defense", min_value=20, max_value=250, value=80, help="Min: 20 (Caterpie) | Max: 250 (Eternatus)")
    
    defense_raw = stat_cols[2].slider("Defense", min_value=5, max_value=250, value=80, help="Min: 5 (Chansey) | Max: 250 (Shuckle)")
    speed_raw = stat_cols[2].slider("Speed", min_value=5, max_value=200, value=80, help="Min: 5 (Shuckle) | Max: 200 (Regieleki)")

# --- Column 2: Type Information ---
with input_cols[1]:
    st.subheader("Type Information")
    primary_type = st.selectbox("Primary Type", POKEMON_TYPES, index=0)
    secondary_type = st.selectbox("Secondary Type", ["None"] + POKEMON_TYPES, index=0)

# -----------------------------------------------------------------------------
# 6. Data Engineering & Derived Stats
# -----------------------------------------------------------------------------
bst = hp_raw + attack_raw + defense_raw + sp_attack_raw + sp_defense_raw + speed_raw
offensive_total = speed_raw + attack_raw + sp_attack_raw
defensive_total = hp_raw + defense_raw + sp_defense_raw
stat_efficiency = bst / MEAN_BST if MEAN_BST > 0 else 0

st.divider()

st.markdown(f'''
    <div class="stat-cards-row">
        <div class="stat-card"><div class="stat-card-label">BASE STAT TOTAL</div><div class="stat-card-value">{bst}</div></div>
        <div class="stat-card"><div class="stat-card-label">OFFENSIVE SCORE</div><div class="stat-card-value">{offensive_total}</div></div>
        <div class="stat-card"><div class="stat-card-label">DEFENSIVE SCORE</div><div class="stat-card-value">{defensive_total}</div></div>
        <div class="stat-card"><div class="stat-card-label">STAT EFFICIENCY</div><div class="stat-card-value">{stat_efficiency:.2f}</div></div>
    </div>
''', unsafe_allow_html=True)

# --- Prepare DataFrames for Models ---
hp_scaled = (hp_raw - SCALERS['hp'][0]) / SCALERS['hp'][1]
attack_scaled = (attack_raw - SCALERS['attack'][0]) / SCALERS['attack'][1]
defense_scaled = (defense_raw - SCALERS['defense'][0]) / SCALERS['defense'][1]
sp_attack_scaled = (sp_attack_raw - SCALERS['sp_attack'][0]) / SCALERS['sp_attack'][1]
sp_defense_scaled = (sp_defense_raw - SCALERS['sp_defense'][0]) / SCALERS['sp_defense'][1]
speed_scaled = (speed_raw - SCALERS['speed'][0]) / SCALERS['speed'][1]

num_types = 1 if secondary_type == "None" else 2
type_dict = {f"type_{t.lower()}": 0 for t in POKEMON_TYPES}
type_dict[f"type_{primary_type.lower()}"] = 1
if secondary_type != "None":
    type_dict[f"type_{secondary_type.lower()}"] = 1

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

designer_dict = base_dict.copy()
designer_dict['base_stat_total'] = bst
designer_df = pd.DataFrame([designer_dict])
if hasattr(designer_model, "feature_names_in_"):
    model_cols = designer_model.feature_names_in_
    for col in model_cols:
        if col not in designer_df.columns:
            designer_df[col] = 0
    designer_df = designer_df[model_cols]

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
tab1, tab2 = st.tabs(["🎮 Developer View", "🏆 Gamer View"])

with tab1:
    st.header("Balance Risk Classification")
    st.write("Classifies whether this Pokémon is likely to be competitively balanced, a high-risk pick, or a low-risk pick in the meta. Powered by a trained Unsupervised `IsolationForest` Pipeline.")

    if st.button("Run Balance Risk Prediction"):
        designer_pred = designer_model.predict(designer_df)[0]
        designer_score = designer_model.decision_function(designer_df)[0]

        st.subheader("Verdict")
        if designer_pred == 1:
            st.markdown('<div class="result-card" style="border-color: #22C55E"><h4 style="margin-top:0;">🔵 BALANCED / LOW RISK</h4>This stat spread fits perfectly within established competitive norms. It is not an anomaly.</div>', unsafe_allow_html=True)
        elif bst > MEDIAN_BST:
            st.markdown('<div class="result-card" style="border-color: #EF4444"><h4 style="margin-top:0;">🔴 HIGH BALANCE RISK (OVERPOWERED)</h4>Potential overpowered profile detected. This Pokémon shows statistical patterns associated with overpowered or hard-to-counter competitive picks. It may require banning or tiering restrictions.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card" style="border-color: #F59E0B"><h4 style="margin-top:0;">🟡 DESIGN RISK (UNDERPOWERED)</h4>This design is mathematically anomalous on the low-end. It may be unviable in competitive play without specific buffs or powerful abilities.</div>', unsafe_allow_html=True)

        st.subheader("Model Decision Math")
        math_cols = st.columns(2)
        
        math_cols[0].metric(
            label="Raw Anomaly Score", 
            value=f"{designer_score:.3f}",
            help="Scores below 0 are flagged as anomalies."
        )
        
        if designer_pred == 1:
            math_cols[1].metric("Sub-Classification Logic", "Score > 0.000", "Normal Design", delta_color="normal")
        elif bst > MEDIAN_BST:
            math_cols[1].metric("Sub-Classification Logic", f"BST {bst} > {MEDIAN_BST:.0f}", "Overpowered Risk", delta_color="inverse")
        else:
            math_cols[1].metric("Sub-Classification Logic", f"BST {bst} ≤ {MEDIAN_BST:.0f}", "Underpowered Risk", delta_color="off")

        st.caption("💡 **How this works:** The AI calculates a **Raw Anomaly Score**. If the score drops below zero, the design is flagged as a statistical outlier. Because outliers can be both extremely strong or extremely weak, the system compares the design's Base Stat Total (BST) against the global median (450) to explain *why* it is an anomaly.")

    st.write("---")
    
    with st.expander("How this prediction works"):
        st.write("This model is an **Isolation Forest**, an unsupervised algorithm that detects anomalies. It was trained on data from all existing Pokémon (Generations 1-9). It doesn't learn 'good' or 'bad' but rather what is 'normal' vs 'abnormal'. A prediction of HIGH BALANCE RISK means the Pokémon's stats are a statistical outlier on the high end, similar to existing legendary or pseudo-legendary Pokémon that often dominate the meta.")

    with st.expander("Input features sent to this model"):
        st.dataframe(designer_df)

    with st.expander("Model limitations"):
        st.warning("This model only analyzes a Pokémon's stats and types. It **does not** account for Abilities, Moves, or Items, which are also critical factors in competitive balance.")

with tab2:
    st.header("Competitive Longevity Prediction")
    st.write("Predicts how many future generations this Pokémon is likely to remain viable in the standard competitive meta. Powered by a Random Forest Regressor.")
    
    if st.button("Run Longevity Prediction"):
        longevity_pred = gamer_model.predict(gamer_df)[0]
        
        st.metric(label="Predicted Viability", value=f"{longevity_pred:.1f} Generations")
        
        if longevity_pred >= 4.0:
            st.markdown('<div class="result-card" style="border-color: #22C55E"><h4 style="margin-top:0;">🟢 SAFE INVESTMENT</h4>This stat spread shows high resilience against historical power creep. It is expected to remain viable for multiple generations.</div>', unsafe_allow_html=True)
        elif longevity_pred >= 2.0:
            st.markdown('<div class="result-card" style="border-color: #F59E0B"><h4 style="margin-top:0;">🟡 MODERATE INVESTMENT</h4>Viable in the current meta, but its longevity is average. It may fall off as power creep continues in future generations.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card" style="border-color: #EF4444"><h4 style="margin-top:0;">🔴 RISKY INVESTMENT</h4>This stat spread is highly vulnerable to shifting meta trends and power creep. It will likely struggle to maintain relevance.</div>', unsafe_allow_html=True)
            
    st.write("---")
    
    with st.expander("How this prediction works"):
        st.write("This model is a **Random Forest Regressor** trained on historical competitive data. It evaluates how stat distributions, typing (Type Coverage), and Stat Efficiency contribute to a Pokémon remaining viable across multiple generations of power creep. It identifies patterns shared by Pokémon that have consistently stayed in standard play.")

    with st.expander("Input features sent to this model"):
        st.dataframe(gamer_df)

    with st.expander("Model limitations"):
        st.warning("This model analyzes base stats and typing combinations. It **does not** account for specific abilities, move pools, or mechanical changes introduced in newer generations (like Mega Evolution, Z-Moves, Dynamax, or Terastallization).")