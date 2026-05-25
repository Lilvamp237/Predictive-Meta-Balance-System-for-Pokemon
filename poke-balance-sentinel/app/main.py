from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (ML logic - unchanged)
# ─────────────────────────────────────────────────────────────────────────────

ALL_TYPES = [
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
    "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
    "psychic", "rock", "steel", "water",
]

# balance risk now uses an unsupervised model from only the six base stats.
BALANCE_RISK_FEATURES = [
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
]

#longevity uses Random Forest with six base stats plus Types
LONGEVITY_FEATURES = [
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
] + [f"type_{t}" for t in ALL_TYPES]

TYPE_OPTIONS = ["None"] + [t.capitalize() for t in ALL_TYPES]

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (ML logic - unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    import joblib

    models = {}

    paths = {
        #updated the balance risk model path to reflect the new unsupervised pipeline
        "balance_risk": ROOT / "models" / "unsupervised_balance_risk_pipeline.joblib",
        "balance_label_map": ROOT / "models" / "balance_label_map.joblib",
        "longevity": ROOT / "models" / "longevity_RandomForest.joblib",
        "scaler": ROOT / "models" / "pokemon_scaler.joblib",
        "type_columns": ROOT / "models" / "type_columns.joblib",
    }

    for key, path in paths.items():
        if path.exists():
            try:
                models[key] = joblib.load(path)
            except Exception as e:
                models[key] = None
                st.error(f"Failed to load {path.name}: {e}")
        else:
            models[key] = None

    return models
# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (ML logic - unchanged)
# ─────────────────────────────────────────────────────────────────────────────

#Updated the build_feature_row() function by splitting this into two dedicated builders: one for balance risk, one for longevity.
def build_balance_risk_row(
    hp, attack, defense, sp_attack, sp_defense, speed
) -> dict:
    return {
        "hp": hp,
        "attack": attack,
        "defense": defense,
        "sp_attack": sp_attack,
        "sp_defense": sp_defense,
        "speed": speed,
    }


def build_longevity_row(
    hp, attack, defense, sp_attack, sp_defense, speed,
    primary_type, secondary_type,
    type_columns=None,
) -> dict:
    row = {
        "hp": hp,
        "attack": attack,
        "defense": defense,
        "sp_attack": sp_attack,
        "sp_defense": sp_defense,
        "speed": speed,
    }

    selected_types = set()
    if primary_type and primary_type.lower() != "none":
        selected_types.add(primary_type.lower())
    if secondary_type and secondary_type.lower() != "none":
        selected_types.add(secondary_type.lower())

    all_type_columns = type_columns if type_columns is not None else [f"type_{t}" for t in ALL_TYPES]
    for col in all_type_columns:
        row[col] = 0

    for t in selected_types:
        col = f"type_{t}"
        if col in row:
            row[col] = 1

    return row

#removed the old align_features
#this align features trying to explicitly control the inputs to fix the leakage-related design changes.
def align_features(row: dict, feature_list: list) -> pd.DataFrame:
    missing = [f for f in feature_list if f not in row]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return pd.DataFrame([{f: row[f] for f in feature_list}])

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Poke Balance Sentinel - Meta-Balance Decision Support",
    page_icon="assets/favicon.ico" if (ROOT / "assets" / "favicon.ico").exists() else "⚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS - Design System
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Design tokens ────────────────────────────────────────── */
    :root {
        --bg-base:      #0D1117;
        --bg-surface:   #161B27;
        --bg-elevated:  #1E2537;
        --accent:       #4F83FF;
        --accent-dim:   #2E4FAD;
        --success:      #22C55E;
        --success-dim:  #14532D;
        --danger:       #EF4444;
        --danger-dim:   #7F1D1D;
        --warning:      #F59E0B;
        --warning-dim:  #78350F;
        --text-primary: #F1F5F9;
        --text-secondary:#94A3B8;
        --text-muted:   #64748B;
        --border:       #2A3347;
        --border-accent:#4F83FF40;
        --radius-sm:    6px;
        --radius-md:    10px;
        --radius-lg:    14px;
    }

    /* ── App shell ────────────────────────────────────────────── */
    .stApp {
        background-color: var(--bg-base);
    }
    section[data-testid="stSidebar"] {
        background-color: var(--bg-surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Hero header ──────────────────────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #161B27 0%, #1a2340 60%, #1E2537 100%);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius-lg);
        padding: 1.6rem 2rem 1.4rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: "";
        position: absolute;
        top: -40px; right: -40px;
        width: 160px; height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, #4F83FF18 0%, transparent 70%);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.3px;
    }
    .hero-sub {
        font-size: 0.92rem;
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.5;
    }
    .hero-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }
    .badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-blue  { background: #4F83FF22; color: #7EB3FF; border: 1px solid #4F83FF44; }
    .badge-green { background: #22C55E22; color: #4ADE80; border: 1px solid #22C55E44; }
    .badge-slate { background: #94A3B822; color: #CBD5E1; border: 1px solid #94A3B844; }

    /* ── Section labels ───────────────────────────────────────── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
        margin-top: 0.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.15rem;
    }

    /* ── Derived stat cards ───────────────────────────────────── */
    .stat-cards-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 0.75rem 0 1rem 0;
    }
    .stat-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.1rem 0.85rem 1.1rem;
        transition: border-color 0.15s ease;
    }
    .stat-card:hover { border-color: var(--border-accent); }
    .stat-card-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .stat-card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .stat-card-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    .stat-card-accent { border-top: 2px solid var(--accent); }
    .stat-card-success { border-top: 2px solid var(--success); }
    .stat-card-danger  { border-top: 2px solid var(--danger); }
    .stat-card-warning { border-top: 2px solid var(--warning); }

    /* ── Prediction result cards ──────────────────────────────── */
    .result-card {
        border-radius: var(--radius-md);
        padding: 1.25rem 1.4rem;
        margin: 0.75rem 0;
        border: 1px solid;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: var(--radius-sm) 0 0 var(--radius-sm);
    }
    .result-card-success {
        background: #0D2B1A;
        border-color: #22C55E44;
    }
    .result-card-success::before { background: var(--success); }
    .result-card-danger {
        background: #2B0D0D;
        border-color: #EF444444;
    }
    .result-card-danger::before { background: var(--danger); }
    .result-card-warning {
        background: #2B1D0D;
        border-color: #F59E0B44;
    }
    .result-card-warning::before { background: var(--warning); }
    .result-card-info {
        background: #0D1B2B;
        border-color: #4F83FF44;
    }
    .result-card-info::before { background: var(--accent); }

    .result-verdict {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .result-verdict-success { color: #4ADE80; }
    .result-verdict-danger  { color: #FCA5A5; }
    .result-verdict-warning { color: #FCD34D; }
    .result-verdict-info    { color: #93C5FD; }

    .result-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }
    .result-body {
        font-size: 0.87rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 0;
    }

    /* ── Confidence bar ───────────────────────────────────────── */
    .conf-bar-wrap {
        margin: 1rem 0 0.25rem 0;
    }
    .conf-bar-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.4rem;
    }
    .conf-bar-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .conf-bar-pct {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .conf-bar-track {
        height: 6px;
        background: var(--bg-elevated);
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    .conf-bar-note {
        font-size: 0.73rem;
        color: var(--text-muted);
        margin-top: 0.35rem;
    }

    /* ── Probability split bar ────────────────────────────────── */
    .prob-split-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        margin: 0.75rem 0;
    }
    .prob-cell {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.75rem 0.9rem;
    }
    .prob-cell-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .prob-cell-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ── Input section panel ──────────────────────────────────── */
    .input-panel {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.25rem 1.25rem 1.25rem;
        margin-bottom: 0.5rem;
    }

    /* ── Streamlit native overrides ───────────────────────────── */
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        letter-spacing: 0.01em;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--accent) !important;
    }
    div[data-testid="stExpander"] summary {
        font-size: 0.83rem;
        font-weight: 600;
        color: var(--text-secondary) !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        background: var(--bg-surface) !important;
    }
    hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

    /* ── Sidebar overrides ────────────────────────────────────── */
    .sidebar-brand {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.15rem;
    }
    .sidebar-sub {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-bottom: 0;
    }
    .sidebar-model-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        margin-bottom: 0.4rem;
    }
    .sidebar-model-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .sidebar-model-name {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-secondary);
    }
    .sidebar-model-arch {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-left: auto;
    }
    .disclaimer-box {
        background: #78350F22;
        border: 1px solid #F59E0B33;
        border-radius: var(--radius-sm);
        padding: 0.65rem 0.8rem;
        font-size: 0.76rem;
        color: #FCD34D;
        line-height: 1.5;
    }

    /* ── Streamlit button ─────────────────────────────────────── */
    [data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent) !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.6rem !important;
        color: #fff !important;
        transition: background 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #3A6FEE !important;
        box-shadow: 0 0 0 3px #4F83FF33 !important;
    }

    /* ── Responsive grid fallback ─────────────────────────────── */
    @media (max-width: 768px) {
        .stat-cards-row { grid-template-columns: repeat(2, 1fr); }
        .prob-split-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 1.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: HTML rendering utilities (UI only)
# ─────────────────────────────────────────────────────────────────────────────

def render_result_card(style: str, verdict_label: str, title: str, body: str):
    """Render a styled result card. style: 'success' | 'danger' | 'warning' | 'info'"""
    st.markdown(
        f"""
        <div class="result-card result-card-{style}">
            <div class="result-verdict result-verdict-{style}">{verdict_label}</div>
            <div class="result-title">{title}</div>
            <p class="result-body">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_bar(label: str, pct: float, color: str = "#4F83FF"):
    """Render a labelled progress bar for a confidence/probability value (0–100)."""
    note = ""
    if pct < 55:
        note = "Near 50% - model is uncertain about this prediction."
    elif pct >= 85:
        note = "Above 85% - model is confident in this prediction."
    else:
        note = "Moderate confidence - treat with appropriate caution."

    st.markdown(
        f"""
        <div class="conf-bar-wrap">
            <div class="conf-bar-header">
                <span class="conf-bar-label">{label}</span>
                <span class="conf-bar-pct">{pct:.1f}%</span>
            </div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
            <div class="conf-bar-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prob_split(p_balanced: float, p_risky: float):
    """Render a 2-cell probability display."""
    st.markdown(
        f"""
        <div class="prob-split-grid">
            <div class="prob-cell">
                <div class="prob-cell-label">P(Balanced / Low Risk)</div>
                <div class="prob-cell-value" style="color:#4ADE80;">{p_balanced*100:.1f}%</div>
            </div>
            <div class="prob-cell">
                <div class="prob-cell-label">P(High Balance Risk)</div>
                <div class="prob-cell-value" style="color:#FCA5A5;">{p_risky*100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(bst: int, off: int, deff: int, bmi_val: float, eff: float):
    """Render the four derived-stat cards."""
    bst_color = "danger" if bst >= 580 else ("warning" if bst >= 480 else "accent")
    st.markdown(
        f"""
        <div class="stat-cards-row">
            <div class="stat-card stat-card-{bst_color}">
                <div class="stat-card-label">Base Stat Total</div>
                <div class="stat-card-value">{bst}</div>
                <div class="stat-card-sub">Avg. fully evolved &asymp; 500</div>
            </div>
            <div class="stat-card stat-card-accent">
                <div class="stat-card-label">Offensive Score</div>
                <div class="stat-card-value">{off}</div>
                <div class="stat-card-sub">ATK + Sp.ATK + SPD</div>
            </div>
            <div class="stat-card stat-card-success">
                <div class="stat-card-label">Defensive Score</div>
                <div class="stat-card-value">{deff}</div>
                <div class="stat-card-sub">HP + DEF + Sp.DEF</div>
            </div>
            <div class="stat-card stat-card-warning">
                <div class="stat-card-label">Stat Efficiency</div>
                <div class="stat-card-value">{eff:.2f}</div>
                <div class="stat-card-sub">BST / 600 - ideal range 0.7&ndash;1.0</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

models = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">Poke Balance Sentinel</div>
        <div class="sidebar-sub">Predictive Meta-Balance System - Gen 1&ndash;9</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        """<div style="font-size:0.83rem;color:#94A3B8;line-height:1.65;">
        This decision-support system uses two trained ML models to evaluate
        any Pokemon's competitive properties:<br><br>
        <b style="color:#F1F5F9;">Balance Risk</b> - will it destabilise the meta?<br>
        <b style="color:#F1F5F9;">Competitive Longevity</b> - how many generations will it stay viable?
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="section-label">Loaded Models</div>', unsafe_allow_html=True)
    for display_name, key, arch in [
        ("Balance Risk", "balance_risk", "sklearn Pipeline"),
        ("Longevity",    "longevity",    "Random Forest Regressor"),
    ]:
        dot_color = "#22C55E" if models.get(key) else "#EF4444"
        status    = "Loaded" if models.get(key) else "Not found"
        st.markdown(
            f"""
            <div class="sidebar-model-row">
                <div class="sidebar-model-dot" style="background:{dot_color};"></div>
                <div>
                    <div class="sidebar-model-name">{display_name}</div>
                    <div style="font-size:0.68rem;color:#64748B;">{arch} &bull; {status}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown('<div class="section-label">How to Use</div>', unsafe_allow_html=True)
    st.markdown(
        """<div style="font-size:0.82rem;color:#94A3B8;line-height:1.75;">
        1. Enter <b style="color:#F1F5F9;">Base Stats</b> in the left column<br>
        2. Set <b style="color:#F1F5F9;">Physical Attributes &amp; Type</b> in the centre<br>
        3. Fill in <b style="color:#F1F5F9;">Game Metadata</b> on the right<br>
        4. Review the <b style="color:#F1F5F9;">Derived Stats</b> row<br>
        5. Select a <b style="color:#F1F5F9;">prediction tab</b> and click Predict
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div class="disclaimer-box">
            All predictions are model estimates based on historical data.
            They are not official competitive rulings.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Page hero header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">Pokemon Meta-Balance Decision Support Tool</div>
        <div class="hero-sub">
            Configure a Pokemon below to receive ML-driven predictions on competitive balance risk
            and long-term viability across future generations.
        </div>
        <div class="hero-badges">
            <span class="badge badge-blue">Gen 1 – 9 Training Data</span>
            <span class="badge badge-green">2 ML Models</span>
            <span class="badge badge-slate">University ML Group Project</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Input form - 3-column layout
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Pokemon Configuration</div>', unsafe_allow_html=True)

left, mid, right = st.columns([1, 1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Base Stats</div>', unsafe_allow_html=True)
    hp         = st.number_input("HP",          min_value=1, max_value=255, value=65, help="Hit points - base durability.")
    attack     = st.number_input("Attack",      min_value=1, max_value=255, value=65, help="Physical attack power.")
    defense    = st.number_input("Defense",     min_value=1, max_value=255, value=65, help="Physical defence.")
    sp_attack  = st.number_input("Sp. Attack",  min_value=1, max_value=255, value=65, help="Special attack power.")
    sp_defense = st.number_input("Sp. Defense", min_value=1, max_value=255, value=65, help="Special defence.")
    speed      = st.number_input("Speed",       min_value=1, max_value=255, value=65, help="Initiative order in battle.")

with mid:
    st.markdown('<div class="section-title">Physical Attributes</div>', unsafe_allow_html=True)
    height_m  = st.number_input("Height (m)",  min_value=0.1, max_value=20.0,   value=1.0,  step=0.1)
    weight_kg = st.number_input("Weight (kg)", min_value=0.1, max_value=1000.0, value=30.0, step=0.5)

    st.markdown('<div class="section-title" style="margin-top:0.75rem;">Type Information</div>', unsafe_allow_html=True)
    primary_type   = st.selectbox("Primary Type",   TYPE_OPTIONS, index=13, help="Every Pokemon has at least one type.")
    secondary_type = st.selectbox("Secondary Type", TYPE_OPTIONS, index=0,  help="Select None for single-type Pokemon.")
    num_types = 1 if secondary_type == "None" else 2
    st.caption(f"Detected type count: **{num_types}**")

with right:
    st.markdown('<div class="section-title">Game Metadata</div>', unsafe_allow_html=True)
    base_experience = st.number_input(
        "Base Experience", min_value=0, max_value=608, value=100,
        help="XP awarded when this Pokemon is defeated in battle.")
    capture_rate = st.number_input(
        "Capture Rate", min_value=0, max_value=255, value=45,
        help="Higher = easier to catch. Max is 255.")
    base_happiness = st.number_input(
        "Base Happiness", min_value=0, max_value=255, value=70,
        help="Starting friendship value.")
    hatch_counter = st.number_input(
        "Hatch Counter", min_value=0, max_value=120, value=20,
        help="Egg cycles required to hatch.")
    gender_rate = st.number_input(
        "Gender Rate", min_value=-1, max_value=8, value=4,
        help="-1 = genderless. 0 = always male. 8 = always female.")

# ─────────────────────────────────────────────────────────────────────────────
# Derived stats row - live metric cards (HTML)
# ─────────────────────────────────────────────────────────────────────────────

base_stat_total = hp + attack + defense + sp_attack + sp_defense + speed
bmi             = weight_kg / (height_m ** 2) if height_m > 0 else 0.0
offensive_total = attack + sp_attack + speed
defensive_total = hp + defense + sp_defense
stat_efficiency = base_stat_total / 600.0

st.divider()
st.markdown('<div class="section-label">Derived Stats</div>', unsafe_allow_html=True)
st.caption("Calculated automatically from your inputs and passed to the models as features.")

render_stat_cards(base_stat_total, offensive_total, defensive_total, bmi, stat_efficiency)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Build shared feature row  (ML logic - unchanged)
# ─────────────────────────────────────────────────────────────────────────────

#updated this section. made it as 2 rows
try:
    balance_feature_row = build_balance_risk_row(
        hp=hp,
        attack=attack,
        defense=defense,
        sp_attack=sp_attack,
        sp_defense=sp_defense,
        speed=speed,
    )

    longevity_type_columns = models.get("type_columns")
    if longevity_type_columns is None:
        longevity_type_columns = [f"type_{t}" for t in ALL_TYPES]

    longevity_feature_row = build_longevity_row(
        hp=hp,
        attack=attack,
        defense=defense,
        sp_attack=sp_attack,
        sp_defense=sp_defense,
        speed=speed,
        primary_type=primary_type,
        secondary_type=secondary_type,
        type_columns=longevity_type_columns,
    )

    feature_error = None
except Exception as e:
    balance_feature_row = None
    longevity_feature_row = None
    feature_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Balance Risk Prediction", "Competitive Longevity Prediction"])

# ── Tab 1: Balance Risk ──────────────────────────────────────────────────────
#updated this section to reflect the new unsupervised clustering approach and label mapping
with tab1:
    st.markdown('<div class="section-title" style="margin-top:0.5rem;">Balance Risk Classification</div>', unsafe_allow_html=True)
    st.caption(
        "Groups this Pokemon into competitive balance clusters using an unsupervised model. "
        "The prediction is based only on the six base stats."
    )
    st.markdown("")

    if feature_error:
        st.error(f"Feature engineering error: {feature_error}")
    elif models["balance_risk"] is None:
        st.error(
            "Balance Risk model could not be loaded. "
            "Ensure `models/balance_risk_kmeans.joblib` exists in the project root."
        )
    elif models["balance_label_map"] is None:
        st.error(
            "Balance label map could not be loaded. "
            "Ensure `models/balance_label_map.joblib` exists in the project root."
        )
    else:
        if st.button("Run Balance Risk Prediction", type="primary"):
            with st.spinner("Running balance-risk model..."):
                try:
                    X = align_features(balance_feature_row, BALANCE_RISK_FEATURES)

                    if models["scaler"] is not None:
                        X_used = models["scaler"].transform(X)
                    else:
                        X_used = X

                    cluster_id = int(models["balance_risk"].predict(X_used)[0])
                    predicted_label = models["balance_label_map"].get(cluster_id, "Unknown")

                    st.markdown('<div class="section-label" style="margin-top:1rem;">Verdict</div>', unsafe_allow_html=True)

                    if predicted_label == "Underpowered":
                        render_result_card(
                            "warning",
                            "Underpowered",
                            "Below typical competitive stat profile",
                            "This Pokemon's six base stats place it in a lower-power cluster. "
                            "It may struggle to keep up unless other battle factors compensate."
                        )
                        bar_color = "#F59E0B"

                    elif predicted_label == "Normal":
                        render_result_card(
                            "success",
                            "Normal",
                            "No major balance concern detected",
                            "This Pokemon's six base stats fall into the normal cluster, "
                            "suggesting a more balanced overall stat profile."
                        )
                        bar_color = "#22C55E"

                    elif predicted_label == "Overpowered":
                        render_result_card(
                            "danger",
                            "Overpowered",
                            "Potential high-power profile detected",
                            "This Pokemon's six base stats place it in a high-power cluster "
                            "that may indicate above-normal balance risk."
                        )
                        bar_color = "#EF4444"

                    else:
                        render_result_card(
                            "info",
                            "Unknown Cluster",
                            "Cluster label not found",
                            f"The model predicted cluster {cluster_id}, but no readable label was found in the saved label map."
                        )
                        bar_color = "#4F83FF"

                    #removed the below line 
                    #render_confidence_bar("Cluster ID", float(cluster_id), bar_color)
                    st.caption(f"Predicted cluster ID: {cluster_id}")

                except ValueError as e:
                    st.error(f"Feature alignment error: {e}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    st.markdown("")

    with st.expander("How this prediction works"):
        st.markdown(
            "The Balance Risk model now uses an **unsupervised clustering** approach rather than a supervised classifier.\n\n"
            "It groups each Pokemon using only these six base stats:\n"
            "- HP\n"
            "- Attack\n"
            "- Defense\n"
            "- Sp. Attack\n"
            "- Sp. Defense\n"
            "- Speed\n\n"
            "The predicted cluster is then mapped into one of three readable labels:\n\n"
            "| Cluster label | Meaning |\n"
            "|---|---|\n"
            "| Underpowered | Lower overall stat profile |\n"
            "| Normal | Typical / balanced stat profile |\n"
            "| Overpowered | Higher overall stat profile |"
        )

    with st.expander("Input features sent to this model"):
        if balance_feature_row is not None:
            try:
                preview = {f: balance_feature_row[f] for f in BALANCE_RISK_FEATURES if f in balance_feature_row}
                st.dataframe(
                    pd.DataFrame([preview]).T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"Could not render feature table: {e}")
        else:
            st.caption("No feature row available due to an earlier error.")

    with st.expander("Model limitations"):
        st.markdown(
            "- Uses only the six base stats, so it does not account for movesets, abilities, items, or team synergy.\n"
            "- Cluster labels are interpretive summaries of unsupervised groups, not official competitive rulings.\n"
            "- Hypothetical or future Pokemon may fall outside the training distribution.\n"
            "- Treat predictions as a supporting signal, not a definitive ruling."
        )

# ── Tab 2: Longevity ─────────────────────────────────────────────────────────
#did few updates here

with tab2:
    st.markdown('<div class="section-title" style="margin-top:0.5rem;">Competitive Longevity Regression</div>', unsafe_allow_html=True)
    st.caption(
        "Estimates how viable this Pokemon will remain across future generations of competitive play. "
        "The output is a continuous score - higher values indicate greater predicted staying power."
    )
    st.markdown("")

    if feature_error:
        st.error(f"Feature engineering error: {feature_error}")
    elif models["longevity"] is None:
        st.error(
            "Longevity model could not be loaded. "
            
            #updated the error message to reflect the new regression model
            "Ensure `models/longevity_random_forest.joblib` exists in the project root."
        )
    else:
        if st.button("Run Longevity Prediction", type="primary"):
            with st.spinner("Running regressor..."):
                try:
                    #updated to use the new longevity feature set and handle potential missing features due to loading issues
                    X = align_features(longevity_feature_row, LONGEVITY_FEATURES)
                    longevity_score = float(models["longevity"].predict(X)[0])

                    st.markdown('<div class="section-label" style="margin-top:1rem;">Result</div>', unsafe_allow_html=True)

                    # Big metric display
                    col_score, col_pad = st.columns([1, 3])
                    col_score.metric(
                        "Longevity Score",
                        f"{longevity_score:.4f}",
                        help="Continuous regression output. Higher = longer predicted competitive viability.",
                    )

                    if longevity_score >= 0.75:
                        render_result_card(
                            "success",
                            "High Longevity",
                            "Strong long-term competitive presence predicted",
                            "This Pokemon is predicted to maintain strong competitive relevance "
                            "across multiple future generations. Its stat profile suggests a lasting meta presence.",
                        )
                        score_color = "#22C55E"
                    elif longevity_score >= 0.40:
                        render_result_card(
                            "info",
                            "Moderate Longevity",
                            "Niche or format-specific viability",
                            "This Pokemon may find consistent use in niche roles or specific formats, "
                            "but is unlikely to dominate universally across all generations.",
                        )
                        score_color = "#4F83FF"
                    else:
                        render_result_card(
                            "warning",
                            "Lower Longevity",
                            "Likely to be outclassed over time",
                            "This Pokemon's profile suggests it may struggle to remain competitive "
                            "as stronger Pokemon are introduced in future generations.",
                        )
                        score_color = "#F59E0B"

                    # Normalise score to a 0–100 display range for the bar (clamp)
                    display_pct = max(0.0, min(longevity_score, 1.0)) * 100
                    render_confidence_bar("Longevity Score (normalised display)", display_pct, score_color)

                    st.caption(
                        "The longevity score is a continuous regression estimate. "
                        "Values outside 0–1 are possible and reflect model extrapolation - "
                        "they are not errors."
                    )

                except ValueError as e:
                    st.error(f"Feature alignment error: {e}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    st.markdown("")

    #updated the explanation to reflect the new regression model and the interpretation of the continuous score
    with st.expander("How this prediction works"):
        st.markdown(
            "The Longevity model is a **Random Forest Regressor** trained to estimate "
            "a Pokemon's expected competitive lifespan.\n\n"
            "It uses the six base stats plus one-hot encoded type information.\n\n"
            "**Score interpretation guide:**\n\n"
            "| Score | Interpretation |\n"
            "|---|---|\n"
            "| >= 0.75 | High longevity - likely to stay relevant across generations |\n"
            "| 0.40 – 0.74 | Moderate - niche or format-specific viability |\n"
            "| < 0.40 | Lower longevity - likely to be outclassed over time |"
        )

    with st.expander("Input features sent to this model"):
        #updated to reflect the new feature set and potential for missing features due to loading issues
        if longevity_feature_row is not None:
            try:
                preview = {f: longevity_feature_row[f] for f in LONGEVITY_FEATURES if f in longevity_feature_row}
                st.dataframe(
                    pd.DataFrame([preview]).T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"Could not render feature table: {e}")
        else:
            st.caption("No feature row available due to an earlier error.")

    with st.expander("Model limitations"):
        st.markdown(
            "- Trained on Generation 1–9 data. Extrapolation to hypothetical Pokemon "
            "may produce unexpected results.\n"
            "- Real competitive viability depends on movesets, abilities, team roles, "
            "and game balance patches - none of which are captured in this model.\n"
            "- Scores outside 0–1 are mathematically valid regression outputs. "
            "Interpret them relative to the typical in-sample range rather than treating "
            "them as errors.\n"
            "- This tool is intended for academic demonstration, not competitive team-building."
        )