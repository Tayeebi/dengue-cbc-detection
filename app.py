import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Dengue Risk AI System",
    page_icon="🦟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "dengue_pipeline.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("Model file not found: dengue_pipeline.pkl")
    st.stop()

# -----------------------------
# PREDICTION
# -----------------------------
def predict_dengue(input_data):
    X = np.array([input_data], dtype=float)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return prediction, probability

# -----------------------------
# UI STYLES
# -----------------------------
st.markdown(
    """
    <style>
    :root{
        --bg: #f6f8fc;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --border: #e5e7eb;
        --blue: #2563eb;
        --blue-hover: #1d4ed8;
        --green: #16a34a;
        --green-hover: #15803d;
        --red-bg: #fef2f2;
        --red-border: #fecaca;
        --green-bg: #ecfdf5;
        --green-border: #bbf7d0;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    html, body, [class*="css"] {
        color: var(--text) !important;
    }

    h1, h2, h3, h4, p, label, div {
        color: var(--text);
    }

    .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.7rem 1.5rem 1.4rem 1.5rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        text-align: center;
        font-size: 2.05rem;
        line-height: 1.15;
        font-weight: 900;
        color: var(--text);
        margin: 0;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1rem;
        line-height: 1.65;
        color: var(--muted);
        margin-top: 0.75rem;
    }

    [data-testid="stForm"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.4rem;
    }

    .subtle {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    /* -----------------------------
       BUTTONS
    ----------------------------- */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] button {
        width: 100% !important;
        background: var(--blue) !important;
        color: #ffffff !important;
        border: 1px solid var(--blue) !important;
        border-radius: 14px !important;
        padding: 0.88rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 900 !important;
        line-height: 1.2 !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 10px 18px rgba(37, 99, 235, 0.16) !important;
        transition: all 0.18s ease-in-out !important;
        text-shadow: none !important;
        -webkit-appearance: none !important;
        appearance: none !important;
        opacity: 1 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--blue-hover) !important;
        border-color: var(--blue-hover) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button *,
    .stDownloadButton > button *,
    div[data-testid="stFormSubmitButton"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
        font-weight: 900 !important;
    }

    /* Inputs */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-weight: 700 !important;
        color: var(--text) !important;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 12px !important;
    }

    div[data-testid="stSelectbox"] > div {
        border-radius: 12px !important;
    }

    /* -----------------------------
       FIX: DROP-DOWN TEXT VISIBILITY (ALL STATES)
    ----------------------------- */
    /* Forces the selected value to be white, overriding WebKit defaults */
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Forces the expanded dropdown list to be white */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    ul[role="listbox"] li span,
    li[role="option"] span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* ----------------------------- */

    /* Result cards */
    .result-positive {
        background: var(--red-bg);
        border: 1px solid var(--red-border);
        border-radius: 18px;
        padding: 1rem;
        color: #991b1b;
        font-size: 1rem;
        line-height: 1.65;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .result-negative {
        background: var(--green-bg);
        border: 1px solid var(--green-border);
        border-radius: 18px;
        padding: 1rem;
        color: #166534;
        font-size: 1rem;
        line-height: 1.65;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .confidence {
        margin-top: 0.5rem;
        color: #334155;
        font-size: 1rem;
        line-height: 1.7;
    }

    .prob-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1rem;
        margin-top: 1.5rem;
    }

    .prob-row {
        margin-top: 0.75rem;
        margin-bottom: 0.35rem;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        font-weight: 800;
        color: #1f2937;
        font-size: 0.98rem;
    }

    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    .stProgress > div > div > div {
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🦟 Dengue Infection Risk Prediction System</div>
        <div class="hero-subtitle">
            A simple, professional clinical web app that predicts dengue risk using Complete Blood Count (CBC) parameters
            and presents the result in a clean, easy-to-read format.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# INPUT FORM
# -----------------------------
with st.form("dengue_form", clear_on_submit=False):
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Enter the patient’s CBC values and basic details below.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        haemoglobin = st.number_input("Haemoglobin", min_value=0.0, max_value=25.0, value=13.0, step=0.1, format="%.1f")
        esr = st.number_input("ESR", min_value=0.0, max_value=200.0, value=16.0, step=0.1, format="%.1f")
        wbc = st.number_input("WBC", min_value=0.0, max_value=100.0, value=6.3, step=0.1, format="%.1f")
        neutrophil = st.number_input("Neutrophil", min_value=0.0, max_value=100.0, value=58.0, step=0.1, format="%.1f")
        lymphocyte = st.number_input("Lymphocyte", min_value=0.0, max_value=100.0, value=33.0, step=0.1, format="%.1f")

    with col2:
        monocyte = st.number_input("Monocyte", min_value=0.0, max_value=100.0, value=6.0, step=0.1, format="%.1f")
        eosinophil = st.number_input("Eosinophil", min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.1f")
        basophil = st.number_input("Basophil", min_value=0.0, max_value=20.0, value=0.0, step=0.1, format="%.1f")
        rbc = st.number_input("RBC", min_value=0.0, max_value=15.0, value=4.77, step=0.01, format="%.2f")
        platelets = st.number_input("Platelets", min_value=0.0, max_value=1000.0, value=70.0, step=0.1, format="%.1f")
        gender = st.selectbox("Gender", ["Male", "Female"])

    submitted = st.form_submit_button("🔍 Predict Dengue Risk")

# -----------------------------
# PREDICTION
# -----------------------------
gender_val = 1 if gender == "Male" else 0

input_data = [
    age,
    haemoglobin,
    esr,
    wbc,
    neutrophil,
    lymphocyte,
    monocyte,
    eosinophil,
    basophil,
    rbc,
    platelets,
    gender_val
]

if submitted:
    X = np.array([input_data], dtype=float)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    st.markdown('<div class="section-title" style="margin-top: 1rem;">🔎 Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-positive">
                ⚠️ DengueAI suggests you may be at <strong>HIGH RISK of Dengue infection</strong>
                based on the current data.
                <div class="confidence">
                    <strong>Model Confidence:</strong> {probability * 100:.2f}% chance of Dengue infection.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-negative">
                ✅ DengueAI suggests you are <strong>not at risk of Dengue infection</strong>
                based on the current data.
                <div class="confidence">
                    <strong>Model Confidence:</strong> {(1 - probability) * 100:.2f}% chance of being healthy.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="prob-card">', unsafe_allow_html=True)
    st.markdown("**📊 Prediction Probabilities**")

    not_dengue = max(0.0, min(1.0, 1 - probability))
    dengue = max(0.0, min(1.0, probability))

    st.markdown(
        f"""
        <div class="prob-row">
            <span>Not Dengue</span>
            <span>{not_dengue * 100:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(not_dengue)

    st.markdown(
        f"""
        <div class="prob-row" style="margin-top:1.5rem;">
            <span>Dengue</span>
            <span>{dengue * 100:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(dengue)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div class="footer-note">
        Research purpose only. Not a medical diagnosis tool.
    </div>
    """,
    unsafe_allow_html=True
)
