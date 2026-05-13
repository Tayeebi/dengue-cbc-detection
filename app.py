import streamlit as st
import joblib
import numpy as np
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="DengueAI",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model_path = "dengue_pipeline.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("❌ Model file not found: dengue_pipeline.pkl")
    st.stop()

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_dengue(input_data):
    X = np.array([input_data], dtype=float)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return prediction, probability

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
/* =========================================================
BACKGROUND
========================================================= */
.stApp {
    background-color: #f8fafc;
}

/* =========================================================
MAIN CONTAINER
========================================================= */
.block-container {
    max-width: 900px;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* =========================================================
GLOBAL TEXT
========================================================= */
html, body, [class*="css"] {
    color: #0f172a !important;
}

h1, h2, h3, h4, h5, h6, p, span, label {
    color: #0f172a !important;
}

/* =========================================================
SIMPLE HERO SECTION
========================================================= */
.hero {
    padding: 6rem 0 2rem 0; 
    margin-bottom: 2rem;
    text-align: center;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.5rem;
    color: #0f172a !important;
}

.hero-subtitle {
    color: #475569 !important;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.5;
    max-width: 700px;
    margin: 0 auto;
}

/* =========================================================
SECTION TITLES
========================================================= */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: #1e293b !important;
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 0.8rem;
}

.subtext {
    color: #64748b !important;
    margin-top: 0.8rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}

/* =========================================================
BUTTONS
========================================================= */
.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    box-shadow: 0px 4px 14px rgba(37,99,235,0.25);
    transition: all 0.2s ease !important;
    margin-top: 1rem;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px);
    box-shadow: 0px 6px 20px rgba(37,99,235,0.3);
}

/* FORCE BUTTON TEXT VISIBILITY */
.stButton > button *,
div[data-testid="stFormSubmitButton"] button * {
    color: white !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* =========================================================
INPUT LABELS
========================================================= */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-weight: 600 !important;
    color: #334155 !important;
    font-size: 0.9rem !important;
}

/* =========================================================
RESULT BOX
======================================================== */
.result-good {
    background: #ecfdf5;
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 1.2rem;
    color: #065f46;
    font-weight: 600;
    line-height: 1.5;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}

.result-bad {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 1.2rem;
    color: #991b1b;
    font-weight: 600;
    line-height: 1.5;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}

/* =========================================================
CONFIDENCE
======================================================== */
.confidence {
    margin-top: 1rem;
    font-size: 1rem;
    color: #475569;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* =========================================================
FOOTER
========================================================= */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 3rem;
    font-size: 0.85rem;
    border-top: 1px solid #e2e8f0;
    padding-top: 1rem;
}

/* =========================================================
PROGRESS BAR
========================================================= */
.stProgress > div > div > div {
    border-radius: 999px;
}

/* =========================================================
REMOVE EXTRA GAPS
========================================================= */
div[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

.element-container {
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER (SIMPLE DESIGN)
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">
        DengueAI: Clinical Decision Support
    </div>
    <div class="hero-subtitle">
        AI-powered risk stratification utilizing complete blood count (CBC) laboratory parameters.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT HEADER
# =========================================================
st.markdown("""
<div class="section-title">
    Patient Data Entry
</div>
<div class="subtext">
    Please input the patient's current CBC laboratory results to assess infection risk.
</div>
""", unsafe_allow_html=True)

# =========================================================
# FORM
# =========================================================
with st.form("dengue_form"):
    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        haemoglobin = st.number_input("Haemoglobin", min_value=0.0, max_value=25.0, value=13.0)
        esr = st.number_input("ESR", min_value=0.0, max_value=200.0, value=16.0)
        wbc = st.number_input("WBC", min_value=0.0, max_value=100.0, value=6.3)
        neutrophil = st.number_input("Neutrophil", min_value=0.0, max_value=100.0, value=58.0)
        lymphocyte = st.number_input("Lymphocyte", min_value=0.0, max_value=100.0, value=33.0)

    # RIGHT COLUMN
    with col2:
        monocyte = st.number_input("Monocyte", min_value=0.0, max_value=100.0, value=6.0)
        eosinophil = st.number_input("Eosinophil", min_value=0.0, max_value=100.0, value=2.0)
        basophil = st.number_input("Basophil", min_value=0.0, max_value=20.0, value=0.0)
        rbc = st.number_input("RBC", min_value=0.0, max_value=15.0, value=4.77)
        platelets = st.number_input("Platelets", min_value=0.0, max_value=1000.0, value=70.0)
        gender = st.selectbox("Gender", ["Male", "Female"])

    submitted = st.form_submit_button("Run Risk Assessment")

# =========================================================
# INPUT PREPARATION
# =========================================================
gender_value = 1 if gender == "Male" else 0

input_data = [
    age, haemoglobin, esr, wbc, neutrophil, 
    lymphocyte, monocyte, eosinophil, basophil, 
    rbc, platelets, gender_value
]

# =========================================================
# PREDICTION RESULT
# =========================================================
if submitted:
    prediction, probability = predict_dengue(input_data)

    st.markdown("""
<div class="section-title" style="margin-top: 2rem;">
    Assessment Results
</div>
    """, unsafe_allow_html=True)

    # POSITIVE
    if prediction == 1:
        st.markdown("""
<div class="result-bad">
    ⚠️ <strong>Alert: High Risk of Dengue Infection Detected</strong><br>
    The model indicates a high probability of Dengue based on the provided CBC hematology profile.
</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
<div class="confidence">
    <strong>Diagnostic Confidence:</strong>
    {probability * 100:.1f}% probability of Dengue indicators.
</div>
        """, unsafe_allow_html=True)

    # NEGATIVE
    else:
        st.markdown("""
<div class="result-good">
    ✅ <strong>Low Risk: No Dengue Indicators Detected</strong><br>
    The model indicates the patient is currently at a low risk of Dengue based on the provided CBC hematology profile.
</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
<div class="confidence">
    <strong>Diagnostic Confidence:</strong>
    {(1 - probability) * 100:.1f}% probability of negative indicators.
</div>
        """, unsafe_allow_html=True)

    # =========================================================
    # PROBABILITIES (Wrapper removed)
    # =========================================================
    st.markdown("""
<div class="section-title" style="border:none; padding:0; font-size:1.1rem; margin-top: 1rem;">
    Statistical Breakdown
</div>
    """, unsafe_allow_html=True)

    healthy_prob = 1 - probability
    dengue_prob = probability

    st.write(f"**Negative for Dengue:** {healthy_prob * 100:.1f}%")
    st.progress(float(healthy_prob))

    st.write(f"**Positive for Dengue:** {dengue_prob * 100:.1f}%")
    st.progress(float(dengue_prob))


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    For investigational and research purposes only. This tool does not replace professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
