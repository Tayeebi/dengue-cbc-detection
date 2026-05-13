import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Dengue Risk AI System",
    page_icon="🦟",
    layout="centered"
)

# -----------------------------
# CLEAN PROFESSIONAL UI
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f5f7fb;
}

/* Global text */
html, body, [class*="css"] {
    color: #0f172a !important;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 16px;
    border: none;
    transition: 0.2s ease-in-out;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    transform: scale(1.02);
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

/* Dataframe */
.stDataFrame {
    background-color: white;
    border-radius: 10px;
}

/* Remove weird white-on-white issues */
h1, h2, h3, p {
    color: #0f172a !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# PROFESSIONAL HEADING
# -----------------------------
st.markdown("""
<h1 style="
    text-align:center;
    font-weight:900;
    font-size:36px;
    margin-bottom:5px;
    color:#0f172a;">
    🦟 Dengue Infection Risk Prediction System
</h1>

<p style="
    text-align:center;
    font-size:16px;
    color:#475569;
    margin-top:0px;">
    Machine Learning–Based Clinical Decision Support Tool Using Complete Blood Count (CBC) Parameters<br>
    for Early Detection and Risk Stratification of Dengue Infection
</p>
""", unsafe_allow_html=True)

# -----------------------------
# FEATURES
# -----------------------------
FEATURES = [
    "Age", "Haemoglobin", "ESR", "WBC",
    "Neutrophil", "Lymphocyte", "Monocyte",
    "Eosinophil", "Basophil", "RBC",
    "Platelets", "Gender_Male"
]

MEDIANS = {
    "Age": 30,
    "Haemoglobin": 13,
    "ESR": 16,
    "WBC": 6.3,
    "Neutrophil": 58,
    "Lymphocyte": 33,
    "Monocyte": 6,
    "Eosinophil": 2,
    "Basophil": 0,
    "RBC": 4.77,
    "Platelets": 70,
    "Gender_Male": 1
}

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if os.path.exists("dengue_pipeline.pkl"):
        return joblib.load("dengue_pipeline.pkl")
    st.error("Model file not found: dengue_pipeline.pkl")
    return None

model = load_model()

if model is None:
    st.stop()

# -----------------------------
# UPLOAD FILE
# -----------------------------
uploaded_file = st.file_uploader("Upload CBC CSV File", type=["csv"])

# -----------------------------
# PROCESS & PREDICT
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Remove unwanted columns
    drop_cols = ["Serial", "Date", "Result"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Gender encoding
    if "Gender" in df.columns:
        df["Gender_Male"] = df["Gender"].map({
            "Male": 1,
            "Female": 0,
            "M": 1,
            "F": 0
        })
        df.drop(columns=["Gender"], inplace=True)

    # Validate columns
    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Fill missing values
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(MEDIANS[col])

    # -----------------------------
    # PREDICTION BUTTON
    # -----------------------------
    if st.button("🔍 Run Dengue Risk Prediction"):

        X = df[FEATURES]

        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.50).astype(int)

        results = pd.DataFrame({
            "Prediction": [
                "🟥 High Risk (Positive)" if p == 1 else "🟩 Low Risk (Negative)"
                for p in preds
            ],
            "Risk Probability (%)": [
                round(float(p) * 100, 2)
                for p in probs
            ]
        })

        # Results table
        st.subheader("📊 Prediction Results")
        st.dataframe(results, use_container_width=True)

        # -----------------------------
        # SUMMARY DASHBOARD
        # -----------------------------
        positive = int(preds.sum())
        negative = len(preds) - positive

        st.subheader("📌 Patient Summary Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(preds))
        col2.metric("High Risk Cases", positive)
        col3.metric("Low Risk Cases", negative)

        # -----------------------------
        # DOWNLOAD
        # -----------------------------
        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Prediction Report",
            csv,
            "dengue_risk_report.csv",
            "text/csv"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("⚠ For research and educational purposes only. Not a substitute for medical diagnosis.")
