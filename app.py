import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Dengue Detection",
    page_icon="🦟",
    layout="centered"
)

# -----------------------------
# CLEAN PROFESSIONAL UI
# -----------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #f5f7fb;
}

/* Force readable text everywhere */
html, body, [class*="css"] {
    color: #111827 !important;
}

/* Title */
h1 {
    color: #111827 !important;
    font-weight: 800;
    text-align: center;
}

/* Sub text */
p {
    color: #374151 !important;
}

/* Button styling */
.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 16px;
    border: none;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    transform: scale(1.02);
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

/* Dataframe */
.stDataFrame {
    background-color: white;
    border-radius: 10px;
}

/* Metrics cards */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🦟 Dengue Detection System")
st.write("Upload CBC CSV file to predict dengue infection risk.")

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
# UPLOAD CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CBC CSV File", type=["csv"])

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Drop unnecessary columns
    drop_cols = ["Serial", "Date", "Result"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Convert Gender
    if "Gender" in df.columns:
        df["Gender_Male"] = df["Gender"].map({
            "Male": 1,
            "Female": 0,
            "M": 1,
            "F": 0
        })
        df.drop(columns=["Gender"], inplace=True)

    # Check missing features
    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Fill missing values
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(MEDIANS[col])

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------
    if st.button("🔍 Predict Dengue Risk"):

        X = df[FEATURES]

        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.50).astype(int)

        results = pd.DataFrame({
            "Prediction": [
                "🟥 Positive" if p == 1 else "🟩 Negative"
                for p in preds
            ],
            "Probability (%)": [
                round(float(p) * 100, 2)
                for p in probs
            ]
        })

        st.subheader("Prediction Results")
        st.dataframe(results, use_container_width=True)

        # -----------------------------
        # SUMMARY
        # -----------------------------
        positive = int(preds.sum())
        negative = len(preds) - positive

        st.subheader("Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(preds))
        col2.metric("Positive Cases", positive)
        col3.metric("Negative Cases", negative)

        # -----------------------------
        # DOWNLOAD
        # -----------------------------
        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Results",
            csv,
            "dengue_results.csv",
            "text/csv"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("⚠ Research purpose only — Not a medical diagnosis tool")
