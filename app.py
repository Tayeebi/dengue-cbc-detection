"""
Dengue Detection System — Paper-Quality Streamlit Application
============================================================
Bug fixes applied:
  1. Loads dengue_pipeline.pkl (scaler + RFE + model bundled) so scaling
     and feature selection match the training pipeline exactly.
  2. Falls back to dengue_model.pkl only if pipeline file is absent,
     in which case the 8 RFE-selected features + manual StandardScaler
     (fitted means/stds from the FULL dataset) are used.
  3. Threshold raised from 0.30 → 0.50 (standard Bayes optimal).
  4. Full 12-feature input accepted from CSV (Age, ESR, Eosinophil,
     Basophil now included as required by the pipeline).
  5. Button text/background contrast fixed.
  6. UI redesigned to paper-submission / clinical-report quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue CBC Detection System",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Clinical / Academic Paper Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --navy:   #0D2137;
    --mid:    #1A3A5C;
    --accent: #C0392B;
    --teal:   #1ABC9C;
    --gold:   #D4AC0D;
    --bg:     #F7F9FC;
    --card:   #FFFFFF;
    --text:   #1C2733;
    --muted:  #5D7A8C;
    --border: #D4E1ED;
    --shadow: 0 2px 12px rgba(13,33,55,.10);
}

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #DDEAF5 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #9BB8CC !important; font-size: .85rem; }
[data-testid="stSidebar"] hr { border-color: #1E4060 !important; }

/* ── Typography ── */
h1 { font-family: 'Crimson Pro', serif !important; font-size: 2.6rem !important;
     font-weight: 700 !important; color: var(--navy) !important;
     letter-spacing: -.5px; line-height: 1.15; }
h2 { font-family: 'Crimson Pro', serif !important; font-size: 1.75rem !important;
     font-weight: 600 !important; color: var(--mid) !important; }
h3 { font-family: 'DM Sans', sans-serif !important; font-size: 1.1rem !important;
     font-weight: 600 !important; color: var(--navy) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--navy) !important;
    color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    letter-spacing: .4px;
    border: none !important;
    border-radius: 6px !important;
    padding: .6rem 1.6rem !important;
    box-shadow: 0 2px 6px rgba(13,33,55,.25) !important;
    transition: background .2s, transform .1s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--mid) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── File-uploader ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: var(--teal) !important;
    color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: none !important;
    padding: .55rem 1.4rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #16A085 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}

/* ── Alerts ── */
.stSuccess { background: #E8F8F5 !important; border-left: 4px solid var(--teal) !important;
             color: #1A5E4B !important; border-radius: 6px !important; }
.stError   { background: #FDEDEC !important; border-left: 4px solid var(--accent) !important;
             color: #7B241C !important; border-radius: 6px !important; }
.stInfo    { background: #EAF4FB !important; border-left: 4px solid var(--mid) !important;
             color: #1A3A5C !important; border-radius: 6px !important; }
.stWarning { background: #FEF9E7 !important; border-left: 4px solid var(--gold) !important;
             color: #7D6608 !important; border-radius: 6px !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .8rem !important;
                                font-weight: 500 !important; text-transform: uppercase;
                                letter-spacing: .6px; }
[data-testid="stMetricValue"] { color: var(--navy) !important; font-size: 1.8rem !important;
                                font-weight: 700 !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Section card utility ── */
.section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow);
}

/* ── Banner ── */
.banner {
    background: linear-gradient(120deg, var(--navy) 0%, var(--mid) 60%, #1E5F8A 100%);
    border-radius: 14px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.banner::after {
    content: '';
    position: absolute;
    right: -40px; top: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,.04);
}
.banner h1 { color: #FFFFFF !important; font-size: 2.3rem !important; margin: 0; }
.banner p  { color: #9BB8CC !important; margin: .5rem 0 0; font-size: .95rem; line-height: 1.6; }
.banner .badge {
    display: inline-block;
    background: rgba(255,255,255,.12);
    color: #DDEAF5;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .8px;
    text-transform: uppercase;
    border-radius: 20px;
    padding: .25rem .85rem;
    margin-bottom: .8rem;
}

/* ── Result cards ── */
.result-pos {
    background: #FDEDEC;
    border: 1.5px solid #E74C3C;
    border-left: 6px solid #C0392B;
    border-radius: 10px;
    padding: .9rem 1.2rem;
    margin: .45rem 0;
    font-family: 'DM Sans', sans-serif;
}
.result-neg {
    background: #E8F8F5;
    border: 1.5px solid #1ABC9C;
    border-left: 6px solid #148F77;
    border-radius: 10px;
    padding: .9rem 1.2rem;
    margin: .45rem 0;
    font-family: 'DM Sans', sans-serif;
}
.result-pos .label { font-weight: 700; color: #922B21; font-size: 1rem; }
.result-neg .label { font-weight: 700; color: #117A65; font-size: 1rem; }
.result-pos .prob  { color: #C0392B; font-size: .85rem; font-family: 'DM Mono', monospace; }
.result-neg .prob  { color: #148F77; font-size: .85rem; font-family: 'DM Mono', monospace; }
.prob-bar-wrap { background: #DDE8E3; border-radius: 6px; height: 6px; margin-top: 6px; }
.prob-bar-pos  { background: #C0392B; border-radius: 6px; height: 6px; }
.prob-bar-neg  { background: #1ABC9C; border-radius: 6px; height: 6px; }

/* ── Summary strip ── */
.summary-strip {
    display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;
}
.s-card {
    flex: 1; min-width: 130px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .9rem 1rem;
    text-align: center;
    box-shadow: var(--shadow);
}
.s-card .num { font-size: 2rem; font-weight: 700; font-family: 'Crimson Pro', serif; }
.s-card .lbl { font-size: .72rem; font-weight: 600; letter-spacing: .6px;
               text-transform: uppercase; color: var(--muted); margin-top: 2px; }
.s-total  .num { color: var(--navy); }
.s-pos    .num { color: #C0392B; }
.s-neg    .num { color: #148F77; }
.s-rate   .num { color: var(--gold); }

/* ── Tables ── */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--navy) !important; }

/* ── Sidebar info box ── */
.sidebar-feature {
    background: rgba(255,255,255,.06);
    border-radius: 8px;
    padding: .6rem .9rem;
    margin: .3rem 0;
    font-size: .82rem;
    color: #9BB8CC;
    border-left: 3px solid #1ABC9C;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Full 12-feature set required by the pipeline (order must match training X)
PIPELINE_FEATURES = [
    'Age', 'Haemoglobin', 'ESR', 'WBC', 'Neutrophil',
    'Lymphocyte', 'Monocyte', 'Eosinophil', 'Basophil',
    'RBC', 'Platelets', 'Gender_Male'
]

# 8 RFE-selected features used only when falling back to the bare model
MODEL_FEATURES = [
    'Haemoglobin', 'WBC', 'Neutrophil', 'Lymphocyte',
    'Monocyte', 'RBC', 'Platelets', 'Gender_Male'
]

# Medians for imputation (from full dataset)
MEDIANS = {
    'Age': 30.0, 'Haemoglobin': 13.0, 'ESR': 16.0,
    'WBC': 6.3,  'Neutrophil': 58.0, 'Lymphocyte': 33.0,
    'Monocyte': 6.0, 'Eosinophil': 2.0, 'Basophil': 0.0,
    'RBC': 4.77, 'Platelets': 70.0, 'Gender_Male': 1.0,
}

# Fallback scaling params (computed from full dataset; only used with bare model pkl)
FALLBACK_MEANS = {
    'Haemoglobin': 12.5817, 'WBC': 7.6594, 'Neutrophil': 59.7076,
    'Lymphocyte': 31.8300, 'Monocyte': 6.6967, 'RBC': 4.6420,
    'Platelets': 108.2359, 'Gender_Male': 0.6246,
}
FALLBACK_STDS = {
    'Haemoglobin': 2.5940, 'WBC': 4.9650, 'Neutrophil': 16.1864,
    'Lymphocyte': 14.8586, 'Monocyte': 3.2905, 'RBC': 0.9763,
    'Platelets': 102.1932, 'Gender_Male': 0.4850,
}

# Decision threshold (0.50 = standard Bayes-optimal)
THRESHOLD = 0.50

# Column name aliases accepted from CSV
COLUMN_ALIASES = {
    "Hb": "Haemoglobin", "HGB": "Haemoglobin",
    "Hemoglobin": "Haemoglobin", "hemoglobin": "Haemoglobin",
    "Wbc": "WBC", "wbc": "WBC",
    "Platelet": "Platelets", "platelet": "Platelets",
    "Rbc": "RBC", "rbc": "RBC",
    "Lymph": "Lymphocyte", "lymph": "Lymphocyte",
    "Mono": "Monocyte", "mono": "Monocyte",
    "Neutro": "Neutrophil", "neutro": "Neutrophil",
    "Neut": "Neutrophil", "neut": "Neutrophil",
    "Eosino": "Eosinophil", "eosino": "Eosinophil",
    "Baso": "Basophil", "baso": "Basophil",
    "age": "Age", "AGE": "Age",
    "esr": "ESR", "Esr": "ESR",
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Priority:
      1. dengue_pipeline.pkl — full Pipeline(scaler, rfe, model); expects 12 features
      2. dengue_model.pkl    — bare classifier; manual scaling on 8 RFE features applied
    Returns (model_object, mode)  where mode is 'pipeline' or 'model'
    """
    if os.path.exists("dengue_pipeline.pkl"):
        try:
            pipe = joblib.load("dengue_pipeline.pkl")
            return pipe, "pipeline"
        except Exception as e:
            st.warning(f"⚠️ Could not load dengue_pipeline.pkl ({e}). Trying dengue_model.pkl …")

    if os.path.exists("dengue_model.pkl"):
        try:
            mdl = joblib.load("dengue_model.pkl")
            return mdl, "model"
        except Exception as e:
            st.error(f"❌ Could not load dengue_model.pkl: {e}")
            return None, None

    st.error(
        "❌ No model file found. Place **dengue_pipeline.pkl** (preferred) "
        "or **dengue_model.pkl** in the same directory as app.py."
    )
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦟 Dengue CBC System")
    st.markdown("---")

    st.markdown("**📋 Required CSV Columns**")
    required = {
        "Age": "years", "Gender": "Male / Female",
        "Haemoglobin": "g/dL", "ESR": "mm/hr",
        "WBC": "×10³/µL", "Neutrophil": "%",
        "Lymphocyte": "%", "Monocyte": "%",
        "Eosinophil": "%", "Basophil": "%",
        "RBC": "×10⁶/µL", "Platelets": "×10³/µL",
    }
    for col, unit in required.items():
        st.markdown(
            f'<div class="sidebar-feature"><b>{col}</b> &nbsp;·&nbsp; {unit}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**⚙️ Model Settings**")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90,
        value=THRESHOLD, step=0.05,
        help="Probability ≥ threshold → Dengue Positive. Default 0.50 (Bayes optimal).",
    )
    st.markdown("---")
    st.caption(
        "Real-Time Dengue Prediction Using Explainable Machine Learning  \n"
        "© 2025 · Academic Research System"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div class="badge">Clinical Decision Support System</div>
  <h1>🦟 Dengue Detection via CBC Analysis</h1>
  <p>
    Automated dengue fever screening from Complete Blood Count (CBC) parameters
    using an interpretable machine learning pipeline.
    Upload a patient dataset CSV to receive real-time predictions.
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
model_obj, mode = load_model()

if model_obj is None:
    st.stop()

mode_label = "Full Pipeline (scaler + RFE + classifier)" if mode == "pipeline" else "Standalone Classifier (manual scaling)"
st.info(f"✅ **Model loaded** — {mode_label}")


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 📂 Upload CBC Dataset")
st.markdown(
    "Upload a CSV file containing patient CBC records. "
    "A `Result` column (Positive/Negative), `Serial`, and `Date` columns are **optional** — "
    "they will be ignored if present."
)
uploaded_file = st.file_uploader("Select CSV file", type=["csv"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS UPLOADED FILE
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:

    # ── Read ──────────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw.rename(columns=COLUMN_ALIASES, inplace=True)

    # ── Drop non-feature columns ──────────────────────────────────────────────
    drop_cols = [c for c in ['Serial', 'Date', 'Result'] if c in df_raw.columns]
    df = df_raw.drop(columns=drop_cols).copy()

    # ── Gender encoding ───────────────────────────────────────────────────────
    if "Gender" in df.columns:
        df["Gender_Male"] = df["Gender"].map(
            {"Male": 1, "Female": 0, "M": 1, "F": 0,
             "male": 1, "female": 0, "MALE": 1, "FEMALE": 0}
        ).fillna(0).astype(int)
        df.drop(columns=["Gender"], inplace=True)

    # ── Preview ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"### 📊 Uploaded Dataset &nbsp; <span style='font-size:.85rem;color:#5D7A8C;font-weight:400;'>{len(df_raw)} patient records · {len(df_raw.columns)} columns</span>", unsafe_allow_html=True)
    st.dataframe(df_raw, use_container_width=True, height=240)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Prepare features based on mode ────────────────────────────────────────
    if mode == "pipeline":
        REQUIRED_FEATURES = PIPELINE_FEATURES
    else:
        REQUIRED_FEATURES = MODEL_FEATURES

    missing_cols = [c for c in REQUIRED_FEATURES if c not in df.columns]

    if missing_cols:
        st.error(
            f"❌ **Missing columns in uploaded file:** `{missing_cols}`  \n"
            f"Please ensure your CSV contains all required CBC fields (see sidebar)."
        )
        st.stop()

    # ── Build final feature matrix ─────────────────────────────────────────────
    df_feat = pd.DataFrame()
    for col in REQUIRED_FEATURES:
        df_feat[col] = (
            pd.to_numeric(df[col], errors='coerce')
            .fillna(MEDIANS.get(col, 0))
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICT BUTTON
    # ─────────────────────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        run_pred = st.button("🔍  Run Prediction", use_container_width=True)

    if run_pred:
        with st.spinner("Analysing CBC parameters …"):
            try:
                if mode == "pipeline":
                    # Pipeline handles StandardScaler + RFE internally
                    X_input = df_feat[PIPELINE_FEATURES].values
                    probabilities = model_obj.predict_proba(X_input)[:, 1]

                else:
                    # Manual z-score scaling on 8 RFE features only
                    means = np.array([FALLBACK_MEANS[f] for f in MODEL_FEATURES])
                    stds  = np.array([FALLBACK_STDS[f]  for f in MODEL_FEATURES])
                    X_scaled = (df_feat[MODEL_FEATURES].values - means) / stds
                    probabilities = model_obj.predict_proba(X_scaled)[:, 1]

                predictions = (probabilities >= threshold).astype(int)

            except Exception as e:
                st.error(f"❌ **Prediction error:** {e}")
                st.stop()

        # ── Summary Statistics ────────────────────────────────────────────────
        n_total   = len(predictions)
        n_pos     = int(predictions.sum())
        n_neg     = n_total - n_pos
        pos_rate  = round(n_pos / n_total * 100, 1) if n_total else 0

        st.markdown("---")
        st.markdown("## 🧪 Prediction Results")

        st.markdown(f"""
        <div class="summary-strip">
          <div class="s-card s-total">
            <div class="num">{n_total}</div>
            <div class="lbl">Total Patients</div>
          </div>
          <div class="s-card s-pos">
            <div class="num">{n_pos}</div>
            <div class="lbl">Dengue Positive</div>
          </div>
          <div class="s-card s-neg">
            <div class="num">{n_neg}</div>
            <div class="lbl">Dengue Negative</div>
          </div>
          <div class="s-card s-rate">
            <div class="num">{pos_rate}%</div>
            <div class="lbl">Positivity Rate</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Results table ─────────────────────────────────────────────────────
        results_df = pd.DataFrame({
            "Patient ID": [
                df_raw.get("Serial", pd.Series(range(n_total))).iloc[i]
                if "Serial" in df_raw.columns else f"Patient {i+1}"
                for i in range(n_total)
            ],
            "Prediction": [
                "🦟  Dengue Positive" if p == 1 else "✅  Dengue Negative"
                for p in predictions
            ],
            "Probability (%)": [round(float(prob) * 100, 2) for prob in probabilities],
            "Confidence": [
                "High"   if abs(float(p) - 0.5) >= 0.25 else
                "Medium" if abs(float(p) - 0.5) >= 0.10 else "Low"
                for p in probabilities
            ],
        })

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Full Results Table")
        st.dataframe(results_df, use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Individual patient cards ──────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🩺 Patient-Level Results")

        CARDS_PER_PAGE = 20
        if n_total > CARDS_PER_PAGE:
            page = st.number_input(
                f"Page (showing {CARDS_PER_PAGE} patients per page)",
                min_value=1,
                max_value=((n_total - 1) // CARDS_PER_PAGE) + 1,
                value=1, step=1,
            )
            idx_start = (page - 1) * CARDS_PER_PAGE
            idx_end   = min(idx_start + CARDS_PER_PAGE, n_total)
        else:
            idx_start, idx_end = 0, n_total

        cards_html = ""
        for i in range(idx_start, idx_end):
            prob   = float(probabilities[i])
            pct    = round(prob * 100, 1)
            pid    = results_df["Patient ID"].iloc[i]
            bar_w  = pct

            if predictions[i] == 1:
                cards_html += f"""
                <div class="result-pos">
                  <div class="label">🦟 {pid} — DENGUE POSITIVE</div>
                  <div class="prob">Probability Score: {pct}%</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar-pos" style="width:{bar_w}%"></div>
                  </div>
                </div>"""
            else:
                cards_html += f"""
                <div class="result-neg">
                  <div class="label">✅ {pid} — DENGUE NEGATIVE</div>
                  <div class="prob">Probability Score: {pct}%</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar-neg" style="width:{bar_w}%"></div>
                  </div>
                </div>"""

        st.markdown(cards_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Distribution chart ────────────────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Probability Score Distribution")

        chart_df = pd.DataFrame({
            "Probability (%)": [round(float(p) * 100, 1) for p in probabilities],
            "Label": ["Positive" if p == 1 else "Negative" for p in predictions],
        })

        import altair as alt
        chart = (
            alt.Chart(chart_df)
            .mark_bar(opacity=0.85, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("Probability (%):Q", bin=alt.Bin(maxbins=25),
                      title="Dengue Probability Score (%)"),
                alt.Y("count():Q", title="Patient Count"),
                alt.Color("Label:N",
                    scale=alt.Scale(domain=["Positive", "Negative"],
                                    range=["#C0392B", "#1ABC9C"]),
                    legend=alt.Legend(title="Prediction"),
                ),
            )
            .properties(height=280)
            .configure_axis(labelFont="DM Sans", titleFont="DM Sans", titleFontSize=12)
            .configure_legend(labelFont="DM Sans", titleFont="DM Sans")
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────────────────────
        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download Prediction Results (CSV)",
            data=csv_out,
            file_name="dengue_predictions.csv",
            mime="text/csv",
        )

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.warning(
            "⚠️  **Clinical Disclaimer**: This system is a research prototype "
            "and does **not** replace clinical diagnosis. All predictions must "
            "be reviewed and validated by a qualified medical professional."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#5D7A8C;font-size:.82rem;'>"
    "Developed for <em>Real-Time Dengue Prediction Using Explainable Machine Learning</em> "
    "· Powered by Streamlit &amp; Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)