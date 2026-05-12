# 🦟 Dengue Detection via CBC Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)

An automated clinical decision support system designed to screen for Dengue fever using standard Complete Blood Count (CBC) parameters. This application utilizes an interpretable machine learning pipeline to analyze batch patient data and provide real-time probability scoring.

### 🌐 Live Web App
**[Insert Your Streamlit Link Here]**

---

## 🚀 Key Features

* **End-to-End ML Pipeline:** Integrates a robust `scikit-learn` pipeline handling missing data imputation, standard scaling, Recursive Feature Elimination (RFE), and logistic regression classification.
* **Batch Processing:** Upload patient CBC records via CSV for immediate, automated screening.
* **Clinical-Grade UI:** Designed with a professional, academic aesthetic tailored for medical data presentation.
* **Confidence Scoring:** Outputs not just binary predictions, but exact probability metrics (Bayes-optimal thresholding at 0.50).

## 🛠️ Required CBC Parameters
To generate predictions, the system requires the following 12 features:
`Age`, `Gender`, `Haemoglobin`, `ESR`, `WBC`, `Neutrophil`, `Lymphocyte`, `Monocyte`, `Eosinophil`, `Basophil`, `RBC`, `Platelets`.

---

## 💻 Local Installation & Setup

If you wish to run this system locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
Install dependencies:
(Note: Ensure you are using scikit-learn==1.6.1 to maintain compatibility with the serialized .pkl models).

Bash
pip install -r requirements.txt
Run the Streamlit app:

Bash
streamlit run app.py
📁 Repository Structure
app.py: The core Streamlit application and UI logic.

dengue_pipeline.pkl: The primary bundled ML pipeline (Scaler + RFE + Classifier).

dengue_model.pkl: Fallback standalone classifier.

dengue_features.pkl: Serialized feature names mapping.

requirements.txt: Environment dependencies.


This system was originally developed as an academic course project focusing on the practical application of machine learning architectures in healthcare diagnostics.

⚠️ Clinical Disclaimer: This system is a research prototype and does not replace professional clinical diagnosis. All automated predictions must be reviewed and validated by a qualified medical professional.


**Next Steps:**
1. Save these two new files (`.gitignore` and `README.md`) in your folder.
2. Run these commands in your terminal to push the new files to your GitHub repo:
   * `git add .`
   * `git commit -m "Added README, description, and gitignore"`
   * `git push`

Streamlit will automatically see these new files and update your live app!