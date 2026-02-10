import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# =====================================================
# App Config
# =====================================================
st.set_page_config(page_title="Alzheimer’s Risk Prediction", layout="centered")

# =====================================================
# Constants (NEW MODEL FEATURES)
# =====================================================
NUMERIC_FEATURES = [
    "Age",
    "MMSE",
    "FunctionalAssessment",
    "ADL",
    "DietQuality",
    "PhysicalActivity",
    "AlcoholConsumption",
    "SystolicBP",
]
BINARY_FEATURES = [
    "MemoryComplaints",
    "BehavioralProblems",
]
EXPECTED_COLS = NUMERIC_FEATURES + BINARY_FEATURES

# =====================================================
# Helpers
# =====================================================
def yesno_to_int(v) -> int:
    """Map UI strings to 0/1."""
    return 1 if str(v).strip().lower() in {"yes", "y", "1", "true"} else 0

def normalize_binary_series(s: pd.Series) -> pd.Series:
    """
    Accepts 0/1, True/False, Yes/No-like strings.
    Returns int 0/1.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        # Treat any nonzero as 1; keep 0 as 0
        return (s.fillna(0).astype(float) != 0).astype(int)

    # Strings / mixed
    return (
        s.fillna("No")
         .astype(str)
         .str.strip()
         .str.lower()
         .isin({"yes", "y", "1", "true", "t"})
         .astype(int)
    )

def safe_load_model():
    """
    Load the saved PyCaret pipeline/model.
    Your artifact name suggests the file is: alzheimers_catboost_pipeline.pkl
    PyCaret's load_model usually expects the name without extension, but we try both.
    """
    for name in ["alzheimers_catboost_pipeline", "alzheimers_catboost_pipeline.pkl"]:
        try:
            return load_model(name)
        except Exception:
            pass
    raise RuntimeError(
        "Could not load model. Ensure 'alzheimers_catboost_pipeline.pkl' is in the working directory."
    )

# =====================================================
# Load Model
# =====================================================
@st.cache_resource
def load_alz_model():
    return safe_load_model()

model = load_alz_model()
st.success("✅ Model loaded successfully")

# =====================================================
# Header
# =====================================================
st.title("🧠 Alzheimer’s Disease Risk Predictor")

st.markdown(
    """
This application predicts Alzheimer’s risk using a **CatBoost model**.
Please fill out the clinical details below.
"""
)

# =====================================================
# Single Prediction
# =====================================================
st.header("👤 Single Patient Prediction")

with st.form("single_prediction_form"):
    st.subheader("Clinical Metrics")
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=0, max_value=120, value=65)
        MMSE = st.slider("MMSE Score", 0, 30, 26)
        FunctionalAssessment = st.slider("Functional Assessment", 0, 10, 7)
        ADL = st.slider("Activities of Daily Living (0–10)", 0, 10, 8)

    with col2:
        DietQuality = st.slider("Diet Quality (1–10)", 1, 10, 5)
        PhysicalActivity = st.slider("Physical Activity (0–10)", 0, 10, 3)
        AlcoholConsumption = st.slider("Alcohol Consumption (units/week)", 0, 40, 2)
        SystolicBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=260, value=120)

    st.subheader("Symptoms / History")
    c1, c2 = st.columns(2)
    with c1:
        MemoryComplaints_ui = st.radio("Memory Complaints?", ["No", "Yes"], horizontal=True)
    with c2:
        BehavioralProblems_ui = st.radio("Behavioral Problems?", ["No", "Yes"], horizontal=True)

    submit = st.form_submit_button("🔍 Predict")

if submit:
    input_df = pd.DataFrame([{
        # Numeric
        "Age": Age,
        "MMSE": MMSE,
        "FunctionalAssessment": FunctionalAssessment,
        "ADL": ADL,
        "DietQuality": DietQuality,
        "PhysicalActivity": PhysicalActivity,
        "AlcoholConsumption": AlcoholConsumption,
        "SystolicBP": SystolicBP,

        # Binary (0/1)
        "MemoryComplaints": yesno_to_int(MemoryComplaints_ui),
        "BehavioralProblems": yesno_to_int(BehavioralProblems_ui),
    }])

    results = predict_model(model, data=input_df, raw_score=True)


    # Prefer probability of class 1 if it exists
    if "prediction_score_1" in results.columns:
        p_alz = float(results["prediction_score_1"].iloc[0])
    elif "prediction_score_0" in results.columns:
        # If your positive class is actually 0, you’ll switch to this (see section below)
        p_alz = 1.0 - float(results["prediction_score_0"].iloc[0])
    else:
        # Fallback: least reliable, but prevents crashes
        p_alz = float(results["prediction_score"].iloc[0])

    label = results["prediction_label"].iloc[0]


    THRESH = 0.50
    if p_alz >= THRESH:
        st.error(f"⚠️ High Alzheimer’s Risk (Probability = {p_alz:.2%})")
    else:
        st.success(f"✅ Low Alzheimer’s Risk (Probability = {p_alz:.2%})")

# =====================================================
# Batch Prediction
# =====================================================
st.header("📂 Batch Prediction (CSV Upload)")

st.markdown(
    """
Uploaded CSV must contain the following columns:

- **Numeric:** Age, MMSE, FunctionalAssessment, ADL, DietQuality, PhysicalActivity, AlcoholConsumption, SystolicBP  
- **Binary (0/1 or Yes/No):** MemoryComplaints, BehavioralProblems
"""
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    missing = set(EXPECTED_COLS) - set(batch_df.columns)
    if missing:
        st.error(f"❌ Missing required columns: {sorted(missing)}")
        st.stop()

    # Keep only the model’s columns (ignore extras safely)
    batch_df = batch_df[EXPECTED_COLS].copy()

    # Normalize binary columns to 0/1
    for col in BINARY_FEATURES:
        batch_df[col] = normalize_binary_series(batch_df[col])

    preds = predict_model(model, data=batch_df)

    st.subheader("Prediction Results")
    st.dataframe(preds, use_container_width=True)

    csv = preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Predictions",
        data=csv,
        file_name="alzheimers_predictions.csv",
        mime="text/csv",
    )