import streamlit as st
import pandas as pd
from pathlib import Path
from pycaret.classification import load_model, predict_model

# =====================================================
# App Config (ONLY ONCE in the combined app)
# =====================================================
st.set_page_config(
    page_title="Clinical Risk Prediction Suite",
    layout="centered",
)

# =====================================================
# Shared Sidebar Navigation
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Heart Disease Predictor", "Alzheimer’s Predictor"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Models are loaded once and cached for performance.")

# Base directory where THIS script lives (robust on Streamlit Cloud)
APP_DIR = Path(__file__).resolve().parent


# =====================================================
# Page 1: Alzheimer’s App (wrapped)
# =====================================================
def render_alzheimers_page():
    # -----------------------------
    # Constants (MODEL FEATURES)
    # -----------------------------
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

    # -----------------------------
    # Helpers
    # -----------------------------
    def yesno_to_int(v) -> int:
        return 1 if str(v).strip().lower() in {"yes", "y", "1", "true"} else 0

    def normalize_binary_series(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.astype(int)

        if pd.api.types.is_numeric_dtype(s):
            return (s.fillna(0).astype(float) != 0).astype(int)

        return (
            s.fillna("No")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"yes", "y", "1", "true", "t"})
            .astype(int)
        )

    def safe_load_alz_model():
        """
        Robust loader for Streamlit Cloud:
        - Always tries to load from the same folder as main_app.py (APP_DIR)
        - Tries both PyCaret name-style and explicit .pkl
        """
        candidates = [
            APP_DIR / "alzheimers_catboost_pipeline",      # PyCaret "name" style
            APP_DIR / "alzheimers_catboost_pipeline.pkl",  # explicit file
        ]

        errors = []
        for c in candidates:
            try:
                return load_model(str(c))
            except Exception as e:
                errors.append((str(c), repr(e)))

        raise RuntimeError(
            "Could not load Alzheimer’s model.\n"
            f"Script directory: {APP_DIR}\n"
            f"Working directory: {Path.cwd()}\n"
            f"Files in script directory: {[p.name for p in APP_DIR.iterdir()]}\n"
            "Tried:\n" + "\n".join([f" - {p} -> {err}" for p, err in errors])
        )

    @st.cache_resource
    def get_alz_model():
        return safe_load_alz_model()

    # -----------------------------
    # Load Model
    # -----------------------------
    try:
        model = get_alz_model()
        st.success("✅ Alzheimer’s model loaded successfully")
    except Exception as e:
        st.error("❌ Failed to load Alzheimer’s model.")
        st.exception(e)
        st.stop()

    # -----------------------------
    # UI
    # -----------------------------
    st.title("Alzheimer’s Disease Risk Predictor")
    st.markdown(
        """
This application predicts Alzheimer’s risk using a **CatBoost model**.
Please fill out the clinical details below.
"""
    )

    # -----------------------------
    # Single Prediction
    # -----------------------------
    st.header("Single Patient Prediction")

    with st.form("alz_single_prediction_form"):
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
            SystolicBP = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=60,
                max_value=260,
                value=120,
            )

        st.subheader("Symptoms / History")
        c1, c2 = st.columns(2)
        with c1:
            MemoryComplaints_ui = st.radio(
                "Memory Complaints?",
                ["No", "Yes"],
                horizontal=True,
                key="alz_mem",
            )
        with c2:
            BehavioralProblems_ui = st.radio(
                "Behavioral Problems?",
                ["No", "Yes"],
                horizontal=True,
                key="alz_beh",
            )

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
            "Age": Age,
            "MMSE": MMSE,
            "FunctionalAssessment": FunctionalAssessment,
            "ADL": ADL,
            "DietQuality": DietQuality,
            "PhysicalActivity": PhysicalActivity,
            "AlcoholConsumption": AlcoholConsumption,
            "SystolicBP": SystolicBP,
            "MemoryComplaints": yesno_to_int(MemoryComplaints_ui),
            "BehavioralProblems": yesno_to_int(BehavioralProblems_ui),
        }])

        try:
            results = predict_model(model, data=input_df, raw_score=True)

            if "prediction_score_1" in results.columns:
                p_alz = float(results["prediction_score_1"].iloc[0])
            elif "prediction_score_0" in results.columns:
                p_alz = 1.0 - float(results["prediction_score_0"].iloc[0])
            else:
                p_alz = float(results["prediction_score"].iloc[0])

            THRESH = 0.50
            if p_alz >= THRESH:
                st.error(f"⚠️ High Alzheimer’s Risk (Probability = {p_alz:.2%})")
            else:
                st.success(f"✅ Low Alzheimer’s Risk (Probability = {p_alz:.2%})")

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

    # -----------------------------
    # Batch Prediction
    # -----------------------------
    st.header("Batch Prediction (CSV Upload)")
    st.markdown(
        """
Uploaded CSV must contain the following columns:

- **Numeric:** Age, MMSE, FunctionalAssessment, ADL, DietQuality, PhysicalActivity, AlcoholConsumption, SystolicBP  
- **Binary (0/1 or Yes/No):** MemoryComplaints, BehavioralProblems
"""
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="alz_upload")

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)

            missing = set(EXPECTED_COLS) - set(batch_df.columns)
            if missing:
                st.error(f"❌ Missing required columns: {sorted(missing)}")
                st.stop()

            batch_df = batch_df[EXPECTED_COLS].copy()

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
        except Exception as e:
            st.error("Batch prediction failed.")
            st.exception(e)


# =====================================================
# Page 2: Heart Disease App (wrapped)
# =====================================================
def render_heart_page():
    # ----------------------------
    # Config (absolute paths)
    # ----------------------------
    TARGET_COL = "HeartDisease"

    TEMPLATE_CSV_PATH = APP_DIR / "heart_modified.csv"
    MODEL_CANDIDATES = [
        APP_DIR / "heart_disease_pipeline_bren",      # PyCaret name style
        APP_DIR / "heart_disease_pipeline_bren.pkl",  # explicit file
    ]

    # ----------------------------
    # Styling (your original CSS)
    # ----------------------------
    NAVY = "#163B73"
    HOVER_BLUE = "#2F6FED"

    st.markdown(
        f"""
        <style>
            header {{visibility: hidden;}}
            .stApp {{
                background-color: white;
            }}
            .block-container {{
                padding-top: 1.2rem !important;
            }}

            h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stTextInput label, .stSelectbox label {{
                color: {NAVY} !important;
            }}

            div[data-baseweb="input"] > div,
            div[data-baseweb="select"] > div {{
                border: 1px solid {NAVY} !important;
                border-radius: 10px !important;
                box-shadow: none !important;
            }}

            div[data-baseweb="input"] > div {{
                background: #F8FAFC !important;
            }}
            div[data-baseweb="input"] div[role="spinbutton"] {{
                border-right: none !important;
                border-left: none !important;
                box-shadow: none !important;
            }}
            div[data-baseweb="input"] button {{
                border-left: none !important;
                border-right: none !important;
            }}

            .small-grey {{
                color: #6B7280;
                font-size: 0.9rem;
                margin-top: -6px;
                margin-bottom: 10px;
            }}

            .stButton>button {{
                background-color: {NAVY} !important;
                color: white !important;
                border-radius: 10px !important;
                border: 0px !important;
                padding: 0.65rem 1.2rem !important;
                font-weight: 700 !important;
            }}
            .stButton>button:hover {{
                background-color: {HOVER_BLUE} !important;
                color: white !important;
                opacity: 1 !important;
            }}
            .stButton>button:focus {{
                outline: none !important;
                box-shadow: 0 0 0 3px rgba(47,111,237,0.25) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------
    # Load model once (robust loader)
    # ----------------------------
    @st.cache_resource
    def get_heart_model():
        errors = []
        for c in MODEL_CANDIDATES:
            try:
                return load_model(str(c))
            except Exception as e:
                errors.append((str(c), repr(e)))

        raise RuntimeError(
            "Could not load Heart model.\n"
            f"Script directory: {APP_DIR}\n"
            f"Working directory: {Path.cwd()}\n"
            f"Files in script directory: {[p.name for p in APP_DIR.iterdir()]}\n"
            "Tried:\n" + "\n".join([f" - {p} -> {err}" for p, err in errors])
        )

    @st.cache_data
    def load_template_features() -> pd.DataFrame:
        df_template = pd.read_csv(str(TEMPLATE_CSV_PATH))
        if TARGET_COL in df_template.columns:
            df_template = df_template.drop(columns=[TARGET_COL])
        return df_template

    def build_input_row(age, gender, chol_total, st_slope, resting_diastolic) -> pd.DataFrame:
        df_template = load_template_features()
        row = df_template.iloc[[0]].copy()

        row["Age"] = int(age)
        row["Gender"] = gender
        row["CholesterolTotal"] = float(chol_total)
        row["ST_Slope"] = st_slope
        row["RestingDiastolicBP"] = float(resting_diastolic)

        return row

    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    # ----------------------------
    # Load Model
    # ----------------------------
    try:
        model = get_heart_model()
        st.success("✅ Heart disease model loaded successfully")
    except Exception as e:
        st.error("❌ Failed to load heart disease model.")
        st.exception(e)
        st.stop()

    # ----------------------------
    # UI: Single prediction
    # ----------------------------
    st.title("Heart Disease Risk Predictor")
    st.write("This application predicts **heart disease** using a CatBoost model.")
    st.write("For Clinic Staff Use: Enter key patient measurements and click **Predict**.")

    st.subheader("Single Prediction")
    st.subheader("Enter key patient inputs")

    age = st.number_input("Age", min_value=1, max_value=120, value=55, step=1, key="heart_age")
    st.markdown("<div class='small-grey'>Enter patient age in years (e.g. 55).</div>", unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["M", "F"], index=0, key="heart_gender")
    st.markdown("<div class='small-grey'>Select patient biological sex as recorded (M/F).</div>", unsafe_allow_html=True)

    chol_total = st.number_input(
        "Cholesterol Total (mg/dL)",
        min_value=0.0,
        max_value=1000.0,
        value=220.0,
        step=1.0,
        key="heart_chol",
    )
    st.markdown("<div class='small-grey'>Total cholesterol in mg/dL (e.g. 180–300).</div>", unsafe_allow_html=True)

    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], index=1, key="heart_slope")
    st.markdown("<div class='small-grey'>ST segment slope from ECG: Up / Flat / Down.</div>", unsafe_allow_html=True)

    resting_diastolic = st.number_input(
        "Resting Diastolic BP (mmHg)",
        min_value=0.0,
        max_value=250.0,
        value=130.0,
        step=1.0,
        key="heart_rdbp",
    )
    st.markdown("<div class='small-grey'>Resting diastolic blood pressure (e.g. 70–110+, mmHg).</div>", unsafe_allow_html=True)

    st.write("")

    if st.button("Predict", key="heart_predict"):
        try:
            input_df = build_input_row(
                age=age,
                gender=gender,
                chol_total=chol_total,
                st_slope=st_slope,
                resting_diastolic=resting_diastolic
            )

            pred = predict_model(model, data=input_df)

            label = int(pred.loc[pred.index[0], "prediction_label"])
            score = float(pred.loc[pred.index[0], "prediction_score"])

            st.markdown("---")
            if label == 1:
                st.error("Prediction: **Heart Disease = YES (1)**")
            else:
                st.success("Prediction: **Heart Disease = NO (0)**")

            st.write(f"Prediction confidence (score): **{score:.3f}**")

            st.caption(
                "Important Note: This tool supports screening and decision support only. "
                "Clinical judgment and confirmatory testing are required."
            )

        except Exception as e:
            st.error("Prediction failed. Please check that the model and template CSV are in the same folder.")
            st.exception(e)

    # ----------------------------
    # UI: Batch upload prediction (FULL dataset columns)
    # ----------------------------
    st.markdown("---")
    st.subheader("Batch Prediction (CSV Upload)")
    st.write(
        "Upload a CSV file containing the **full dataset columns** (e.g., Age, Gender, ChestPainType, "
        "RestingDiastolicBP, CholesterolTotal, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, etc.). "
        "The app will generate predictions and allow you to download the results."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="heart_upload")

    if uploaded is not None:
        try:
            batch_df_raw = pd.read_csv(uploaded)

            batch_df = batch_df_raw.copy()
            if TARGET_COL in batch_df.columns:
                batch_df_features = batch_df.drop(columns=[TARGET_COL])
            else:
                batch_df_features = batch_df

            batch_pred = predict_model(model, data=batch_df_features)

            out_df = batch_df_raw.copy()
            out_df["predict heart disease"] = batch_pred["prediction_label"].astype(int)
            out_df["prediction confidence score"] = batch_pred["prediction_score"].astype(float)

            total = len(out_df)
            pos = int((out_df["predict heart disease"] == 1).sum())
            neg = int((out_df["predict heart disease"] == 0).sum())
            avg_score = float(out_df["prediction confidence score"].mean())

            st.markdown("### Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total records", f"{total}")
            c2.metric("Predicted YES (1)", f"{pos}")
            c3.metric("Predicted NO (0)", f"{neg}")
            st.write(f"Average prediction confidence score: **{avg_score:.3f}**")

            st.markdown("### Preview (first 10 rows)")
            st.dataframe(out_df.head(10), use_container_width=True)

            st.download_button(
                label="Download predictions as CSV",
                data=to_csv_bytes(out_df),
                file_name="heart_disease_batch_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("Batch prediction failed. Please ensure your CSV has the correct columns and format.")
            st.exception(e)


# =====================================================
# Router
# =====================================================
if page == "Heart Disease Predictor":
    render_heart_page()
else:
    render_alzheimers_page()
