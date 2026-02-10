import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from io import BytesIO

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

MODEL_NAME = "heart_disease_pipeline_bren"  # my pickle file
TEMPLATE_CSV = "heart_modified.csv"         # should be in same folder as app.py
TARGET_COL = "HeartDisease"

# ----------------------------
# Styling (white background, brighter navy accents)
# ----------------------------
NAVY = "#163B73"          # brighter navy (less "black")
HOVER_BLUE = "#2F6FED"    # brighter hover blue

st.markdown(
    f"""
    <style>
        /* --- Remove top random bar/space --- */
        header {{visibility: hidden;}}
        .stApp {{
            background-color: white;
        }}
        /* reduce top padding that creates the blank bar look */
        .block-container {{
            padding-top: 1.2rem !important;
        }}

        /* --- Typography --- */
        h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stTextInput label, .stSelectbox label {{
            color: {NAVY} !important;
        }}

        /* --- Inputs border / rounding --- */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            border: 1px solid {NAVY} !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }}

        /* --- Make number inputs seamless (reduce inner divider look) --- */
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

        /* --- Button --- */
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


# Load model once
@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

model = get_model()


# Helpers
@st.cache_data
def load_template_features() -> pd.DataFrame:
    """Loads the template feature row (no target) to help construct full feature set when needed."""
    df_template = pd.read_csv(TEMPLATE_CSV)
    if TARGET_COL in df_template.columns:
        df_template = df_template.drop(columns=[TARGET_COL])
    return df_template

def build_input_row(age, gender, chol_total, st_slope, resting_diastolic) -> pd.DataFrame:
    """
    Because the PyCaret pipeline was trained with many columns, a 5-field input alone is not enough.
    We create a single-row dataframe with the same columns as training, using TEMPLATE_CSV as a template row,
    then overwrite the key 5 user inputs.
    """
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


# UI: Single prediction
st.title("Heart Disease Risk Predictor")
st.write("This application predicts **heart disease** using a CatBoost model.")
st.write("For Clinic Staff Use: Enter key patient measurements and click **Predict**.")

st.subheader("Single Prediction")
st.subheader("Enter key patient inputs")

age = st.number_input("Age", min_value=1, max_value=120, value=55, step=1)
st.markdown("<div class='small-grey'>Enter patient age in years (e.g. 55).</div>", unsafe_allow_html=True)

gender = st.selectbox("Gender", ["M", "F"], index=0)
st.markdown("<div class='small-grey'>Select patient biological sex as recorded (M/F).</div>", unsafe_allow_html=True)

chol_total = st.number_input("Cholesterol Total (mg/dL)", min_value=0.0, max_value=1000.0, value=220.0, step=1.0)
st.markdown("<div class='small-grey'>Total cholesterol in mg/dL (e.g. 180–300).</div>", unsafe_allow_html=True)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], index=1)
st.markdown("<div class='small-grey'>ST segment slope from ECG: Up / Flat / Down.</div>", unsafe_allow_html=True)

resting_diastolic = st.number_input("Resting Diastolic BP (mmHg)", min_value=0.0, max_value=250.0, value=130.0, step=1.0)
st.markdown("<div class='small-grey'>Resting diastolic blood pressure (e.g. 70–110+, mmHg).</div>", unsafe_allow_html=True)

st.write("")

if st.button("Predict"):
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


# UI: Batch upload prediction (FULL dataset columns)
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
st.write(
    "Upload a CSV file containing the **full dataset columns** (e.g., Age, Gender, ChestPainType, "
    "RestingDiastolicBP, CholesterolTotal, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, etc.). "
    "The app will generate predictions and allow you to download the results."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        batch_df_raw = pd.read_csv(uploaded)

        # If target exists, keep a copy but do NOT use it for prediction
        batch_df = batch_df_raw.copy()
        if TARGET_COL in batch_df.columns:
            batch_df_features = batch_df.drop(columns=[TARGET_COL])
        else:
            batch_df_features = batch_df

        # Predict using the model (expects full feature set)
        batch_pred = predict_model(model, data=batch_df_features)

        # Add required columns to the ORIGINAL uploaded dataframe (so i will be able to return the same columns user give)
        out_df = batch_df_raw.copy()
        out_df["predict heart disease"] = batch_pred["prediction_label"].astype(int)
        out_df["prediction confidence score"] = batch_pred["prediction_score"].astype(float)

        # Quick summary
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

        # Download
        st.download_button(
            label="Download predictions as CSV",
            data=to_csv_bytes(out_df),
            file_name="heart_disease_batch_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("Batch prediction failed. Please ensure your CSV has the correct columns and format.")
        st.exception(e)
