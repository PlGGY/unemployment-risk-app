import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title="Unemployment Risk Predictor", layout="wide")

"""
Streamlit App — Unemployment Risk Predictor 
===================================================================
streamlit, pandas, numpy, joblib, tensorflow
"""

# ---------------------------- Load artefacts -----------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model = keras.models.load_model("unemployment_model.keras", compile=False)
    X_cols = joblib.load("X_train_columns.pkl")
    y_probs = np.load("train_pred_proba.npy")
    thr_20 = np.percentile(y_probs, 80)
    thr_40 = np.percentile(y_probs, 60)
    return model, X_cols, thr_20, thr_40

model, X_train_cols, thr_20, thr_40 = load_assets()

# ---------------------------- Helper ------------------------------------

def predict(user_dict: dict):
    X = pd.DataFrame([user_dict])
    X_enc = pd.get_dummies(X).reindex(columns=X_train_cols, fill_value=0).astype("float32")
    prob = float(model.predict(X_enc, verbose=0).flatten()[0])
    if prob >= thr_20:
        risk = "High Risk"
    elif prob >= thr_40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    return prob, risk

# ---------------------------- UI Layout ----------------------------------

st.title("🧮 Unemployment Risk Predictor")
st.markdown("Drag the sliders or pick values on the left, then click **Predict** to estimate unemployment probability within one month.")

with st.sidebar:
    st.header("Input Features")

    # ---------- Drag‑able sliders ----------
    educ = st.select_slider(
        "Education Level",
        options=[
            "Less than High School", "High School", "Some College",
            "Bachelor's", "Master's", "Doctorate"
        ],
        value="Some College"
    )

    occ = st.select_slider(
        "Occupation Category",
        options=[
            "Management, Business, Science, and Arts",
            "Service Occupations",
            "Sales and Office",
            "Natural Resources, Construction, and Maintenance",
            "Production, Transportation, and Material Moving",
            "Other"
        ],
        value="Service Occupations"
    )

    region = st.select_slider(
        "Region",
        options=["Northeast", "Midwest", "South", "West", "Unknown"],
        value="South"
    )

    division = st.select_slider(
        "Census Division",
        options=[
            "New England", "Middle Atlantic", "East North Central",
            "West North Central", "South Atlantic", "East South Central",
            "West South Central", "Mountain", "Pacific", "State not identified", "Unknown"
        ],
        value="Pacific"
    )

    race = st.select_slider(
        "Race / Ethnicity",
        options=[
            "White", "Black", "Native American", "Asian",
            "Pacific Islander", "Other/Multiple"
        ],
        value="White"
    )

    # ---------- Other inputs (dropdown / radio / slider) ----------
    industry = st.selectbox("Industry Group", [
        "Accommodation and Food Services", "Manufacturing", "Retail Trade",
        "Professional and Technical Services", "Other"])

    immigrant = st.selectbox("Immigration Status", [
        "Native", "Foreign Born – Naturalized", "Foreign Born – Non-Citizen"])

    marital = st.selectbox("Marital Status", [
        "Single", "Married", "Separated/Divorced", "Widowed"])

    disability = st.selectbox("Disability Status", ["No Difficulty", "Difficulty"])

    age = st.slider("Age", 16, 80, 30)

    sex_map = {"Male": 1, "Female": 2}
    sex_choice = st.radio("Sex", list(sex_map.keys()), horizontal=True)
    sex = sex_map[sex_choice]

    predict_btn = st.button("🚀 Predict", type="primary")

# ---------------------------- Prediction display ------------------------
if predict_btn:
    user_input = {
        "educ_category": educ,
        "occupation_category": occ,
        "IND_GROUP": industry,
        "race_category": race,
        "immigration_status": immigrant,
        "marital_status": marital,
        "disability_status": disability,
        "AGE": age,
        "SEX": sex,
        "region_category": region,
        "division_label": division,
    }

    prob, risk = predict(user_input)

    st.subheader("Result")
    st.metric("Predicted Probability", f"{prob:.2%}")

    risk_colors = {"High Risk": "danger", "Medium Risk": "warning", "Low Risk": "success"}
    getattr(st, risk_colors[risk])(f"Risk Level: {risk}")

    with st.expander("Show input details"):
        st.json(user_input)
