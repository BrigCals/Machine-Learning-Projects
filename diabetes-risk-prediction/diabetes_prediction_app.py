import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load trained model
# -----------------------------
# Use a relative path to look in the same directory as the script
# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model file
model_path = os.path.join(script_dir, "diabetes_prediction_model.joblib")

# Alternatively, if Streamlit's working directory is the folder containing the app, this also works:
# model_path = "diabetes_prediction_model.joblib"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at: {model_path}. Please check your file paths.")
    st.stop()


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetes Prediction")
st.write("Enter the following health metrics:")


# -----------------------------
# User inputs
# -----------------------------
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)

glucose = st.number_input(
    "Glucose level (mg/dL)",
    min_value=0.0,
    max_value=300.0,
    value=120.0
)

bloodpressure = st.number_input(
    "Blood Pressure (mmHg)",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

skinthickness = st.number_input(
    "Skin Thickness (mm)",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin level (mu U/ml)",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.number_input(
    "BMI (kg/m²)",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabPedFun = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [bloodpressure],
        "SkinThickness": [skinthickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabPedFun],
        "Age": [age]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.error("Prediction: Diabetic")
    else:
        st.success("Prediction: Non-diabetic")
