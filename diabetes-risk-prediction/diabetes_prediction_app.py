import os
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    # Get the folder where this Python file is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path to the saved model
    model_path = os.path.join(
        BASE_DIR,
        "diabetes_model.joblib"
    )

    # Check if the model exists
    if not os.path.exists(model_path):
        st.error(
            f"Model file not found.\n\n"
            f"Expected location:\n{model_path}"
        )
        st.stop()

    # Load the trained model
    try:
        model = joblib.load(model_path)
        return model

    except ModuleNotFoundError as e:
        st.error(
            f"Missing Python module required by the saved model: {e}"
        )
        st.info(
            "Add the missing package to requirements.txt "
            "and redeploy the app."
        )
        st.stop()

    except Exception as e:
        st.error(
            f"Unable to load the model:\n\n{e}"
        )
        st.stop()


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🩺 Diabetes Risk Predictor")

st.write(
    "Enter the patient's information below to predict "
    "the diabetes outcome."
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    bloodpressure = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0

    )


with col2:

    skinthickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    insulin = st.number_input(
        "Insulin (mu U/ml)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )


col3, col4 = st.columns(2)


with col3:

    diabetespedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )


with col4:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
):

    # Create dataframe using the SAME feature order
    # used during model training
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            bloodpressure,
            skinthickness,
            insulin,
            bmi,
            diabetespedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]


        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ The model predicts a positive diabetes outcome."
            )

        else:

            st.success(
                "✅ The model predicts a negative diabetes outcome."
            )


        # --------------------------------------------------
        # PREDICTION PROBABILITY
        # --------------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_data)[0]

            diabetes_probability = probability[1] * 100
            non_diabetes_probability = probability[0] * 100


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Diabetes Probability",
                    f"{diabetes_probability:.2f}%"
                )


            with col2:

                st.metric(
                    "Non-Diabetes Probability",
                    f"{non_diabetes_probability:.2f}%"
                )


        # --------------------------------------------------
        # SHOW INPUT DATA
        # --------------------------------------------------

        st.subheader("Input Data")

        st.dataframe(
            input_data,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"Prediction error:\n\n{e}"
        )