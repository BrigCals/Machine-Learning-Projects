import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD TRAINED MODEL
# Automatically searches the repository for the model file
# ============================================================

@st.cache_resource
def load_model():

    # Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Search through all folders and subfolders
    for root, dirs, files in os.walk(base_dir):

        if "diabetes_prediction_model.joblib" in files:

            model_path = os.path.join(
                root,
                "diabetes_prediction_model.joblib"
            )

            return joblib.load(model_path)

    # Model was not found
    return None


model = load_model()


# ============================================================
# HEADER AND DESCRIPTION
# ============================================================

st.markdown(
    """
    <div style='text-align: center;
                padding: 20px;
                border-bottom: 2px solid #4CAF50;
                margin-bottom: 30px;'>

        <h1 style='color: #2E7D32;'>
            🩺 Diabetes Risk Assessment Tool
        </h1>

        <p style='font-size: 1.1em; color: #555;'>
            Enter your health metrics below to predict the likelihood
            of diabetes. This tool uses a machine learning model
            based on clinical data.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK IF MODEL EXISTS
# ============================================================

if model is None:

    st.error(
        "⚠️ Model file not found! "
        "Please ensure 'diabetes_prediction_model.joblib' "
        "is somewhere in the repository."
    )

    st.stop()


# ============================================================
# USER INPUT SECTION
# ============================================================

with st.container():

    st.markdown("### 📋 Patient Health Data")

    # Form prevents the app from predicting
    # every time an input changes
    with st.form("patient_data_form"):

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with col1:

            st.markdown("**Personal Info**")

            age = st.number_input(
                "Age (Years)",
                min_value=1,
                max_value=120,
                value=30,
                step=1,
                help="Age in years."
            )

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Number of times pregnant."
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=70.0,
                value=25.0,
                format="%.1f",
                help="Body mass index (weight in kg/(height in m)^2)."
            )


        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with col2:

            st.markdown("**Blood Tests**")

            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0.0,
                max_value=300.0,
                value=120.0,
                format="%.1f",
                help="Plasma glucose concentration from a 2-hour oral glucose tolerance test."
            )

            insulin = st.number_input(
                "Insulin (μU/ml)",
                min_value=0.0,
                max_value=900.0,
                value=80.0,
                format="%.1f",
                help="2-hour serum insulin."
            )


        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with col3:

            st.markdown("**Other Metrics**")

            bloodpressure = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                format="%.1f",
                help="Diastolic blood pressure."
            )

            skinthickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                format="%.1f",
                help="Triceps skin fold thickness."
            )

            diabPedFun = st.number_input(
                "Diabetes Pedigree",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                format="%.3f",
                help="Diabetes pedigree function (genetic risk score)."
            )


        # ----------------------------------------------------
        # SUBMIT BUTTON
        # ----------------------------------------------------

        st.markdown("<br>", unsafe_allow_html=True)

        submit_button = st.form_submit_button(
            label="🔍 Analyze Risk",
            use_container_width=True
        )


# ============================================================
# PREDICTION
# ============================================================

if submit_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner("Analyzing patient data..."):

        try:

            prediction = model.predict(input_data)


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown("---")

            st.markdown("### 📊 Assessment Result")


            # ------------------------------------------------
            # DIABETES PREDICTED
            # ------------------------------------------------

            if prediction[0] == 1:

                st.error(
                    "#### 🔴 High Risk: Diabetic Profile Detected"
                )

                st.info(
                    "The model indicates a higher predicted likelihood "
                    "of diabetes based on the provided metrics. "
                    "**Please consult with a healthcare professional "
                    "for a formal diagnosis and advice.**"
                )


            # ------------------------------------------------
            # NO DIABETES PREDICTED
            # ------------------------------------------------

            else:

                st.success(
                    "#### 🟢 Low Risk: Non-Diabetic Profile Detected"
                )

                st.info(
                    "The model indicates a lower predicted likelihood "
                    "of diabetes based on the provided metrics. "
                    "This prediction should not be considered a "
                    "medical diagnosis."
                )


        # ----------------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------------

        except Exception as e:

            st.error(
                f"An error occurred during prediction: {e}"
            )

            st.write(
                "Please check the input data format and "
                "model compatibility."
            )


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <p style='text-align: center;
              font-size: 0.8em;
              color: gray;'>

        Disclaimer: This application is for educational purposes
        only and should not be used as a substitute for professional
        medical advice, diagnosis, or treatment.

    </p>
    """,
    unsafe_allow_html=True
)