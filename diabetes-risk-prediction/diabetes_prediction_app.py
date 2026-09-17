import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


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
# MODEL PATH
# ============================================================

# Get the folder where this Python file is located
APP_DIR = Path(__file__).resolve().parent

# The model should be in the SAME folder as this app
MODEL_PATH = APP_DIR / "diabetes_prediction_model.joblib"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained machine learning model.

    The model file is expected to be located in the same
    directory as this Streamlit application.
    """

    if not MODEL_PATH.exists():
        return None, (
            f"Model file was not found.\n\n"
            f"Expected location:\n{MODEL_PATH}"
        )

    try:
        model = joblib.load(MODEL_PATH)
        return model, None

    except Exception as e:
        return None, (
            f"Could not load the trained model.\n\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}"
        )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        padding: 20px;
        border-bottom: 2px solid #4CAF50;
        margin-bottom: 30px;
    }

    .main-title h1 {
        color: #2E7D32;
        margin-bottom: 10px;
    }

    .main-title p {
        font-size: 1.1rem;
        color: #555;
    }

    .section-title {
        color: #2E7D32;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: gray;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        <h1>🩺 Diabetes Risk Assessment Tool</h1>
        <p>
            Enter the patient's health measurements below to obtain
            a machine-learning-based diabetes risk classification.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🩺 About the Tool")

    st.write(
        """
        This application uses a trained machine learning model
        to classify a patient's diabetes risk based on selected
        clinical measurements.
        """
    )

    st.divider()

    st.subheader("Input Variables")

    st.write(
        """
        • Pregnancies  
        • Glucose  
        • Blood Pressure  
        • Skin Thickness  
        • Insulin  
        • BMI  
        • Diabetes Pedigree Function  
        • Age
        """
    )

    st.divider()

    st.caption(
        "For educational and demonstration purposes only."
    )


# ============================================================
# PATIENT INPUT
# ============================================================

st.markdown(
    '<h2 class="section-title">📋 Patient Health Data</h2>',
    unsafe_allow_html=True
)


with st.form("patient_data_form"):

    col1, col2, col3 = st.columns(3)


    # ========================================================
    # COLUMN 1
    # ========================================================

    with col1:

        st.markdown("### 👤 Personal Information")

        age = st.number_input(
            "Age (Years)",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
            help="Patient age in years."
        )

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of times the patient has been pregnant."
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            format="%.1f",
            help="Body Mass Index."
        )


    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:

        st.markdown("### 🩸 Blood Tests")

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            format="%.1f",
            help=(
                "Plasma glucose concentration measured during "
                "the oral glucose tolerance test."
            )
        )

        insulin = st.number_input(
            "Insulin (μU/ml)",
            min_value=0.0,
            max_value=900.0,
            value=80.0,
            format="%.1f",
            help="Two-hour serum insulin."
        )


    # ========================================================
    # COLUMN 3
    # ========================================================

    with col3:

        st.markdown("### 📊 Other Metrics")

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
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            format="%.3f",
            help="Diabetes pedigree function."
        )


    # ========================================================
    # SUBMIT
    # ========================================================

    st.markdown("")

    submit_button = st.form_submit_button(
        label="🔍 Analyze Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submit_button:

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame(
        {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [bloodpressure],
            "SkinThickness": [skinthickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabPedFun],
            "Age": [age]
        }
    )


    # ========================================================
    # DISPLAY INPUT DATA
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<h2 class="section-title">📊 Entered Patient Data</h2>',
        unsafe_allow_html=True
    )

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    with st.spinner("Loading trained model..."):

        model, model_error = load_model()


    # ========================================================
    # MODEL LOADING ERROR
    # ========================================================

    if model is None:

        st.error("⚠️ The trained model could not be loaded.")

        st.code(
            model_error,
            language="text"
        )

        st.warning(
            """
            Please verify that:

            1. The model file exists in the same folder as this app.
            2. The model was saved correctly.
            3. The scikit-learn version used to load the model is
               compatible with the version used during training.
            """
        )

        st.stop()


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    with st.spinner("Analyzing patient data..."):

        try:

            # ------------------------------------------------
            # MAKE PREDICTION
            # ------------------------------------------------

            prediction = model.predict(input_data)

            predicted_class = int(prediction[0])


            # ====================================================
            # ASSESSMENT RESULT
            # ====================================================

            st.markdown("---")

            st.markdown(
                '<h2 class="section-title">📋 Assessment Result</h2>',
                unsafe_allow_html=True
            )


            # ====================================================
            # CLASS 1
            # ====================================================

            if predicted_class == 1:

                st.error(
                    "### 🔴 Higher-Risk Classification"
                )

                st.info(
                    """
                    The machine learning model classified the
                    provided patient measurements as belonging to
                    the diabetic-risk class.

                    This result is a model prediction and is **not
                    a medical diagnosis**. A qualified healthcare
                    professional should interpret the patient's
                    clinical information and perform appropriate
                    diagnostic testing.
                    """
                )


            # ====================================================
            # CLASS 0
            # ====================================================

            elif predicted_class == 0:

                st.success(
                    "### 🟢 Lower-Risk Classification"
                )

                st.info(
                    """
                    The machine learning model classified the
                    provided patient measurements as belonging to
                    the non-diabetic-risk class.

                    This result is a model prediction and does
                    not rule out diabetes or other medical
                    conditions. Clinical assessment should still
                    be performed by a qualified healthcare
                    professional when appropriate.
                    """
                )


            # ====================================================
            # UNEXPECTED CLASS
            # ====================================================

            else:

                st.warning(
                    f"""
                    ⚠️ The model returned an unexpected
                    classification value: {predicted_class}
                    """
                )


            # ====================================================
            # OPTIONAL PROBABILITY
            # ====================================================

            if hasattr(model, "predict_proba"):

                try:

                    probabilities = model.predict_proba(
                        input_data
                    )[0]

                    st.markdown("### 📈 Model Output")

                    if len(probabilities) >= 2:

                        non_diabetic_probability = (
                            probabilities[0] * 100
                        )

                        diabetic_probability = (
                            probabilities[1] * 100
                        )

                        prob_col1, prob_col2 = st.columns(2)


                        # ----------------------------------------
                        # NON-DIABETIC PROBABILITY
                        # ----------------------------------------

                        with prob_col1:

                            st.metric(
                                "Non-Diabetic Class",
                                f"{non_diabetic_probability:.1f}%"
                            )


                        # ----------------------------------------
                        # DIABETIC PROBABILITY
                        # ----------------------------------------

                        with prob_col2:

                            st.metric(
                                "Diabetic-Risk Class",
                                f"{diabetic_probability:.1f}%"
                            )


                        st.caption(
                            """
                            These percentages represent the
                            model's estimated class probabilities
                            and should not be interpreted as
                            diagnostic certainty.
                            """
                        )


                except Exception as probability_error:

                    # Probability calculation is optional.
                    # The classification result is still displayed.

                    st.caption(
                        "Model probabilities are unavailable."
                    )


        # ========================================================
        # PREDICTION ERROR
        # ========================================================

        except Exception as e:

            st.error(
                "❌ An error occurred while making the prediction."
            )

            st.code(
                f"{type(e).__name__}: {str(e)}",
                language="text"
            )

            st.warning(
                """
                Please verify that the input column names and
                data format match the data used when training
                the machine learning model.
                """
            )


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

        <p>
            <strong>Disclaimer:</strong>
            This application is intended for educational and
            demonstration purposes only. It is not a substitute
            for professional medical advice, diagnosis, or treatment.
        </p>

        <p>
            The machine learning output should not be interpreted
            as a definitive medical diagnosis.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)
