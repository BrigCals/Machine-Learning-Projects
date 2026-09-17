import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration (Must be the first Streamlit command)
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",  # Changed to wide for better column usage
    initial_sidebar_state="expanded"
)


# -----------------------------
# Load trained model (with error handling for missing file)
# -----------------------------
@st.cache_resource  # Cache the model load so it doesn't reload every rerun
def load_model():
    try:
        return joblib.load("diabetes_prediction_model.joblib")
    except FileNotFoundError:
        return None


model = load_model()

# -----------------------------
# Header and Description
# -----------------------------
st.markdown("""
    <div style='text-align: center; padding: 20px; border-bottom: 2px solid #4CAF50; margin-bottom: 30px;'>
        <h1 style='color: #2E7D32;'>🩺 Diabetes Risk Assessment Tool</h1>
        <p style='font-size: 1.1em; color: #555;'>
            Enter your health metrics below to predict the likelihood of diabetes. 
            This tool uses a machine learning model based on clinical data.
        </p>
    </div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ **Model file not found!** Please ensure 'diabetes_prediction_model.joblib' is in the same directory.")
    st.stop()  # Stop execution if model is missing

# -----------------------------
# User Input Section (Using a Form and Columns)
# -----------------------------
with st.container():
    st.markdown("### 📋 Patient Health Data")

    # Create a form so the app doesn't rerun on every single input change
    with st.form("patient_data_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Info**")
            age = st.number_input(
                "Age (Years)",
                min_value=1, max_value=120, value=30, step=1,
                help="Age in years."
            )
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0, max_value=20, value=0, step=1,
                help="Number of times pregnant."
            )
            bmi = st.number_input(
                "BMI",
                min_value=0.0, max_value=70.0, value=25.0, format="%.1f",
                help="Body mass index (weight in kg/(height in m)^2)."
            )

        with col2:
            st.markdown("**Blood Tests**")
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0.0, max_value=300.0, value=120.0, format="%.1f",
                help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test."
            )
            insulin = st.number_input(
                "Insulin (μU/ml)",
                min_value=0.0, max_value=900.0, value=80.0, format="%.1f",
                help="2-Hour serum insulin."
            )

        with col3:
            st.markdown("**Other Metrics**")
            bloodpressure = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=0.0, max_value=200.0, value=70.0, format="%.1f",
                help="Diastolic blood pressure."
            )
            skinthickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0, max_value=100.0, value=20.0, format="%.1f",
                help="Triceps skin fold thickness."
            )
            diabPedFun = st.number_input(
                "Diabetes Pedigree",
                min_value=0.0, max_value=3.0, value=0.5, format="%.3f",
                help="Diabetes pedigree function (genetic risk score)."
            )

        # Submit button for the form
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        submit_button = st.form_submit_button(label="🔍 Analyze Risk", use_container_width=True)

# -----------------------------
# Prediction & Results Display
# -----------------------------
if submit_button:
    # 1. Create input DataFrame
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

    # 2. Add a spinner for visual feedback while "processing" (even if fast)
    with st.spinner("Analyzing patient data..."):
        # Predict
        try:
            prediction = model.predict(input_data)

            # (Optional) If your model supports predict_proba, get confidence score
            # probabilities = model.predict_proba(input_data)[0]
            # confidence = max(probabilities) * 100

            st.markdown("---")
            st.markdown("### 📊 Assessment Result")

            # 3. Display Result with enhanced styling
            if prediction[0] == 1:
                st.error("#### 🔴 High Risk: Diabetic Profile Detected")
                st.info(
                    "The model indicates a high probability of diabetes based on the provided metrics. "
                    "**Please consult with a healthcare professional for a formal diagnosis and advice.**"
                )
                # If you have probabilities, you could add:
                # st.metric(label="Confidence Score", value=f"{confidence:.1f}%")

            else:
                st.success("#### 🟢 Low Risk: Non-Diabetic Profile Detected")
                st.info(
                    "The model indicates a low probability of diabetes based on the provided metrics. "
                    "Continue maintaining a healthy lifestyle."
                )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check the input data format and model compatibility.")

# -----------------------------
# Footer / Disclaimer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.8em; color: gray;'>"
    "Disclaimer: This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment."
    "</p>",
    unsafe_allow_html=True
)