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
# CUSTOM CSS (THE NEW DESIGN)
# ============================================================

st.markdown("""
<style>
    /* Import a modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Background & Font */
    .stApp {
        background: linear-gradient(135deg, #0a0b10 0%, #1a1b2e 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #2dd4bf !important;
    }

    /* Main Title Gradient */
    .main-title h1 {
        background: linear-gradient(90deg, #2dd4bf 0%, #facc15 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .main-title p {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Section Titles */
    .section-title {
        color: #2dd4bf;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #facc15;
        padding-left: 10px;
    }

    /* Input Fields (Dark Glassmorphism) */
    div[data-testid="stTextInput"] input, 
    div[data-testid="stNumberInput"] input {
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #ffffff !important;
        border: 1px solid rgba(45, 212, 191, 0.3) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    div[data-testid="stTextInput"] input:focus, 
    div[data-testid="stNumberInput"] input:focus {
        border: 1px solid #2dd4bf !important;
        box-shadow: 0 0 10px rgba(45, 212, 191, 0.5) !important;
    }

    /* Labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 400 !important;
    }

    /* Form Submit Button (Gradient Pill) */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #14b8a6 0%, #facc15 100%) !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 12px 30px !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.4) !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.6) !important;
    }

    /* Custom Result Cards */
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }

    .result-high-risk {
        border-left: 6px solid #ef4444;
    }
    
    .result-low-risk {
        border-left: 6px solid #2dd4bf;
    }

    .result-card h3 {
        margin-top: 0;
        font-weight: 600;
    }

    .result-high-risk h3 { color: #ef4444; }
    .result-low-risk h3 { color: #2dd4bf; }

    /* Metric Cards (Inspired by the 82% circle in the image) */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="stMetricValue"] {
        color: #facc15 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    /* Dataframe Styling */
    div[data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
        padding: 30px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "diabetes_prediction_model.joblib"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
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
# HEADER
# ============================================================

st.markdown("""
    <div class="main-title">
        <h1>🩺 Diabetes Risk Assessment</h1>
        <p>Enter the patient's health measurements below for AI-driven risk classification.</p>
    </div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🩺 About")
    st.write("""
    This application uses a trained machine learning model
    to classify a patient's diabetes risk based on selected
    clinical measurements.
    """)
    st.divider()
    st.markdown("### 📋 Input Variables")
    st.markdown("""
    - Pregnancies
    - Glucose
    - Blood Pressure
    - Skin Thickness
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Age
    """)
    st.divider()
    st.caption("For educational and demonstration purposes only.")


# ============================================================
# PATIENT INPUT
# ============================================================

st.markdown('<h2 class="section-title">📋 Patient Health Data</h2>', unsafe_allow_html=True)

with st.form("patient_data_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Personal Info")
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, step=1)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

    with col2:
        st.markdown("#### 🩸 Blood Tests")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, format="%.1f")
        insulin = st.number_input("Insulin (μU/ml)", min_value=0.0, max_value=900.0, value=80.0, format="%.1f")

    with col3:
        st.markdown("#### 📊 Other Metrics")
        bloodpressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0, format="%.1f")
        skinthickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, format="%.1f")
        diabPedFun = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🔍 Analyze Risk", use_container_width=True)


# ============================================================
# PREDICTION
# ============================================================

if submit_button:
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

    st.markdown("---")
    st.markdown('<h2 class="section-title">📊 Entered Patient Data</h2>', unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    with st.spinner("Loading trained model..."):
        model, model_error = load_model()

    if model is None:
        st.error("⚠️ The trained model could not be loaded.")
        st.code(model_error, language="text")
        st.stop()

    with st.spinner("Analyzing patient data..."):
        try:
            prediction = model.predict(input_data)
            predicted_class = int(prediction[0])

            st.markdown("---")
            st.markdown('<h2 class="section-title">📋 Assessment Result</h2>', unsafe_allow_html=True)

            # Custom HTML Result Cards
            if predicted_class == 1:
                st.markdown("""
                <div class="result-card result-high-risk">
                    <h3>🔴 Higher-Risk Classification</h3>
                    <p>The machine learning model classified the provided patient measurements as belonging to the diabetic-risk class.</p>
                    <p><em>This result is a model prediction and is <strong>not a medical diagnosis</strong>. A qualified healthcare professional should interpret the patient's clinical information.</em></p>
                </div>
                """, unsafe_allow_html=True)
            elif predicted_class == 0:
                st.markdown("""
                <div class="result-card result-low-risk">
                    <h3>🟢 Lower-Risk Classification</h3>
                    <p>The machine learning model classified the provided patient measurements as belonging to the non-diabetic-risk class.</p>
                    <p><em>This result is a model prediction and does not rule out diabetes or other medical conditions. Clinical assessment should still be performed.</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ The model returned an unexpected classification value: {predicted_class}")

            # Probability Metrics (Styled as glowing cards)
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(input_data)[0]
                    st.markdown("### 📈 Model Output")
                    if len(probabilities) >= 2:
                        non_diabetic_prob = probabilities[0] * 100
                        diabetic_prob = probabilities[1] * 100
                        
                        prob_col1, prob_col2 = st.columns(2)
                        with prob_col1:
                            st.metric("Non-Diabetic Class", f"{non_diabetic_prob:.1f}%")
                        with prob_col2:
                            st.metric("Diabetic-Risk Class", f"{diabetic_prob:.1f}%")
                            
                        st.caption("These percentages represent the model's estimated class probabilities and should not be interpreted as diagnostic certainty.")
                except Exception:
                    st.caption("Model probabilities are unavailable.")

        except Exception as e:
            st.error("❌ An error occurred while making the prediction.")
            st.code(f"{type(e).__name__}: {str(e)}", language="text")
            st.warning("Please verify that the input column names and data format match the data used when training the model.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <p><strong>Disclaimer:</strong> This application is intended for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p>The machine learning output should not be interpreted as a definitive medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
