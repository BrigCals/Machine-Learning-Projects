import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS (Inspired by reference image)
# ============================================================
st.markdown(
    """
    <style>
    /* Global Styles & Dark Theme Base */
    :root {
        --bg-color: #0f0c29; /* Dark purple/navy background */
        --panel-bg: #1e1b4b; /* Slightly lighter panel background */
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --neon-green: #34d399; /* From the image's green gradient */
        --neon-blue: #38bdf8;  /* From the image's blue gradient */
        --neon-yellow: #fde047; /* From the image's yellow gradient */
        --neon-pink: #f472b6;   /* From the background pink */
    }

    .stApp {
        background: linear-gradient(to bottom right, var(--bg-color), #302b63, #24243e);
        color: var(--text-primary);
    }

    /* Gradient Text Utility */
    .gradient-text {
        background: linear-gradient(90deg, var(--neon-yellow), var(--neon-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Main Title Area */
    .main-title-container {
        text-align: center;
        padding: 40px 20px;
        background: rgba(30, 27, 75, 0.6);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    .main-title-container h1 {
        font-size: 3rem;
        margin-bottom: 15px;
    }

    .main-title-container p {
        font-size: 1.2rem;
        color: var(--text-secondary);
        max-width: 600px;
        margin: 0 auto;
    }

    /* Section Headings */
    .section-title {
        color: var(--neon-blue);
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid rgba(56, 189, 248, 0.3);
        padding-bottom: 10px;
        display: inline-block;
    }

    /* Cards for Input Columns */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: var(--panel-bg);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Style Number Inputs */
    .stNumberInput > div > div > input {
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--neon-green) !important;
        box-shadow: 0 0 0 1px var(--neon-green) !important;
    }

    /* Submit Button Styling (Gradient) */
    .stButton > button {
        background: linear-gradient(90deg, #10b981, #3b82f6);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 15px 30px;
        font-size: 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        width: 100%;
        margin-top: 20px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
        background: linear-gradient(90deg, #059669, #2563eb);
    }

    /* Result Containers */
    .result-high {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.1), rgba(159, 18, 57, 0.4));
        border-left: 5px solid #f43f5e;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .result-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 95, 70, 0.4));
        border-left: 5px solid #10b981;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }

    /* Footer */
    .footer-container {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    """Load the trained machine learning model."""
    if not MODEL_PATH.exists():
        return None, f"Model file not found at: {MODEL_PATH}"
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {type(e).__name__} - {str(e)}"

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="main-title-container">
        <h1><span class="gradient-text">Diabetes Risk</span> Assessment</h1>
        <p>Advanced machine-learning analysis based on key clinical measurements.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<h2 style="color: #38bdf8;">🧬 About the Tool</h2>', unsafe_allow_html=True)
    st.write(
        "This application utilizes a predictive model to assess the likelihood of diabetes risk based on user-provided health metrics."
    )
    st.divider()
    
    st.markdown('<h3 style="color: #fde047;">Input Parameters</h3>', unsafe_allow_html=True)
    
    # Styled list in sidebar
    st.markdown("""
    <ul style="color: #94a3b8; line-height: 1.8;">
        <li><strong style="color:white;">Pregnancies:</strong> Number of times pregnant</li>
        <li><strong style="color:white;">Glucose:</strong> Plasma glucose concentration</li>
        <li><strong style="color:white;">Blood Pressure:</strong> Diastolic pressure (mmHg)</li>
        <li><strong style="color:white;">Skin Thickness:</strong> Triceps skinfold (mm)</li>
        <li><strong style="color:white;">Insulin:</strong> 2-Hour serum insulin (mu U/ml)</li>
        <li><strong style="color:white;">BMI:</strong> Body mass index</li>
        <li><strong style="color:white;">DPF:</strong> Diabetes pedigree function</li>
        <li><strong style="color:white;">Age:</strong> Years</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.info("⚠️ For educational and demonstration purposes only. Not a medical diagnosis.")

# ============================================================
# PATIENT INPUT
# ============================================================
st.markdown('<div class="section-title">📋 Enter Health Metrics</div>', unsafe_allow_html=True)

with st.form("patient_data_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # COLUMN 1: Personal Info
    with col1:
        st.markdown('<h3 style="color: #34d399; margin-bottom: 20px;">👤 Profile</h3>', unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, step=1)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

    # COLUMN 2: Blood Tests
    with col2:
        st.markdown('<h3 style="color: #f472b6; margin-bottom: 20px;">🩸 Vitals</h3>', unsafe_allow_html=True)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, format="%.1f")
        insulin = st.number_input("Insulin (μU/ml)", min_value=0.0, max_value=900.0, value=80.0, format="%.1f")
        bloodpressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0, format="%.1f")

    # COLUMN 3: Other Metrics
    with col3:
        st.markdown('<h3 style="color: #fde047; margin-bottom: 20px;">📊 Specifics</h3>', unsafe_allow_html=True)
        skinthickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, format="%.1f")
        diabPedFun = st.number_input("Pedigree Function (DPF)", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")

    # SUBMIT BUTTON
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🚀 Analyze Risk Profile", use_container_width=True)

# ============================================================
# PREDICTION & RESULTS
# ============================================================
if submit_button:
    # Create DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies], "Glucose": [glucose], "BloodPressure": [bloodpressure],
        "SkinThickness": [skinthickness], "Insulin": [insulin], "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabPedFun], "Age": [age]
    })

    st.markdown('<div class="section-title">📊 Input Summary</div>', unsafe_allow_html=True)
    
    # Styled dataframe container
    st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    # Load Model
    with st.spinner("Analyzing data with AI model..."):
        model, model_error = load_model()

    if model is None:
        st.error("⚠️ Failed to load the predictive model.")
        st.code(model_error, language="text")
        st.info("Ensure 'diabetes_prediction_model.joblib' is in the app directory.")
        st.stop()

    # Make Prediction
    try:
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        
        st.markdown('<div class="section-title">🎯 Assessment Results</div>', unsafe_allow_html=True)

        if predicted_class == 1:
            st.markdown(
                """
                <div class="result-high">
                    <h3 style="color: #f43f5e; margin: 0 0 10px 0;">🚨 Higher-Risk Classification Detected</h3>
                    <p style="margin: 0; color: #e2e8f0;">The model indicates patterns consistent with higher diabetes risk based on the provided metrics. Please consult a healthcare professional for a formal evaluation.</p>
                </div>
                """, unsafe_allow_html=True
            )
        elif predicted_class == 0:
            st.markdown(
                """
                <div class="result-low">
                    <h3 style="color: #10b981; margin: 0 0 10px 0;">✅ Lower-Risk Classification</h3>
                    <p style="margin: 0; color: #e2e8f0;">The model indicates lower risk based on these specific metrics. Continue maintaining healthy habits. Remember, this does not rule out underlying conditions.</p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.warning(f"Unexpected classification value: {predicted_class}")

        # Probabilities (If model supports it)
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(input_data)[0]
                
                st.markdown('<h4 style="color: #94a3b8; margin-top: 30px;">Model Confidence Breakdown</h4>', unsafe_allow_html=True)
                
                if len(probabilities) >= 2:
                    p_low = probabilities[0] * 100
                    p_high = probabilities[1] * 100
                    
                    # Custom metric layout
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="Probability of Lower Risk (Class 0)", value=f"{p_low:.1f}%")
                    with m_col2:
                        st.metric(label="Probability of Higher Risk (Class 1)", value=f"{p_high:.1f}%")
                        
            except Exception:
                pass # Fail silently for probabilities

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.code(str(e))

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-container">
        <p><strong style="color: var(--neon-pink);">Disclaimer:</strong> This application is intended for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
        <p style="opacity: 0.7;">The machine learning output should not be interpreted as a definitive medical diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)
