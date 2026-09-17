import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Health Dashboard | Diabetes Risk",
    page_icon="🧑‍⚕️",
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
# CUSTOM CSS (Inspired by the clean, light mobile UI reference)
# ============================================================

st.markdown(
    """
    <style>
    /* Light Theme Backgrounds */
    .stApp {
        background-color: #F0F4F8; /* Soft blue-gray background */
        color: #1E293B; /* Dark slate text for high legibility */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    [data-testid="stSidebar"] * {
        color: #334155;
    }
    
    /* Headers and Text */
    h1, h2, h3, h4 {
        color: #0F172A;
        font-weight: 700 !important;
    }
    
    p, span {
        color: #475569;
    }

    /* Card/Form Container Styling */
    [data-testid="stForm"] {
        background-color: #FFFFFF;
        border-radius: 28px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
        border: none;
        margin-bottom: 24px;
    }

    /* Form Elements and Inputs */
    div[data-baseweb="input"] > div, div[data-baseweb="number-input"] > div {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        color: #0F172A !important;
        transition: all 0.2s ease;
    }
    
    div[data-baseweb="input"] > div:focus-within, div[data-baseweb="number-input"] > div:focus-within {
        border-color: #93C5FD !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(147, 197, 253, 0.2) !important;
    }
    
    /* Number input text color fix */
    input[type="number"] {
        color: #0F172A !important;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #1E293B; /* Dark, pill-shaped button like the nav bar in reference */
        color: #FFFFFF;
        border: none;
        border-radius: 30px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.2);
    }
    
    div.stButton > button:first-child:hover {
        background-color: #0F172A;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(30, 41, 59, 0.3);
        color: #FFFFFF;
    }

    /* Custom Dashboard Header */
    .dash-header {
        display: flex;
        align-items: center;
        margin-bottom: 30px;
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 28px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
    
    .dash-avatar {
        width: 56px;
        height: 56px;
        background-color: #E0F2FE;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-right: 16px;
    }

    /* Result Cards */
    .result-card {
        background-color: #FFFFFF;
        border-radius: 28px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
        text-align: center;
    }
    
    .status-circle-danger {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background-color: #FFF1F2;
        border: 4px solid #F43F5E; /* Matches the vibrant orange/red from reference */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px auto;
    }
    
    .status-circle-safe {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background-color: #F0FDF4;
        border: 4px solid #4ADE80; /* Muted green */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px auto;
    }

    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #94A3B8;
        padding: 20px;
        margin-top: 40px;
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
    <div class="dash-header">
        <div class="dash-avatar">
            🩺
        </div>
        <div>
            <h1 style="margin: 0; font-size: 24px;">Clinical Dashboard</h1>
            <p style="margin: 0; color: #64748B; font-size: 14px;">Diabetes Risk Assessment Tool</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    # Use HTML to make sidebar titles look cleaner
    st.markdown("<h3 style='margin-bottom: 5px;'>ℹ️ Tool Overview</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 0.9rem; margin-bottom: 20px;">
        This application classifies a patient's diabetes risk using a pre-trained machine learning model based on clinical metrics.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h3 style='margin-bottom: 5px;'>📊 Required Metrics</h3>", unsafe_allow_html=True)
    
    # Styled list mimicking the rounded UI
    metrics = ["Pregnancies", "Glucose Level", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Pedigree Function", "Age"]
    for m in metrics:
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; padding: 10px 15px; border-radius: 12px; margin-bottom: 8px; border: 1px solid #E2E8F0; font-size: 0.9rem; font-weight: 500;">
                {m}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br><hr style='border-color: #E2E8F0;'>", unsafe_allow_html=True)
    st.caption("For educational and demonstration purposes only.")

# ============================================================
# PATIENT INPUT FORM
# ============================================================

# The st.form will automatically pick up the CSS styling to look like a large white card
with st.form("patient_data_form"):
    st.markdown("<h3 style='margin-top: 0;'>Patient Metrics</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem; margin-bottom: 20px;'>Enter the data below to run the assessment.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    # COLUMN 1: Personal
    with col1:
        st.markdown("<div style='display:flex; align-items:center; gap:8px;'><div style='background:#E0F2FE; width:24px; height:24px; border-radius:50%; display:flex; justify-content:center; align-items:center;'>👤</div> <strong>Personal</strong></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, step=1)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

    # COLUMN 2: Blood Tests
    with col2:
        st.markdown("<div style='display:flex; align-items:center; gap:8px;'><div style='background:#FEE2E2; width:24px; height:24px; border-radius:50%; display:flex; justify-content:center; align-items:center;'>🩸</div> <strong>Blood Tests</strong></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, format="%.1f")
        insulin = st.number_input("Insulin (μU/ml)", min_value=0.0, max_value=900.0, value=80.0, format="%.1f")

    # COLUMN 3: Other Metrics
    with col3:
        st.markdown("<div style='display:flex; align-items:center; gap:8px;'><div style='background:#DCFCE7; width:24px; height:24px; border-radius:50%; display:flex; justify-content:center; align-items:center;'>🩺</div> <strong>Vitals & Gen</strong></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        bloodpressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0, format="%.1f")
        skinthickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, format="%.1f")
        diabPedFun = st.number_input("Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Run Analysis", use_container_width=True)


# ============================================================
# PREDICTION & RESULTS
# ============================================================

if submit_button:
    # Prepare Input Data
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

    # Load Model
    with st.spinner("Analyzing patient metrics..."):
        model, model_error = load_model()

    if model is None:
        st.error("⚠️ The trained model could not be loaded.")
        st.code(model_error, language="text")
        st.stop()

    # Make Prediction
    try:
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        
        # Determine styling based on class
        if predicted_class == 1:
            status_class = "status-circle-danger"
            status_text_color = "#F43F5E"
            status_icon = "⚠️"
            status_title = "High Risk"
            status_desc = "The model classifies these metrics as elevated risk for diabetes. Please consult a healthcare provider for clinical diagnosis."
        else:
            status_class = "status-circle-safe"
            status_text_color = "#22C55E"
            status_icon = "✓"
            status_title = "Low Risk"
            status_desc = "The model classifies these metrics as lower risk. Continue maintaining a healthy lifestyle. This does not rule out medical conditions."
            
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Top section of result card
        st.markdown("<h3 style='margin-bottom: 30px;'>Assessment Complete</h3>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            # Big Circular Indicator matching the dashboard style
            st.markdown(
                f"""
                <div class="{status_class}">
                    <span style="font-size: 40px;">{status_icon}</span>
                    <span style="color: {status_text_color}; font-weight: bold; font-size: 18px; margin-top: 5px;">{status_title}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with res_col2:
            st.markdown(
                f"""
                <div style="text-align: left; padding-top: 10px;">
                    <p style="font-size: 16px; line-height: 1.6;">{status_desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Probabilities Visualization
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                if len(probabilities) >= 2:
                    p_low = probabilities[0] * 100
                    p_high = probabilities[1] * 100
                    
                    st.markdown("<div style='text-align: left; margin-top: 20px;'><strong style='color: #0F172A;'>Confidence Score</strong></div>", unsafe_allow_html=True)
                    
                    # Pill-shaped light theme progress bars
                    st.markdown(
                        f"""
                        <div style="margin-top: 10px; text-align: left;">
                            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px;">
                                <span>Low Risk</span>
                                <strong>{p_low:.1f}%</strong>
                            </div>
                            <div style="width: 100%; background-color: #F1F5F9; border-radius: 10px; height: 12px; overflow: hidden;">
                                <div style="width: {p_low}%; height: 100%; background-color: #4ADE80; border-radius: 10px;"></div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 15px; text-align: left;">
                            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px;">
                                <span>High Risk</span>
                                <strong>{p_high:.1f}%</strong>
                            </div>
                            <div style="width: 100%; background-color: #F1F5F9; border-radius: 10px; height: 12px; overflow: hidden;">
                                <div style="width: {p_high}%; height: 100%; background-color: #F43F5E; border-radius: 10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )
        
        st.markdown('</div>', unsafe_allow_html=True) # End Results Card

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.code(f"{type(e).__name__}: {str(e)}", language="text")

# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <p><strong>Disclaimer:</strong> This application is a demonstration of machine learning capabilities and is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    </div>
    """,
    unsafe_allow_html=True
)
