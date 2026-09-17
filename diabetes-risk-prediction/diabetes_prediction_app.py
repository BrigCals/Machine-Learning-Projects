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
# CUSTOM CSS (Dark Glowing Green Theme)
# ============================================================
st.markdown(
    """
    <style>
    /* GLOBAL APP */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(46, 213, 115, 0.18), transparent 40%),
                    radial-gradient(circle at 90% 90%, rgba(11, 163, 96, 0.15), transparent 40%),
                    linear-gradient(135deg, #06090a 0%, #0d1214 48%, #04080a 100%);
        background-attachment: fixed;
        color: #ffffff;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* MAIN GLASS CONTAINER */
    .block-container {
        max-width: 960px;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        padding: 42px 42px 30px 42px !important;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid rgba(46, 213, 115, 0.15);
        box-shadow: 0 25px 70px rgba(0,0,0,0.5), inset 0 1px 0 rgba(46, 213, 115, 0.1);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 12, 9, 0.97), rgba(4, 8, 6, 0.95)) !important;
        border-right: 1px solid rgba(46, 213, 115, 0.15);
    }
    .sidebar-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    .brand-icon { width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, rgba(46, 213, 115, 0.25), rgba(11, 163, 96, 0.2)); border: 1px solid rgba(46, 213, 115, 0.4); font-size: 25px; box-shadow: 0 0 20px rgba(46, 213, 115, 0.3); }
    .brand-title { font-size: 20px; font-weight: 800; color: #ffffff; text-shadow: 0 0 10px rgba(46, 213, 115, 0.3); }
    .brand-subtitle { font-size: 12px; color: #9da3b8; margin-top: 2px; }
    
    .side-section-title { color: #f5f7ff; font-size: 18px; font-weight: 800; margin-bottom: 18px; margin-top: 15px;}
    .side-step { display: flex; gap: 12px; margin-bottom: 18px; color: #ffffff; }
    .side-step > span { width: 31px; height: 31px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; border-radius: 10px; background: rgba(46, 213, 115, 0.1); border: 1px solid rgba(46, 213, 115, 0.3); color: #2ed573; font-size: 11px; font-weight: 800; box-shadow: 0 0 10px rgba(46, 213, 115, 0.2); }
    .side-step small { color: #8e95a5; line-height: 1.5; font-size: 12px;}
    
    .model-info { padding: 15px; border-radius: 15px; background: rgba(255,255,255,0.02); border: 1px solid rgba(46, 213, 115, 0.15); margin-top: 20px;}
    .model-info-title { color: #ffffff; font-weight: 700; margin-bottom: 4px; }
    .model-info-text { color: #9fa6bd; font-size: 13px; margin-bottom: 9px; }
    .model-status { color: #8fe7b2; font-size: 12px; display: flex; align-items: center; gap: 7px; text-shadow: 0 0 8px rgba(46, 213, 115, 0.5); }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #2ed573; box-shadow: 0 0 10px rgba(46, 213, 115, 0.9); }

    /* HEADER */
    .health-badge { width: fit-content; margin: 0 auto 14px auto; padding: 6px 13px; border-radius: 999px; background: rgba(46, 213, 115, 0.1); border: 1px solid rgba(46, 213, 115, 0.3); color: #2ed573; font-size: 10px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; box-shadow: 0 0 15px rgba(46, 213, 115, 0.2); }
    .main-title { text-align: center; font-size: clamp(32px, 5vw, 46px); line-height: 1.1; font-weight: 850; letter-spacing: -1.5px; color: #ffffff !important; margin: 0; }
    .subtitle { text-align: center; color: #a7adbf !important; font-size: 15px; margin-top: 12px; margin-bottom: 35px; }

    /* FORM HEADERS */
    .section-title { display: flex; align-items: center; gap: 8px; color: #f4f5ff; font-size: 18px; font-weight: 800; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 10px;}
    .section-icon { color: #2ed573; font-size: 20px; text-shadow: 0 0 10px rgba(46, 213, 115, 0.5); }

    /* NUMBER INPUTS - FIXED VISIBILITY */
    div[data-testid="stNumberInput"] label p { 
        color: #a7adbf !important; 
        font-size: 13px !important; 
        font-weight: 600 !important; 
    }
    
    /* Force dark background on the input wrapper */
    div[data-testid="stNumberInput"] div[data-baseweb="input"] { 
        background-color: #0d1214 !important; 
        border: 1px solid rgba(46, 213, 115, 0.2) !important; 
        border-radius: 12px !important; 
        transition: all 0.3s ease;
    }
    
    /* Focus state for the input wrapper */
    div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
        border: 1px solid rgba(46, 213, 115, 0.8) !important;
        box-shadow: 0 0 15px rgba(46, 213, 115, 0.3) !important;
        background-color: #0d1214 !important;
    }
    
    /* Force white text on the input element itself */
    div[data-testid="stNumberInput"] input { 
        background-color: transparent !important; 
        color: #ffffff !important; 
        -webkit-text-fill-color: #ffffff !important; 
        font-size: 15px !important;
        font-weight: 600 !important;
    }

    /* Style the +/- buttons */
    div[data-testid="stNumberInput"] button {
        background-color: transparent !important;
        color: #2ed573 !important;
        border: none !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        color: #ffffff !important;
        background-color: rgba(46, 213, 115, 0.2) !important;
        border-radius: 8px !important;
    }

    /* BUTTONS - GREEN GLOWING GRADIENT */
    div[data-testid="stFormSubmitButton"] button { 
        min-height: 52px !important; 
        border-radius: 14px !important; 
        font-weight: 800 !important; 
        font-size: 15px !important; 
        transition: transform 0.2s ease, box-shadow 0.2s ease !important; 
        width: 100% !important;
        margin-top: 15px;
        background: linear-gradient(135deg, #2ed573, #0ba360) !important; 
        color: #000000 !important; 
        border: none !important; 
        box-shadow: 0 0 25px rgba(46, 213, 115, 0.5) !important; 
    }
    div[data-testid="stFormSubmitButton"] button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 0 35px rgba(46, 213, 115, 0.7) !important; 
    }

    /* RESULT CARD */
    .result-card { padding: 30px; border-radius: 20px; margin-top: 25px; margin-bottom: 20px; text-align: center; backdrop-filter: blur(15px); }
    .result-high { background: linear-gradient(135deg, rgba(255,82,82,0.1), rgba(255,82,82,0.02)); border: 1px solid rgba(255,82,82,0.4); box-shadow: 0 0 35px rgba(255,82,82,0.15); }
    .result-low { background: linear-gradient(135deg, rgba(46,213,115,0.1), rgba(46,213,115,0.02)); border: 1px solid rgba(46,213,115,0.4); box-shadow: 0 0 35px rgba(46,213,115,0.15); }
    .result-icon { font-size: 42px; margin-bottom: 10px; filter: drop-shadow(0 0 10px rgba(255,255,255,0.3)); }
    .result-label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #a7adbf; margin-bottom: 8px; }
    .result-value { font-size: 28px; font-weight: 850; margin-bottom: 10px;}
    .high-text { color: #ff5252; text-shadow: 0 0 15px rgba(255,82,82,0.4); }
    .low-text { color: #2ed573; text-shadow: 0 0 15px rgba(46,213,115,0.4); }
    .result-desc { color: #d7dbea; font-size: 14px; line-height: 1.5; margin: 0 auto; max-width: 90%;}

    /* CONFIDENCE BARS */
    .confidence-title { font-size: 16px; font-weight: 800; color: #f3f5fb; margin-top: 25px; margin-bottom: 15px; }
    .confidence-row { margin-bottom: 15px; background: rgba(0,0,0,0.2); padding: 12px 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); }
    .confidence-header { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; font-weight: 700; color: #d7dbea; }
    .confidence-track { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.05); overflow: hidden; }
    .confidence-fill-low { height: 100%; background: linear-gradient(90deg, #2ed573, #0ba360); border-radius: inherit; box-shadow: 0 0 10px rgba(46,213,115,0.5); }
    .confidence-fill-high { height: 100%; background: linear-gradient(90deg, #ff5252, #ff1744); border-radius: inherit; box-shadow: 0 0 10px rgba(255,82,82,0.5); }

    /* FOOTER */
    .app-footer { text-align: center; color: #737a90; font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); }
    
    /* DATAFRAME */
    [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(46, 213, 115, 0.2); }
    
    /* FIX FOR STREAMLIT METRICS IN DARK MODE */
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div[data-testid="stMetricLabel"] { color: #a7adbf !important; }
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">🧬</div>
            <div>
                <div class="brand-title">DiaPredict</div>
                <div class="brand-subtitle">Health Analytics AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="side-section-title">📊 Required Vitals</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="side-step">
            <span>01</span>
            <div>
                <b>Profile Data</b><br>
                <small>Age, Pregnancies, and Body Mass Index (BMI).</small>
            </div>
        </div>

        <div class="side-step">
            <span>02</span>
            <div>
                <b>Blood Work</b><br>
                <small>Glucose and Insulin levels from testing.</small>
            </div>
        </div>

        <div class="side-step">
            <span>03</span>
            <div>
                <b>Specifics</b><br>
                <small>Blood pressure, skin thickness, and genetics.</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="model-info">
            <div class="model-info-title">🧠 AI Engine</div>
            <div class="model-info-text">Predictive Risk Classifier</div>
            <div class="model-status">
                <span class="status-dot"></span> System Ready
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="health-badge">✦ MACHINE LEARNING HEALTH DIAGNOSTICS</div>
    <div class="main-title">Diabetes Risk Predictor</div>
    <div class="subtitle">Enter patient metrics to generate an AI-driven risk assessment.</div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PATIENT INPUT
# ============================================================
with st.form("patient_data_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    # COLUMN 1: Personal Info
    with col1:
        st.markdown('<div class="section-title"><span class="section-icon">👤</span> Profile</div>', unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, step=1)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

    # COLUMN 2: Blood Tests
    with col2:
        st.markdown('<div class="section-title"><span class="section-icon">🩸</span> Vitals</div>', unsafe_allow_html=True)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, format="%.1f")
        insulin = st.number_input("Insulin (μU/ml)", min_value=0.0, max_value=900.0, value=80.0, format="%.1f")
        bloodpressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0, format="%.1f")

    # COLUMN 3: Other Metrics
    with col3:
        st.markdown('<div class="section-title"><span class="section-icon">🧬</span> Specifics</div>', unsafe_allow_html=True)
        skinthickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, format="%.1f")
        diabPedFun = st.number_input("Pedigree Function (DPF)", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")

    # SUBMIT BUTTON
    submit_button = st.form_submit_button(label="🔍 Generate Risk Assessment", use_container_width=True)

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

    st.markdown('<br><div class="section-title"><span class="section-icon">📋</span> Input Summary</div>', unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    # Load Model
    with st.spinner("Analyzing physiological data..."):
        model, model_error = load_model()

    if model is None:
        st.error("⚠️ Failed to load the predictive model.")
        st.code(model_error, language="text")
        st.stop()

    # Make Prediction
    try:
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        
        if predicted_class == 1:
            st.markdown(
                """
                <div class="result-card result-high">
                    <div class="result-icon">🛑</div>
                    <div class="result-label">Assessment Result</div>
                    <div class="result-value high-text">ELEVATED RISK DETECTED</div>
                    <p class="result-desc">The predictive model identified patterns associated with higher diabetes risk based on the provided metrics. Please consult a healthcare professional.</p>
                </div>
                """, unsafe_allow_html=True
            )
        elif predicted_class == 0:
            st.markdown(
                """
                <div class="result-card result-low">
                    <div class="result-icon">✅</div>
                    <div class="result-label">Assessment Result</div>
                    <div class="result-value low-text">LOWER RISK IDENTIFIED</div>
                    <p class="result-desc">The model indicates a lower risk profile based on these specific metrics. Continue maintaining healthy habits and routine checkups.</p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.warning(f"Unexpected classification value: {predicted_class}")

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(input_data)[0]
                
                if len(probabilities) >= 2:
                    p_low = probabilities[0] * 100
                    p_high = probabilities[1] * 100
                    
                    st.markdown('<div class="confidence-title">Model Confidence Breakdown</div>', unsafe_allow_html=True)
                    
                    st.markdown(
                        f"""
                        <div class="confidence-row">
                            <div class="confidence-header">
                                <span>✅ Lower Risk (Class 0)</span>
                                <span>{p_low:.1f}%</span>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill-low" style="width:{p_low:.1f}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="confidence-row">
                            <div class="confidence-header">
                                <span>🛑 Elevated Risk (Class 1)</span>
                                <span>{p_high:.1f}%</span>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill-high" style="width:{p_high:.1f}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                        
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
    <div class="app-footer">
        <strong>Disclaimer:</strong> Educational and demonstration purposes only. Not a medical diagnosis.<br>
        DiaPredict AI &nbsp;•&nbsp; Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
