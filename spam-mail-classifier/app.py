import streamlit as st
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL FILE PATHS
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(CURRENT_DIR, "spam_classifier_model.joblib")
VECTORIZER_PATH = os.path.join(CURRENT_DIR, "vectorizer.joblib")

# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(
            """
            ❌ Unable to load the required model files.
            Make sure these files are located in the same folder as `app.py`:
            • spam_classifier_model.joblib
            • vectorizer.joblib
            """
        )
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("❌ An unexpected error occurred while loading the model.")
        st.exception(e)
        st.stop()

model, vectorizer = load_models()

# ============================================================
# SESSION STATE
# ============================================================
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
    <div class="brand-icon">🛡️</div>
    <div>
        <div class="brand-title">SpamShield</div>
        <div class="brand-subtitle">AI Email Security</div>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
<div class="side-section-title">ℹ️ How It Works</div>

<div class="side-step">
    <span>01</span>
    <div>
        <b>Paste email</b><br>
        <small>Enter the message you want to analyze.</small>
    </div>
</div>

<div class="side-step">
    <span>02</span>
    <div>
        <b>Run analysis</b><br>
        <small>The machine learning model processes the text.</small>
    </div>
</div>

<div class="side-step">
    <span>03</span>
    <div>
        <b>Review result</b><br>
        <small>See the predicted class and confidence.</small>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
<div class="model-info">
    <div class="model-info-title">🤖 Model</div>
    <div class="model-info-text">Naive Bayes text classifier</div>
    <div class="model-status">
        <span class="status-dot"></span> Model loaded
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# CSS (SINGLE DARK THEME)
# ============================================================
custom_css = """
<style>
    /* GLOBAL APP */
    .stApp {
        background: radial-gradient(circle at 5% 95%, rgba(123, 62, 255, 0.28), transparent 38%),
                    radial-gradient(circle at 95% 5%, rgba(0, 153, 255, 0.25), transparent 38%),
                    linear-gradient(135deg, #080514 0%, #100a2b 48%, #071a38 100%);
        background-attachment: fixed;
        color: #ffffff;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* MAIN GLASS CONTAINER */
    .block-container {
        max-width: 920px;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        padding: 42px 42px 30px 42px !important;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025));
        border: 1px solid rgba(255,255,255,0.13);
        box-shadow: 0 25px 70px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.10);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(12, 7, 32, 0.97), rgba(8, 5, 24, 0.95)) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    .brand-icon { width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, rgba(106,72,255,0.35), rgba(0,170,255,0.30)); border: 1px solid rgba(255,255,255,0.13); font-size: 25px; box-shadow: 0 0 25px rgba(90,80,255,0.18); }
    .brand-title { font-size: 20px; font-weight: 800; color: #ffffff; }
    .brand-subtitle { font-size: 12px; color: #9da3b8; margin-top: 2px; }
    .side-section-title { color: #f5f7ff; font-size: 18px; font-weight: 800; margin-bottom: 18px; }
    .side-step { display: flex; gap: 12px; margin-bottom: 18px; color: #ffffff; }
    .side-step > span { width: 31px; height: 31px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; border-radius: 10px; background: rgba(108,83,255,0.16); border: 1px solid rgba(127,115,255,0.25); color: #b8b0ff; font-size: 11px; font-weight: 800; }
    .side-step small { color: #949bb1; line-height: 1.5; }
    .model-info { padding: 15px; border-radius: 15px; background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); }
    .model-info-title { color: #ffffff; font-weight: 700; margin-bottom: 4px; }
    .model-info-text { color: #9fa6bd; font-size: 13px; margin-bottom: 9px; }
    .model-status { color: #8fe7b2; font-size: 12px; display: flex; align-items: center; gap: 7px; }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #52e28c; box-shadow: 0 0 10px rgba(82,226,140,0.8); }

    /* HEADER */
    .security-badge { width: fit-content; margin: 0 auto 14px auto; padding: 6px 13px; border-radius: 999px; background: rgba(99,80,255,0.12); border: 1px solid rgba(125,110,255,0.28); color: #b9b0ff; font-size: 10px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }
    .main-title { text-align: center; font-size: clamp(36px, 5vw, 50px); line-height: 1.05; font-weight: 850; letter-spacing: -2px; color: #ffffff !important; margin: 0; }
    .subtitle { text-align: center; color: #a7adbf !important; font-size: 15px; margin-top: 12px; margin-bottom: 28px; }

    /* QUICK TEST SELECTBOX */
    div[data-testid="stSelectbox"] label { color: #dedff0 !important; font-size: 12px !important; font-weight: 700 !important; }
    div[data-baseweb="select"] > div { background: rgba(15,17,38,0.85) !important; border: 1px solid rgba(255,255,255,0.13) !important; border-radius: 13px !important; color: #f5f7ff !important; }
    div[data-baseweb="select"] span { color: #e8e9f3 !important; }

    /* EMAIL INPUT HEADER */
    .email-input-header { display: flex; justify-content: space-between; align-items: center; margin-top: 16px; margin-bottom: 5px; }
    .email-input-title { display: flex; align-items: center; gap: 8px; color: #f4f5ff; font-size: 15px; font-weight: 800; }
    .email-icon { color: #8e7cff; font-size: 17px; }
    .email-input-status { padding: 4px 9px; border-radius: 999px; background: rgba(126,111,255,0.10); border: 1px solid rgba(126,111,255,0.20); color: #9185e8; font-size: 9px; font-weight: 800; letter-spacing: 0.8px; }
    .email-input-description { color: #81889e; font-size: 12px; margin-bottom: 10px; }

    /* EMAIL TEXTAREA - FIXED FONT COLOR */
    div[data-testid="stTextArea"] { width: 100%; }
    div[data-testid="stTextArea"] div[data-baseweb="textarea"] { 
        background: rgba(15, 17, 38, 0.6) !important; 
        border: 1px solid rgba(255,255,255,0.2) !important; 
        border-radius: 12px !important; 
    }
    div[data-testid="stTextArea"] textarea { 
        background: transparent !important; 
        color: #ffffff !important; /* FIXED: explicitly set text color to white */
        -webkit-text-fill-color: #ffffff !important; /* FIXED: for webkit browsers */
        font-size: 16px !important;
        padding: 15px !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder { color: #8c93a8 !important; opacity: 1 !important; }
    
    .character-count { text-align: right; color: #70778b; font-size: 11px; margin-top: 5px; margin-bottom: 10px; }

    /* BUTTONS */
    .stButton > button { min-height: 49px !important; border-radius: 14px !important; font-weight: 800 !important; font-size: 14px !important; transition: transform 0.2s ease, box-shadow 0.2s ease !important; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,0,0,0.25) !important; }
    div[data-testid="stButton"] button[kind="primary"] { background: linear-gradient(135deg, #7567ff, #4b8cff) !important; color: white !important; border: none !important; box-shadow: 0 8px 24px rgba(89,94,255,0.30) !important; }

    /* RESULT CARD */
    .result-card { padding: 23px; border-radius: 19px; margin-top: 22px; margin-bottom: 20px; text-align: center; backdrop-filter: blur(15px); }
    .result-spam { background: linear-gradient(135deg, rgba(255,55,82,0.13), rgba(255,30,65,0.045)); border: 1px solid rgba(255,86,108,0.35); box-shadow: 0 0 35px rgba(255,45,72,0.08); }
    .result-safe { background: linear-gradient(135deg, rgba(49,222,139,0.12), rgba(39,180,115,0.04)); border: 1px solid rgba(66,220,142,0.30); box-shadow: 0 0 35px rgba(55,220,135,0.07); }
    .result-icon { font-size: 34px; margin-bottom: 5px; }
    .result-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #999fb3; margin-bottom: 5px; }
    .result-value { font-size: 30px; font-weight: 850; }
    .spam-text { color: #ff7285; }
    .safe-text { color: #67e29d; }

    /* CONFIDENCE */
    .confidence-title { font-size: 18px; font-weight: 800; color: #f3f5fb; margin-top: 25px; margin-bottom: 15px; }
    .confidence-row { margin-bottom: 15px; }
    .confidence-header { display: flex; justify-content: space-between; margin-bottom: 7px; font-size: 13px; font-weight: 700; color: #d7dbea; }
    .confidence-track { height: 9px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden; }
    .confidence-fill-spam { height: 100%; background: linear-gradient(90deg, #ff5570, #ff304c); border-radius: inherit; }
    .confidence-fill-safe { height: 100%; background: linear-gradient(90deg, #48dd96, #25ba78); border-radius: inherit; }

    /* FOOTER & MOBILE */
    .app-footer { text-align: center; color: #737a90; font-size: 12px; margin-top: 24px; }
    footer { visibility: hidden; }
    @media (max-width: 700px) {
        .block-container { padding: 28px 18px 22px 18px !important; margin-top: 0.5rem; border-radius: 20px; }
        .main-title { font-size: 36px; }
        .subtitle { font-size: 13px; }
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown(
    """
<div class="security-badge">✦ MACHINE LEARNING EMAIL SECURITY</div>
<div class="main-title">SpamShield AI 🛡️</div>
<div class="subtitle">Analyze an email and determine whether it is likely to be spam.</div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# QUICK TEST
# ============================================================
sample_choice = st.selectbox(
    "Quick Test",
    ["None — enter my own email", "Example: Promotional Spam", "Example: Normal Email"],
    label_visibility="visible"
)

if sample_choice == "Example: Promotional Spam":
    st.session_state.email_text = (
        "Congratulations!\n\n"
        "You have been selected to receive a $1,000 gift card. "
        "Click the link below immediately to claim your reward. "
        "This limited-time offer expires today!\n\n"
        "Claim your reward now!"
    )
elif sample_choice == "Example: Normal Email":
    st.session_state.email_text = (
        "Hi John,\n\n"
        "Just a reminder that our meeting is scheduled for tomorrow "
        "at 10:00 AM. Please bring the updated report.\n\n"
        "Thanks!"
    )

st.markdown(
    """
<div class="email-input-header">
    <div class="email-input-title">
        <span class="email-icon">✉</span> Email Content
    </div>
    <div class="email-input-status">TEXT ANALYSIS</div>
</div>
<div class="email-input-description">Paste the complete email message below for analysis.</div>
    """,
    unsafe_allow_html=True
)

email_text = st.text_area(
    "Email content",
    value=st.session_state.email_text,
    placeholder="Example:\n\nDear customer,\n\nYou have won a $1,000 gift card...",
    height=220,
    label_visibility="collapsed"
)

st.session_state.email_text = email_text
character_count = len(email_text)

st.markdown(
    f"""
<div class="character-count">{character_count:,} characters</div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ACTION BUTTONS & CLASSIFICATION
# ============================================================
col1, col2 = st.columns([3, 1], gap="small")

with col1:
    classify_clicked = st.button("🔍  Analyze Email", use_container_width=True, type="primary")

with col2:
    clear_clicked = st.button("✕  Clear", use_container_width=True)

if clear_clicked:
    st.session_state.email_text = ""
    st.rerun()

if classify_clicked:
    if not email_text.strip():
        st.warning("⚠️ Please enter an email message before running the analysis.")
    else:
        with st.spinner("Analyzing email..."):
            try:
                email_vector = vectorizer.transform([email_text])
                prediction = model.predict(email_vector)[0]
                probability = model.predict_proba(email_vector)[0]
                classes = list(model.classes_)

                if 1 not in classes or 0 not in classes:
                    st.error("The loaded model does not contain the expected classes 0 and 1.\n\nExpected:\n• 0 = Not Spam\n• 1 = Spam")
                    st.stop()

                spam_index = classes.index(1)
                not_spam_index = classes.index(0)

                prob_spam = probability[spam_index] * 100
                prob_not_spam = probability[not_spam_index] * 100
                is_spam = prediction == 1

            except Exception as e:
                st.error("❌ An error occurred while analyzing the email.")
                st.exception(e)
                st.stop()

        if is_spam:
            st.markdown(
                """
<div class="result-card result-spam">
    <div class="result-icon">🚨</div>
    <div class="result-label">Classification Result</div>
    <div class="result-value spam-text">SPAM DETECTED</div>
</div>
                """,
                unsafe_allow_html=True
            )
            st.warning("This email contains patterns commonly associated with spam messages. Avoid clicking links or sharing personal information unless the sender is verified.")
        else:
            st.markdown(
                """
<div class="result-card result-safe">
    <div class="result-icon">✅</div>
    <div class="result-label">Classification Result</div>
    <div class="result-value safe-text">NOT SPAM</div>
</div>
                """,
                unsafe_allow_html=True
            )
            st.success("The model does not identify this message as spam based on the patterns it learned.")

        st.markdown(
            """
<div class="confidence-title">Confidence Scores</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="confidence-row">
    <div class="confidence-header">
        <span>✅ Not Spam</span>
        <span>{prob_not_spam:.1f}%</span>
    </div>
    <div class="confidence-track">
        <div class="confidence-fill-safe" style="width:{prob_not_spam:.1f}%;"></div>
    </div>
</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="confidence-row">
    <div class="confidence-header">
        <span>🚨 Spam</span>
        <span>{prob_spam:.1f}%</span>
    </div>
    <div class="confidence-track">
        <div class="confidence-fill-spam" style="width:{prob_spam:.1f}%;"></div>
    </div>
</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
<div style="text-align:center; color:#747b90; font-size:11px; margin-top:22px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.07);">
    ⚠️ Classification is a machine learning prediction, not a guarantee of email safety.
</div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<div class="app-footer">
    🛡️ SpamShield AI &nbsp;•&nbsp; Powered by Machine Learning &nbsp;•&nbsp; Streamlit
</div>
    """,
    unsafe_allow_html=True
)
