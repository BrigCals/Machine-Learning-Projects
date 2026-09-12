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
# MODEL LOADING
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "spam_classifier_model.joblib"
)

VECTORIZER_PATH = os.path.join(
    CURRENT_DIR,
    "vectorizer.joblib"
)


@st.cache_resource
def load_models():
    """Load the trained model and text vectorizer once."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        return model, vectorizer

    except FileNotFoundError as e:
        st.error(
            "Unable to load the required model files.\n\n"
            "Make sure these files are in the same folder as app.py:\n"
            "• spam_classifier_model.joblib\n"
            "• vectorizer.joblib"
        )
        st.exception(e)
        st.stop()

    except Exception as e:
        st.error("An unexpected error occurred while loading the model.")
        st.exception(e)
        st.stop()


model, vectorizer = load_models()


# ============================================================
# SESSION STATE
# ============================================================
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False


# ============================================================
# THEME
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

    theme = st.radio(
        "Appearance",
        [
            "🌙 Night Mode",
            "☀️ Day Mode"
        ],
        index=0
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="side-section-title">
            ℹ️ How It Works
        </div>

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
            <div class="model-info-text">
                Naive Bayes text classifier
            </div>
            <div class="model-status">
                <span class="status-dot"></span>
                Model loaded
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NIGHT MODE CSS
# ============================================================
if theme == "🌙 Night Mode":

    custom_css = """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 90%,
                rgba(123, 62, 255, 0.26),
                transparent 38%
            ),
            radial-gradient(
                circle at 90% 5%,
                rgba(0, 153, 255, 0.24),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #080514 0%,
                #100a2b 50%,
                #07152f 100%
            );

        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }


    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(12, 7, 32, 0.96),
                rgba(8, 5, 24, 0.94)
            ) !important;

        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                135deg,
                rgba(106, 72, 255, 0.35),
                rgba(0, 170, 255, 0.30)
            );

        border: 1px solid rgba(255,255,255,0.13);
        font-size: 25px;
        box-shadow: 0 0 25px rgba(90,80,255,0.18);
    }

    .brand-title {
        font-size: 20px;
        font-weight: 800;
        color: white;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #9da3b8;
        margin-top: 2px;
    }

    .side-section-title {
        color: #f5f7ff;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .side-step {
        display: flex;
        gap: 12px;
        margin-bottom: 18px;
        color: #ffffff;
    }

    .side-step > span {
        width: 31px;
        height: 31px;
        flex-shrink: 0;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 10px;

        background: rgba(108, 83, 255, 0.16);
        border: 1px solid rgba(127, 115, 255, 0.25);

        color: #b8b0ff;
        font-size: 11px;
        font-weight: 800;
    }

    .side-step small {
        color: #949bb1;
        line-height: 1.4;
    }

    .model-info {
        padding: 15px;
        border-radius: 15px;

        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .model-info-title {
        color: white;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .model-info-text {
        color: #9fa6bd;
        font-size: 13px;
        margin-bottom: 9px;
    }

    .model-status {
        color: #8fe7b2;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 7px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #52e28c;
        box-shadow: 0 0 10px rgba(82,226,140,0.8);
    }


    /* --------------------------------------------------------
       MAIN CARD
       -------------------------------------------------------- */

    .main-card {
        padding: 38px 42px 30px 42px;

        border-radius: 28px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.075),
                rgba(255,255,255,0.025)
            );

        border: 1px solid rgba(255,255,255,0.13);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.10);

        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);

        margin-top: 15px;
    }


    /* --------------------------------------------------------
       HEADER
       -------------------------------------------------------- */

    .security-badge {
        width: fit-content;
        margin: 0 auto 15px auto;

        padding: 7px 14px;

        border-radius: 999px;

        background: rgba(99, 80, 255, 0.12);
        border: 1px solid rgba(125, 110, 255, 0.25);

        color: #b9b0ff;

        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .main-title {
        text-align: center;
        font-size: clamp(34px, 5vw, 48px);
        line-height: 1.1;
        font-weight: 850;
        letter-spacing: -1.8px;

        color: white !important;

        margin: 0;
        text-shadow: 0 0 28px rgba(255,255,255,0.15);
    }

    .subtitle {
        text-align: center;
        color: #a7adbf !important;
        font-size: 16px;

        margin-top: 12px;
        margin-bottom: 30px;
    }


    /* --------------------------------------------------------
       INPUT
       -------------------------------------------------------- */

    .input-label {
        color: #e8ebf5;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .input-help {
        color: #858ca2;
        font-size: 12px;
        margin-bottom: 10px;
    }

    .stTextArea textarea {

        min-height: 210px !important;

        background:
            linear-gradient(
                135deg,
                rgba(0,0,0,0.33),
                rgba(14,13,35,0.42)
            ) !important;

        color: #f5f7ff !important;

        border:
            1px solid rgba(255,255,255,0.13)
            !important;

        border-radius: 17px !important;

        padding: 18px !important;

        font-size: 14px !important;

        line-height: 1.6 !important;

        transition: all 0.25s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(126,111,255,0.85) !important;

        box-shadow:
            0 0 0 3px rgba(108,86,255,0.12),
            0 0 25px rgba(91,77,255,0.08) !important;
    }

    .stTextArea textarea::placeholder {
        color: #5f6476 !important;
    }


    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton > button {

        min-height: 48px !important;

        border-radius: 14px !important;

        font-weight: 800 !important;

        border: 1px solid rgba(255,255,255,0.10) !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 10px 25px rgba(0,0,0,0.25) !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background:
            linear-gradient(
                135deg,
                #7567ff,
                #4b8cff
            ) !important;

        color: white !important;

        border: none !important;

        box-shadow:
            0 8px 24px rgba(89,94,255,0.30) !important;
    }


    /* --------------------------------------------------------
       RESULT
       -------------------------------------------------------- */

    .result-card {

        padding: 22px;

        border-radius: 19px;

        margin-top: 20px;
        margin-bottom: 20px;

        text-align: center;

        backdrop-filter: blur(15px);
    }

    .result-spam {

        background:
            linear-gradient(
                135deg,
                rgba(255,55,82,0.13),
                rgba(255,30,65,0.045)
            );

        border:
            1px solid rgba(255,86,108,0.35);

        box-shadow:
            0 0 35px rgba(255,45,72,0.08);
    }

    .result-safe {

        background:
            linear-gradient(
                135deg,
                rgba(49,222,139,0.12),
                rgba(39,180,115,0.04)
            );

        border:
            1px solid rgba(66,220,142,0.30);

        box-shadow:
            0 0 35px rgba(55,220,135,0.07);
    }

    .result-icon {
        font-size: 32px;
        margin-bottom: 6px;
    }

    .result-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #999fb3;

        margin-bottom: 5px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 850;
    }

    .spam-text {
        color: #ff7285;
    }

    .safe-text {
        color: #67e29d;
    }


    /* --------------------------------------------------------
       CONFIDENCE
       -------------------------------------------------------- */

    .confidence-title {
        font-size: 18px;
        font-weight: 800;

        color: #f3f5fb;

        margin-top: 25px;
        margin-bottom: 14px;
    }

    .confidence-row {
        margin-bottom: 15px;
    }

    .confidence-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 7px;

        font-size: 13px;
        font-weight: 700;
        color: #d7dbea;
    }

    .confidence-track {
        height: 9px;

        border-radius: 999px;

        background: rgba(255,255,255,0.07);

        overflow: hidden;
    }

    .confidence-fill-spam {
        height: 100%;
        background: linear-gradient(
            90deg,
            #ff5570,
            #ff304c
        );
        border-radius: inherit;
    }

    .confidence-fill-safe {
        height: 100%;
        background: linear-gradient(
            90deg,
            #48dd96,
            #25ba78
        );
        border-radius: inherit;
    }


    /* --------------------------------------------------------
       FOOTER
       -------------------------------------------------------- */

    .app-footer {
        text-align: center;

        color: #737a90;

        font-size: 12px;

        margin-top: 20px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """

# ============================================================
# DAY MODE CSS
# ============================================================
else:

    custom_css = """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 90%,
                rgba(124,84,255,0.10),
                transparent 38%
            ),
            radial-gradient(
                circle at 90% 5%,
                rgba(0,135,255,0.10),
                transparent 35%
            ),
            #edf2f8;

        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(245,248,252,0.92) !important;
        border-right: 1px solid rgba(20,30,50,0.08);
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;

        display: flex;
        align-items: center;
        justify-content: center;

        background: linear-gradient(
            135deg,
            #eeeaff,
            #e5f3ff
        );

        border: 1px solid rgba(60,80,120,0.10);

        font-size: 25px;
    }

    .brand-title {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #6b7280;
    }

    .side-section-title {
        color: #172033;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .side-step {
        display: flex;
        gap: 12px;
        margin-bottom: 18px;
        color: #172033;
    }

    .side-step > span {
        width: 31px;
        height: 31px;
        flex-shrink: 0;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 10px;

        background: #f0edff;
        border: 1px solid #dfd8ff;

        color: #6555d8;
        font-size: 11px;
        font-weight: 800;
    }

    .side-step small {
        color: #6b7280;
    }

    .model-info {
        padding: 15px;
        border-radius: 15px;

        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(40,50,80,0.08);
    }

    .model-info-title {
        color: #172033;
        font-weight: 700;
    }

    .model-info-text {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 9px;
    }

    .model-status {
        color: #16814d;
        font-size: 12px;
    }

    .status-dot {
        display: inline-block;

        width: 7px;
        height: 7px;

        border-radius: 50%;
        background: #27bd72;

        margin-right: 5px;
    }

    .main-card {

        padding: 38px 42px 30px 42px;

        border-radius: 28px;

        background:
            rgba(255,255,255,0.78);

        border:
            1px solid rgba(255,255,255,0.95);

        box-shadow:
            0 25px 70px rgba(30,45,80,0.10);

        backdrop-filter: blur(25px);

        margin-top: 15px;
    }

    .security-badge {

        width: fit-content;

        margin: 0 auto 15px auto;

        padding: 7px 14px;

        border-radius: 999px;

        background: #f0edff;
        border: 1px solid #ddd5ff;

        color: #6656d7;

        font-size: 11px;
        font-weight: 800;

        letter-spacing: 1px;
    }

    .main-title {

        text-align: center;

        font-size: clamp(34px, 5vw, 48px);

        line-height: 1.1;

        font-weight: 850;

        color: #111827 !important;

        margin: 0;
    }

    .subtitle {

        text-align: center;

        color: #667085 !important;

        font-size: 16px;

        margin-top: 12px;

        margin-bottom: 30px;
    }

    .input-label {

        color: #1f2937;

        font-size: 14px;

        font-weight: 700;

        margin-bottom: 8px;
    }

    .input-help {

        color: #7a8292;

        font-size: 12px;

        margin-bottom: 10px;
    }

    .stTextArea textarea {

        min-height: 210px !important;

        background: rgba(255,255,255,0.9) !important;

        color: #111827 !important;

        border: 1px solid #d7dce5 !important;

        border-radius: 17px !important;

        padding: 18px !important;

        font-size: 14px !important;

        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {

        border-color: #7464ef !important;

        box-shadow:
            0 0 0 3px rgba(116,100,239,0.12) !important;
    }

    .stTextArea textarea::placeholder {
        color: #a0a6b2 !important;
    }

    .stButton > button {

        min-height: 48px !important;

        border-radius: 14px !important;

        font-weight: 800 !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    div[data-testid="stButton"] button[kind="primary"] {

        background:
            linear-gradient(
                135deg,
                #6e5cf4,
                #4488ee
            ) !important;

        color: white !important;

        border: none !important;

        box-shadow:
            0 8px 22px rgba(82,100,230,0.25) !important;
    }

    .result-card {

        padding: 22px;

        border-radius: 19px;

        margin-top: 20px;

        text-align: center;
    }

    .result-spam {

        background: #fff2f4;

        border: 1px solid #ffcbd3;
    }

    .result-safe {

        background: #effbf4;

        border: 1px solid #c2efd5;
    }

    .result-icon {
        font-size: 32px;
    }

    .result-label {

        font-size: 12px;

        text-transform: uppercase;

        letter-spacing: 1px;

        color: #7b8190;
    }

    .result-value {

        font-size: 30px;

        font-weight: 850;
    }

    .spam-text {
        color: #d92d4d;
    }

    .safe-text {
        color: #17864e;
    }

    .confidence-title {

        font-size: 18px;

        font-weight: 800;

        color: #1f2937;

        margin-top: 25px;

        margin-bottom: 14px;
    }

    .confidence-row {
        margin-bottom: 15px;
    }

    .confidence-header {

        display: flex;

        justify-content: space-between;

        margin-bottom: 7px;

        font-size: 13px;

        font-weight: 700;

        color: #343b4a;
    }

    .confidence-track {

        height: 9px;

        border-radius: 999px;

        background: #e6e9ef;

        overflow: hidden;
    }

    .confidence-fill-spam {

        height: 100%;

        background:
            linear-gradient(
                90deg,
                #ff6078,
                #e83c59
            );

        border-radius: inherit;
    }

    .confidence-fill-safe {

        height: 100%;

        background:
            linear-gradient(
                90deg,
                #45d98d,
                #22b873
            );

        border-radius: inherit;
    }

    .app-footer {

        text-align: center;

        color: #7d8492;

        font-size: 12px;

        margin-top: 20px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """


st.markdown(custom_css, unsafe_allow_html=True)


# ============================================================
# MAIN CARD
# ============================================================

st.markdown(
    '<div class="main-card">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="security-badge">
        ✦ MACHINE LEARNING EMAIL SECURITY
    </div>

    <div class="main-title">
        SpamShield AI 🛡️
    </div>

    <div class="subtitle">
        Analyze an email and determine whether it is likely to be spam.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SAMPLE EMAILS
# ============================================================

sample_choice = st.selectbox(
    "Quick test",
    [
        "None — enter my own email",
        "Example: Promotional Spam",
        "Example: Normal Email"
    ],
    label_visibility="collapsed"
)


if sample_choice == "Example: Promotional Spam":

    st.session_state.email_text = (
        "Congratulations! You have been selected to receive a "
        "$1,000 gift card. Click the link below immediately to "
        "claim your reward. This limited-time offer expires today!"
    )

elif sample_choice == "Example: Normal Email":

    st.session_state.email_text = (
        "Hi John,\n\n"
        "Just a reminder that our meeting is scheduled for tomorrow "
        "at 10:00 AM. Please bring the updated report.\n\n"
        "Thanks!"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    """
    <div class="input-label">
        📩 Email Content
    </div>

    <div class="input-help">
        Paste the complete email message below for analysis.
    </div>
    """,
    unsafe_allow_html=True
)

email_text = st.text_area(
    "Email content",
    value=st.session_state.email_text,
    placeholder=(
        "Example:\n\n"
        "Dear customer,\n"
        "You have won a $1,000 gift card...\n"
    ),
    height=220,
    label_visibility="collapsed"
)

st.session_state.email_text = email_text


# ============================================================
# CHARACTER COUNT
# ============================================================

character_count = len(email_text)

st.markdown(
    f"""
    <div style="
        text-align:right;
        color:#80879a;
        font-size:11px;
        margin-top:-8px;
        margin-bottom:12px;
    ">
        {character_count:,} characters
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ACTION BUTTONS
# ============================================================

col1, col2 = st.columns([3, 1])

with col1:

    classify_clicked = st.button(
        "🔍  Analyze Email",
        use_container_width=True,
        type="primary"
    )

with col2:

    clear_clicked = st.button(
        "✕  Clear",
        use_container_width=True
    )


if clear_clicked:

    st.session_state.email_text = ""
    st.session_state.prediction_done = False

    st.rerun()


# ============================================================
# CLASSIFICATION
# ============================================================

if classify_clicked:

    if not email_text.strip():

        st.warning(
            "Please enter an email message before running the analysis."
        )

    else:

        with st.spinner("Analyzing email..."):

            try:

                # ------------------------------------------------
                # Convert email text into feature representation
                # ------------------------------------------------
                email_vector = vectorizer.transform([email_text])

                # ------------------------------------------------
                # Make prediction
                # ------------------------------------------------
                prediction = model.predict(email_vector)[0]

                # ------------------------------------------------
                # Get probabilities
                # ------------------------------------------------
                probability = model.predict_proba(email_vector)[0]

                # ------------------------------------------------
                # Determine class positions safely
                # ------------------------------------------------
                classes = list(model.classes_)

                spam_index = classes.index(1)
                not_spam_index = classes.index(0)

                prob_spam = probability[spam_index] * 100
                prob_not_spam = probability[not_spam_index] * 100

                is_spam = prediction == 1

                st.session_state.prediction_done = True

            except Exception as e:

                st.error(
                    "An error occurred while analyzing the email."
                )

                st.exception(e)

                st.stop()


        # ========================================================
        # RESULT
        # ========================================================

        if is_spam:

            st.markdown(
                f"""
                <div class="result-card result-spam">

                    <div class="result-icon">
                        🚨
                    </div>

                    <div class="result-label">
                        Classification Result
                    </div>

                    <div class="result-value spam-text">
                        SPAM DETECTED
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "This email contains patterns commonly associated "
                "with spam messages. Avoid clicking links or sharing "
                "personal information unless the sender is verified."
            )

        else:

            st.markdown(
                f"""
                <div class="result-card result-safe">

                    <div class="result-icon">
                        ✅
                    </div>

                    <div class="result-label">
                        Classification Result
                    </div>

                    <div class="result-value safe-text">
                        NOT SPAM
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "The model does not identify this message as spam "
                "based on the patterns it learned."
            )


        # ========================================================
        # CONFIDENCE SCORES
        # ========================================================

        st.markdown(
            '<div class="confidence-title">Confidence Scores</div>',
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
                    <div
                        class="confidence-fill-safe"
                        style="width:{prob_not_spam:.1f}%;">
                    </div>
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
                    <div
                        class="confidence-fill-spam"
                        style="width:{prob_spam:.1f}%;">
                    </div>
                </div>

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
        🛡️ SpamShield AI &nbsp;•&nbsp;
        Powered by Machine Learning &nbsp;•&nbsp;
        Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)
