import streamlit as st
import joblib
import os

# --- Page configuration (MUST BE FIRST) ---
st.set_page_config(page_title="Spam Mail Classifier", page_icon="📧", layout="centered", initial_sidebar_state="expanded")

# --- Robust File Loading ---
# Get the absolute path of the directory where app.py lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the current script
model_path = os.path.join(current_dir, "spam_classifier_model.joblib")
vectorizer_path = os.path.join(current_dir, "vectorizer.joblib")

@st.cache_resource # Cache the model loading to speed up app performance
def load_models():
    try:
        # Load the models using the absolute paths
        model = joblib.load(model_path)
        v = joblib.load(vectorizer_path)
        return model, v
    except FileNotFoundError as e:
        # Provide a helpful error message if files are missing
        st.error(f"Error loading models: Could not find the necessary .joblib files. Please ensure they are in the same directory as app.py. Details: {e}")
        st.stop() # Stop execution if models fail to load

# Load the models
model, v = load_models()

# --- Sidebar for Theme & Info ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/spam.png", width=60)
    
    # Theme Toggle
    theme = st.radio("🌓 Appearance", ["Night Mode (Neon Glass)", "Day Mode (Frosted Glass)"])
    st.markdown("---")
    
    st.title("ℹ️ How to Use")
    st.markdown("""
    1. **Paste** the text of a suspicious email into the main text box.
    2. **Click** the 'Classify' button.
    3. **Review** the prediction and confidence scores.
    
    ---
    **About:**
    This app uses a Machine Learning model (Naive Bayes) to analyze text patterns common in spam emails.
    """)

# --- Dynamic CSS based on Theme ---
if theme == "Night Mode (Neon Glass)":
    custom_css = """
    <style>
        /* Main Dark App Background */
        .stApp {
            background-color: #0b061f;
            background-image: 
                radial-gradient(circle at 10% 90%, rgba(138, 43, 226, 0.4) 0%, transparent 50%),
                radial-gradient(circle at 90% 10%, rgba(0, 119, 255, 0.3) 0%, transparent 40%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        
        /* Glass Sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(11, 6, 31, 0.6) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Glassmorphism Main Container */
        .block-container {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-radius: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.3);
            border-left: 1px solid rgba(255, 255, 255, 0.3);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            padding: 3rem 2rem !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
            margin-top: 2rem;
        }
        
        /* Typography overrides */
        h1, h2, h3, p, label, .stMarkdown p, li { color: #ffffff !important; }
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #ffffff;
            text-align: center;
            margin-bottom: 5px;
            text-shadow: 0 0 15px rgba(255,255,255,0.3);
        }
        .subtitle {
            font-size: 18px;
            color: #b0b0b0;
            text-align: center;
            margin-bottom: 25px;
        }
        
        /* Glowing Pill Button */
        .stButton > button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 30px !important;
            font-weight: bold !important;
            border: none !important;
            box-shadow: 0 0 15px rgba(255,255,255,0.6) !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 25px rgba(255,255,255,0.9) !important;
        }
        
        /* Dark Input Box */
        .stTextArea textarea {
            background-color: rgba(0, 0, 0, 0.4) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 12px !important;
        }
        .stTextArea textarea::placeholder { color: #888 !important; }
        
        /* Result styling */
        .result-box {
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .spam {
            background: rgba(255, 50, 50, 0.15);
            color: #ff9999 !important;
            border: 1px solid rgba(255, 50, 50, 0.4);
            box-shadow: 0 0 20px rgba(255, 50, 50, 0.2);
        }
        .not-spam {
            background: rgba(50, 255, 100, 0.1);
            color: #99ffbb !important;
            border: 1px solid rgba(50, 255, 100, 0.4);
            box-shadow: 0 0 20px rgba(50, 255, 100, 0.15);
        }
        footer {visibility: hidden;}
    </style>
    """
else:
    custom_css = """
    <style>
        /* Main Light App Background */
        .stApp {
            background-color: #f0f4f8;
            background-image: 
                radial-gradient(circle at 10% 90%, rgba(138, 43, 226, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 90% 10%, rgba(0, 119, 255, 0.1) 0%, transparent 40%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        
        /* Glass Sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.5) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        /* Light Frosted Container */
        .block-container {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 1);
            padding: 3rem 2rem !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
        }
        
        /* Typography overrides */
        h1, h2, h3, p, label, .stMarkdown p, li { color: #111111 !important; }
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #111111;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 25px;
        }
        
        /* Dark Pill Button */
        .stButton > button {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border-radius: 30px !important;
            font-weight: bold !important;
            border: none !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.25) !important;
        }
        
        /* Light Input Box */
        .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.8) !important;
            color: #111 !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            border-radius: 12px !important;
        }
        
        /* Result styling */
        .result-box {
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .spam {
            background: rgba(255, 50, 50, 0.1);
            color: #d32f2f !important;
            border: 1px solid rgba(255, 50, 50, 0.3);
        }
        .not-spam {
            background: rgba(50, 255, 100, 0.15);
            color: #2e7d32 !important;
            border: 1px solid rgba(50, 255, 100, 0.4);
        }
        footer {visibility: hidden;}
    </style>
    """

# Apply the selected CSS
st.markdown(custom_css, unsafe_allow_html=True)

# --- Main UI ---
st.markdown("<div class='main-title'>Spam Mail Classifier 📧</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Instantly check if an email is safe or spam.</div>", unsafe_allow_html=True)

# --- Input box ---
email_text = st.text_area("Paste the email content below:", height=200, placeholder="Dear customer, you have won a $1000 gift card...")

# --- Classify Action ---
# Use columns to center the button slightly
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_clicked = st.button("🔍 Classify Email", use_container_width=True, type="primary")

if classify_clicked:
    if not email_text.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        with st.spinner("Analyzing email content..."):
            # Transform text to numeric form
            email_count = v.transform([email_text])
            prediction = model.predict(email_count)[0]
            probability = model.predict_proba(email_count)[0]

            # Prepare result
            is_spam = bool(prediction == 1)
            label = "Spam" if is_spam else "Not Spam"
            prob_spam = probability[1] * 100
            prob_not_spam = probability[0] * 100

        # Display result with colors
        if is_spam:
            st.markdown(f"<div class='result-box spam'>🚨 Verdict: <b>{label}</b></div>", unsafe_allow_html=True)
            st.error("Exercise caution. This message exhibits common characteristics of spam or phishing attempts.")
        else:
            st.markdown(f"<div class='result-box not-spam'>✅ Verdict: <b>{label}</b></div>", unsafe_allow_html=True)
            st.success("This message appears to be safe and does not match our spam patterns.")

        # Show probabilities nicely formatted
        st.markdown("### Confidence Scores")
        
        # Use progress bars for visual representation of probability
        st.write(f"**Not Spam:** {prob_not_spam:.1f}%")
        st.progress(int(prob_not_spam))
        
        st.write(f"**Spam:** {prob_spam:.1f}%")
        st.progress(int(prob_spam))

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Built with Streamlit • Machine Learning Projects</p>", unsafe_allow_html=True)
