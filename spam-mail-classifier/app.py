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

# --- Custom CSS for styling ---
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: 800;
            color: #1E88E5; /* Switched to a cleaner blue */
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 25px;
        }
        .result-box {
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .spam {
            background-color: #ffebee;
            color: #d32f2f;
            border: 2px solid #ef9a9a;
        }
        .not-spam {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 2px solid #a5d6a7;
        }
        /* Hide Streamlit default footer */
        footer {visibility: hidden;}
        /* Make text area stand out a bit more */
        .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #cccccc;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/spam.png", width=60) # Add a little icon
    st.title("ℹ️ How to Use")
    st.markdown("""
    1. **Paste** the text of a suspicious email into the main text box.
    2. **Click** the 'Classify' button.
    3. **Review** the prediction and confidence scores.
    
    ---
    **About:**
    This app uses a Machine Learning model (Naive Bayes) to analyze text patterns common in spam emails.
    """)

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
