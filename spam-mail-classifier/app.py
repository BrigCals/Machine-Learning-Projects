import streamlit as st
import joblib

# --- Load model and vectorizer ---
model = joblib.load("spam-mail-classifier/spam_classifier_model.joblib")
v = joblib.load("spam-mail-classifier/vectorizer.joblib")

# --- Page configuration ---
st.set_page_config(page_title="Spam Mail Classifier", page_icon="📧", layout="centered")

# --- Custom CSS for styling ---
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 10px;
        }
        .result-box {
            font-size: 22px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .spam {
            background-color: #ffcccc;
            color: #b30000;
        }
        .not-spam {
            background-color: #ccffcc;
            color: #006600;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("ℹ️ How to Use")
st.sidebar.write("""
1. Type or paste the email text in the box.  
2. Click **Classify**.  
3. See whether it's Spam or Not Spam.  
""")

# --- Title ---
st.markdown("<div class='main-title'>Spam Mail Classifier 📧</div>", unsafe_allow_html=True)
st.write("Enter an email text below and let the app classify it for you!")

# --- Input box ---
email_text = st.text_area("✉️ Enter email text:")

# --- Classify Button ---
if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text.")
    else:
        # Transform text to numeric form
        email_count = v.transform([email_text])
        prediction = model.predict(email_count)[0]
        probability = model.predict_proba(email_count)[0]

        # Prepare result
        label = "Spam" if prediction == 1 else "Not Spam"
        prob_spam = probability[1] * 100
        prob_not_spam = probability[0] * 100

        # Display result with colors
        if prediction == 1:
            st.markdown(f"<div class='result-box spam'>The email is predicted to be: <b>{label}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box not-spam'>The email is predicted to be: <b>{label}</b></div>", unsafe_allow_html=True)

        # Show probabilities
        st.write(f"**Probability of Spam:** {prob_spam:.2f}%")
        st.write(f"**Probability of Not Spam:** {prob_not_spam:.2f}%")

# --- Footer ---
st.markdown("---")

