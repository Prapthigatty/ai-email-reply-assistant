import nltk
nltk.download("stopwords")


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("intent_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

STOP_WORDS = set(stopwords.words("english"))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

# ---------------- RULE-BASED FALLBACK ----------------
def rule_based_intent(text):
    text = text.lower()
    if "sorry" in text or "apolog" in text:
        return "apology"
    if "follow up" in text or "reminder" in text:
        return "follow_up"
    if "please" in text and ("send" in text or "share" in text or "provide" in text):
        return "request"
    if "know more" in text or "details" in text or "information" in text:
        return "inquiry"
    return None

# ---------------- REPLY GENERATOR ----------------
def generate_reply(intent):
    replies = {
        "request": "Thank you for your email. I will review your request and get back to you shortly.",
        "apology": "Thank you for informing me. No problem at all. Please let me know if you need any further assistance.",
        "follow_up": "Thank you for the follow-up. I will check the status and update you soon.",
        "inquiry": "Thank you for your inquiry. I will share the required information with you shortly."
    }
    return replies.get(intent, "Thank you for your email. I will get back to you shortly.")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Email Reply Assistant", layout="centered")

st.title("📧 AI-Based Email Reply Assistant")
st.write("Paste an email below to get a professional reply suggestion.")

email_input = st.text_area("Enter email content", height=180)

if st.button("Generate Reply"):
    if email_input.strip() == "":
        st.warning("Please enter an email.")
    else:
        # Rule-based first
        rule_intent = rule_based_intent(email_input)

        cleaned_email = clean_text(email_input)
        email_vector = vectorizer.transform([cleaned_email])

        if rule_intent:
            predicted_intent = rule_intent
            confidence = 1.0
        else:
            proba = model.predict_proba(email_vector)[0]
            confidence = max(proba)
            predicted_intent = model.classes_[proba.argmax()]

        st.subheader("🔍 Detected Intent")
        st.success(predicted_intent)

        st.subheader("📊 Confidence Score")
        st.write(f"{confidence:.2f}")

        st.subheader("✉️ Suggested Reply")
        if confidence < 0.5:
            st.warning("Low confidence prediction. Showing a generic professional reply.")
            st.info("Thank you for your email. I will review it and respond shortly.")
        else:
            st.info(generate_reply(predicted_intent))
