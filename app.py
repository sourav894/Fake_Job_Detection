streamlit run app.py
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import string

# Load model and vectorizer
model = joblib.load('fake_job_detector.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Download NLTK data if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-alphabetic characters
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("🕵️‍♂️ Fake Job Posting Detector")

st.markdown("""
Enter a job description below, and the model will tell you whether it's likely **Fake** or **Real**.
""")

user_input = st.text_area("📝 Paste Job Description Here:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Prediction: **Fake Job** (Confidence: {probability:.2f})")
        else:
            st.success(f"✅ Prediction: **Real Job** (Confidence: {probability:.2f})")
