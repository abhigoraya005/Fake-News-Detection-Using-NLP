import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("models/random_forest_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Text Preprocessing Function
# -----------------------------
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    return text
# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Detection Using NLP")

st.markdown(
"""
Detect whether a news article is **Fake** or **Real**
using Natural Language Processing and Machine Learning.
"""
)

# Sidebar
st.sidebar.header("Project Information")

st.sidebar.success("Random Forest Classifier")

st.sidebar.write("Accuracy : **99.82%**")

st.sidebar.write("Dataset : **44,182 Articles**")

st.sidebar.write("Features : **TF-IDF (5000)**")

st.sidebar.write("Developer")

st.sidebar.info("Abhiney Kumar\n\nNIT Jalandhar")

# Input
news = st.text_area(
    "Paste News Article Here",
    height=250
)

# Predict
if st.button("Predict News"):

    if news.strip() == "":
        st.warning("Please enter a news article.")
    else:

        cleaned = preprocess(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector).max() * 100

        st.divider()

        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.metric(
            "Confidence",
            f"{probability:.2f}%"
        )

st.divider()

st.caption(
    "Fake News Detection Using NLP | Random Forest + TF-IDF | NIT Jalandhar"
)