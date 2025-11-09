# app.py
import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import os

st.set_page_config(page_title="MAHB Sentiment Analyzer", layout="wide")

# ========================
# NLTK setup
# ========================
NLTK_DATA_PATH = "nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

nltk.download('punkt', download_dir=NLTK_DATA_PATH)
nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
nltk.download('wordnet', download_dir=NLTK_DATA_PATH)
nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)

stop_words = set(stopwords.words('english'))
domain_words = {
    "airport","klia","staff","malaysia","malaysian","flight","terminal","gate","counter",
    "immigration","airline","airlines","plane","arrival","departure","queue","checkin",
    "baggage","luggage"
}
stop_words.update(domain_words)
lemmatizer = WordNetLemmatizer()

# ========================
# Preprocessing
# ========================
def get_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def preprocess(text):
    if not isinstance(text, str):
        return ""
    # Remove weird chars
    text = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(tok, get_pos(tag))
        for tok, tag in tagged
        if tok not in stop_words
    ]
    return " ".join(lemmas)

# ========================
# Load Models
# ========================
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svm = joblib.load("svm_model_3class.pkl")  # new 3-class SVM
    return tfidf, svm

tfidf, svm = load_models()

# ========================
# Predict
# ========================
def compute_confidence(model, X):
    # Convert decision_function output to softmax probabilities
    decision = model.decision_function(X)
    e_x = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
    return probs  # array of probabilities for all classes

def predict_sentiment(text):
    processed = preprocess(text)
    X = tfidf.transform([processed])
    probs = compute_confidence(svm, X)[0]  # 1D array
    classes = svm.classes_
    pred_idx = np.argmax(probs)
    return classes[pred_idx], probs[pred_idx]*100  # return label and confidence %

# ========================
# Streamlit UI
# ========================
st.title("MAHB Customer Review Sentiment Analyzer")
st.markdown("**Model:** Tuned LinearSVC (3-class)")

user_input = st.text_area("Enter your review:", height=180)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter a review first.")
    else:
        pred, conf = predict_sentiment(user_input)
        st.subheader("Sentiment Result")
        st.markdown(f"**Sentiment:** {pred}")
        st.markdown(f"**Confidence:** {conf:.2f}%")
