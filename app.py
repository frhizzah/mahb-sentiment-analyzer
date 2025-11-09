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

# ------------------------
# NLTK setup
# ------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
domain_words = {
    "airport","klia","staff","malaysia","malaysian","flight","terminal","gate","counter",
    "immigration","airline","airlines","plane","arrival","departure","queue","checkin",
    "baggage","luggage"
}
stop_words.update(domain_words)
lemmatizer = WordNetLemmatizer()

# ------------------------
# Preprocessing function
# ------------------------
def get_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def preprocess(text):
    if not isinstance(text, str):
        return ""
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

# ------------------------
# Load models
# ------------------------
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svm = joblib.load("svm_model_3class.pkl")
    return tfidf, svm

tfidf, svm = load_models()

# ------------------------
# Compute confidence
# ------------------------
from scipy.special import softmax

def compute_confidence(model, X):
    decision = model.decision_function(X)
    if len(decision.shape) == 1:
        prob_pos = 1 / (1 + np.exp(-decision))
        return max(prob_pos, 1-prob_pos)
    e_x = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
    return np.max(probs)

# ------------------------
# Predict sentiment
# ------------------------
def predict_sentiment(text):
    processed = preprocess(text)
    X = tfidf.transform([processed])
    pred = svm.predict(X)[0]
    # confidence as percentage
    decision = svm.decision_function(X)
    if len(decision.shape)==1:
        conf = 1/(1+np.exp(-decision))
        conf_pct = float(max(conf,1-conf)*100)
    else:
        probs = softmax(decision, axis=1)
        conf_pct = float(np.max(probs)*100)
    return pred, conf_pct

# ------------------------
# Streamlit UI
# ------------------------
st.title("MAHB Customer Review Sentiment Analyzer")
st.markdown("**Model:** Tuned LinearSVC 3-Class")

user_input = st.text_area("Enter your review:", height=180)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter a review first.")
    else:
        pred, conf_pct = predict_sentiment(user_input)
        st.subheader("Sentiment Result")
        st.markdown(f"**Sentiment:** {pred}")
        st.markdown(f"**Confidence:** {conf_pct:.2f}%")
        st.progress(int(conf_pct))
