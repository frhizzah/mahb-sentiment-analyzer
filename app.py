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

st.set_page_config(page_title="MAHB Sentiment Analyzer", layout="wide")

# -----------------------
# Force NLTK downloads
# -----------------------
nltk_packages = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

# -----------------------
# Stopwords & Lemmatizer
# -----------------------
stop_words = set(stopwords.words('english'))
domain_words = {
    "airport","klia","staff","malaysia","malaysian","flight","terminal","gate","counter",
    "immigration","airline","airlines","plane","arrival","departure","queue","checkin",
    "baggage","luggage"
}
stop_words.update(domain_words)
lemmatizer = WordNetLemmatizer()

# -----------------------
# Preprocessing functions
# -----------------------
def get_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def handle_negation(text):
    text = re.sub(r"\bnot\b\s+(\w+)", r"not_\1", text)
    text = re.sub(r"\bno\b\s+(\w+)", r"no_\1", text)
    return text

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = handle_negation(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(tok, get_pos(tag)) for tok, tag in tagged if tok not in stop_words]
    return " ".join(lemmas)

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svm = joblib.load("svm_model_tuned.pkl")  # should be CalibratedClassifierCV
    return tfidf, svm

tfidf, svm = load_models()

# -----------------------
# Prediction function
# -----------------------
def predict_sentiment(text):
    processed = preprocess(text)
    X = tfidf.transform([processed])
    pred = svm.predict(X)[0]
    probs = svm.predict_proba(X)[0]
    conf_dict = dict(zip(svm.classes_, probs))
    conf_max = conf_dict[pred]
    return pred, conf_max, conf_dict

# -----------------------
# Streamlit UI
# -----------------------
st.title("MAHB Customer Review Sentiment Analyzer")
st.markdown("**Model:** Tuned LinearSVC + CalibratedClassifierCV")

user_input = st.text_area("Enter your review:", height=180)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter a review first.")
    else:
        pred, conf_max, conf_dict = predict_sentiment(user_input)
        st.subheader("Sentiment Result")
        st.markdown(f"**Sentiment:** {pred} ({conf_max*100:.2f}%)")
        st.markdown("**Class Probabilities:**")
        for cls, prob in conf_dict.items():
            st.write(f"{cls}: {prob*100:.2f}%")
