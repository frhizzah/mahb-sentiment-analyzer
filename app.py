# app.py
import streamlit as st
import joblib
import numpy as np
import re
import string
import os
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="MAHB Sentiment Analyzer", layout="wide")

# -------------------------------
# NLTK setup
# -------------------------------
NLTK_DATA_PATH = "nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download all required NLTK resources
nltk.download("punkt", download_dir=NLTK_DATA_PATH)
nltk.download("stopwords", download_dir=NLTK_DATA_PATH)
nltk.download("wordnet", download_dir=NLTK_DATA_PATH)
nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DATA_PATH)
nltk.download("vader_lexicon", download_dir=NLTK_DATA_PATH)

# -------------------------------
# Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))
domain_words = {
    "airport","klia","staff","malaysia","malaysian","flight","terminal","gate","counter",
    "immigration","airline","airlines","plane","arrival","departure","queue","checkin",
    "baggage","luggage"
}
stop_words.update(domain_words)
lemmatizer = WordNetLemmatizer()

def get_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

# Handle negations for better accuracy
def handle_negation(text):
    text = re.sub(r"\b(not|no|never|n't)\s+(\w+)", r"not_\2", text)
    return text

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = handle_negation(text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(tok, get_pos(tag))
        for tok, tag in tagged
        if tok not in stop_words
    ]
    return " ".join(lemmas)

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svm = joblib.load("svm_model_tuned.pkl")
    sia = SentimentIntensityAnalyzer()
    return tfidf, svm, sia

tfidf, svm, sia = load_models()

# -------------------------------
# Prediction helpers
# -------------------------------
def compute_svm_confidence(model, X):
    """Compute probability-like confidence for SVM."""
    decision = model.decision_function(X)
    if len(decision.shape) == 1:
        prob_pos = 1 / (1 + np.exp(-decision))
        return max(prob_pos, 1-prob_pos)
    e_x = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
    return np.max(probs)

def predict_sentiment(text):
    """Combine SVM and VADER for better accuracy."""
    processed = preprocess(text)
    X = tfidf.transform([processed])
    
    # SVM prediction
    svm_pred = svm.predict(X)[0]
    svm_conf = compute_svm_confidence(svm, X)
    
    # VADER analysis
    vader_scores = sia.polarity_scores(text)
    if vader_scores['compound'] >= 0.05:
        vader_pred = "Positive"
    elif vader_scores['compound'] <= -0.05:
        vader_pred = "Negative"
    else:
        vader_pred = "Neutral"
    vader_conf = abs(vader_scores['compound'])
    
    # Combine: simple rule-based ensemble
    if vader_pred == "Neutral":
        final_pred = svm_pred
        final_conf = svm_conf
    else:
        # weight VADER and SVM equally
        if vader_pred == svm_pred:
            final_pred = svm_pred
            final_conf = (svm_conf + vader_conf)/2
        else:
            # choose prediction with higher confidence
            final_pred = svm_pred if svm_conf >= vader_conf else vader_pred
            final_conf = max(svm_conf, vader_conf)
    
    return final_pred, final_conf

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("MAHB Customer Review Sentiment Analyzer")
st.markdown("**Model:** Tuned LinearSVC + VADER Ensemble")

user_input = st.text_area("Enter your review:", height=180)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter a review first.")
    else:
        pred, conf = predict_sentiment(user_input)
        st.subheader("Sentiment Result")
        st.markdown(f"**Sentiment:** {pred}")
        st.progress(int(conf * 100))
