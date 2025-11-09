import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
import os

st.set_page_config(page_title="MAHB Sentiment Analyzer", layout="wide")

# ===== NLTK Setup =====
NLTK_DATA_PATH = "nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download required resources
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon']:
    try:
        nltk.data.find(resource)
    except:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

stop_words = set(stopwords.words('english'))
domain_words = {
    "airport","klia","staff","malaysia","malaysian","flight","terminal","gate","counter",
    "immigration","airline","airlines","plane","arrival","departure","queue","checkin",
    "baggage","luggage"
}
stop_words.update(domain_words)
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()


# ===== Text Preprocessing =====
def get_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def handle_negation(text):
    # simple negation handling
    text = re.sub(r"\b(not|never|no)\b\s+(\w+)", r"\1_\2", text)
    return text

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = handle_negation(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(tok, get_pos(tag))
        for tok, tag in tagged
        if tok not in stop_words
    ]
    return " ".join(lemmas)


# ===== Load Models =====
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        svm = joblib.load("svm_model_tuned.pkl")
        return tfidf, svm
    except Exception as e:
        st.error("Model files not found or not trained. Please check paths.")
        st.stop()

tfidf, svm = load_models()


# ===== Prediction Function =====
def compute_confidence(model, X):
    decision = model.decision_function(X)
    if len(decision.shape) == 1:  # binary
        prob_pos = 1 / (1 + np.exp(-decision))
        return max(prob_pos, 1-prob_pos)
    e_x = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
    return np.max(probs)

def predict_sentiment(text):
    processed = preprocess(text)
    X = tfidf.transform([processed])
    svm_pred = svm.predict(X)[0]
    confidence = compute_confidence(svm, X)

    # Ensemble with VADER for neutral detection
    vader_scores = sia.polarity_scores(text)
    if 0.45 < vader_scores['neu'] < 0.95:
        final_pred = "Neutral"
        confidence = vader_scores['neu']
    else:
        final_pred = svm_pred

    return final_pred, confidence


# ===== Streamlit UI =====
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
