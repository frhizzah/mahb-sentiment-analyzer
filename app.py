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

# --- Streamlit page config ---
st.set_page_config(page_title="MAHB Sentiment Analyzer", layout="wide")

# --- NLTK setup ---
NLTK_DATA_PATH = "nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_PATH)  # ensure local folder is searched first

# Download required NLTK packages if missing
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_PATH, quiet=True)

# --- Preprocessing setup ---
stop_words = set(stopwords.words('english'))
domain_words = {
    "airport","klia","staff","malaysia","malaysian","flight","terminal","gate","counter",
    "immigration","airline","airlines","plane","arrival","departure","queue","checkin",
    "baggage","luggage"
}
stop_words.update(domain_words)
lemmatizer = WordNetLemmatizer()

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def preprocess(text):
    """Normalize, clean, tokenize, remove stopwords, and lemmatize text."""
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

# --- Load model & vectorizer ---
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svm = joblib.load("svm_model_tuned.pkl")
    return tfidf, svm

tfidf, svm = load_models()

def compute_confidence(model, X):
    """Compute confidence as max probability using decision function."""
    decision = model.decision_function(X)
    if len(decision.shape) == 1:
        prob_pos = 1 / (1 + np.exp(-decision))
        return max(prob_pos, 1 - prob_pos)
    e_x = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
    return np.max(probs)

# --- Streamlit UI ---
st.title("MAHB Customer Review Sentiment Analyzer")
st.markdown("**Model:** Tuned LinearSVC")

user_input = st.text_area("Enter your review:", height=180)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter a review first.")
    else:
        # Preprocess and predict
        processed = preprocess(user_input)
        X = tfidf.transform([processed])
        pred = svm.predict(X)[0]
        confidence = compute_confidence(svm, X)

        # Display results
        st.subheader("Sentiment Result")
        st.markdown(f"**Sentiment:** {pred}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        st.progress(int(confidence * 100))
