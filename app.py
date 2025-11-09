import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk
import os
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack

st.set_page_config(page_title="MAHB Sentiment Analyzer", layout="wide")

# NLTK setup
NLTK_DATA_PATH = "nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)
for resource in [
    "punkt", "punkt_tab", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"
]:
    nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)

stop_words = set(stopwords.words('english'))
domain_words = {
    "airport","klia","staff","malaysia","malaysian","flight","terminal","gate",
    "counter","immigration","airline","airlines","plane","arrival","departure",
    "queue","checkin","baggage","luggage"
}
stop_words.update(domain_words)
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

negation_words = set(["not","no","never","n't","none","nobody","nothing","neither","nor","nowhere","hardly","scarcely","barely"])

def handle_negation(text, window=3):
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in negation_words:
            out.append(tok)
            for j in range(1, window+1):
                if i+j < len(tokens):
                    out.append(tokens[i+j] + "_NEG")
            i += window + 1
        else:
            out.append(tok)
            i += 1
    return " ".join(out)

def get_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = handle_negation(text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(tok, get_pos(tag))
        for tok, tag in tagged if tok not in stop_words
    ]
    return " ".join(lemmas)

@st.cache_resource
def load_models():
    word = joblib.load("tfidf_word.pkl")
    char = joblib.load("tfidf_char.pkl")
    model = joblib.load("svm_model_tuned_calibrated.pkl")
    return word, char, model

tfidf_word, tfidf_char, model = load_models()

st.title("MAHB Customer Review Sentiment Analyzer")
st.markdown("**Model:** Calibrated LinearSVC (TF-IDF + VADER)")

user_input = st.text_area("Enter your review:", height=180)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter a review first.")
    else:
        processed = preprocess(user_input)
        vader_score = np.array([[sia.polarity_scores(user_input)['compound']]])
        Xw = tfidf_word.transform([processed])
        Xc = tfidf_char.transform([processed])
        X = hstack([Xw, Xc, vader_score])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X).max()

        st.subheader("Sentiment Result")
        st.markdown(f"**Sentiment:** {pred}")
        st.progress(int(prob * 100))
