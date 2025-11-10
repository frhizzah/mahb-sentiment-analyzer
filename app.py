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

# --- Robust NLTK setup for Streamlit Cloud (v3.8+) ---
NLTK_DATA_PATH = "nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_PATH)

resources = [
    ("punkt_tab", "tokenizers/punkt_tab/english"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")
]

for pkg, subpath in resources:
    try:
        nltk.data.find(subpath)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_PATH, quiet=True)

# --- Preprocessing setup ---
stop_words = set(stopwords.words('english'))
domain_words = {
    "klia", "airport"
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
    
    st.write(f"🔍 DEBUG - Input length: {len(text)} chars")
    
    text = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')
    st.write(f"🔍 DEBUG - After encoding: {len(text)} chars")
    
    text = re.sub(r'\s+', ' ', text).strip()
    st.write(f"🔍 DEBUG - After regex: {len(text)} chars")
    
    text = text.lower()
    
    # Handle contrastive markers - mark comparisons
    contrastive_patterns = [
        (r'(but|however|instead|rather than|unlike|compared to|whereas)\s+(\w+)', r'\1 CONTRAST_\2'),
        (r'(send|go|fly)\s+(to|through)\s+\w+.*?(best|better|excellent)', r'COMPARISON_POSITIVE'),
        (r'(next door|neighbor|other airport).*?(best|better|excellent)', r'COMPARISON_POSITIVE')
    ]
    
    for i, (pattern, replacement) in enumerate(contrastive_patterns):
        before_len = len(text)
        text = re.sub(pattern, replacement, text)
        if len(text) != before_len:
            st.write(f"🔍 DEBUG - Pattern {i+1} matched! Length: {len(text)} chars")
    
    st.write(f"🔍 DEBUG - After all contrastive patterns: {len(text)} chars")
    
    # Negation handling
    text = re.sub(r"\bnot\b\s+(\w+)", r"not_\1", text)
    text = re.sub(r"\bno\b\s+(\w+)", r"no_\1", text)
    
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(tok, get_pos(tag))
        for tok, tag in tagged
        if tok not in stop_words
    ]
    
    result = " ".join(lemmas)
    st.write(f"🔍 DEBUG - Final result length: {len(result)} chars")
    st.write(f"🔍 DEBUG - Number of tokens: {len(lemmas)}")
    
    return result

# --- Load model & vectorizer ---
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svm = joblib.load("svm_model_tuned.pkl")
    return tfidf, svm

tfidf, svm = load_models()

def compute_confidence(model, X):
    """Compute confidence as max probability using predict_proba."""
    probs = model.predict_proba(X)
    return np.max(probs)

# --- Streamlit UI ---
st.title("MAHB Customer Review Sentiment Analyzer")
st.markdown("**Model:** Tuned LinearSVC")

user_input = st.text_area("Enter your review:", height=180)

if st.button("Analyze"):
    if not user_input.strip():
        st.error("Please enter a review first.")
    else:
        try:
            # Preprocess and predict
            processed = preprocess(user_input)
            st.write("**Debug - Processed text:**", processed)
            X = tfidf.transform([processed])
            pred = svm.predict(X)[0]
            confidence = compute_confidence(svm, X)

            # Display results
            st.subheader("Sentiment Result")
            st.markdown(f"**Sentiment:** {pred}")
            st.markdown(f"**Confidence:** {confidence*100:.2f}%")
            st.progress(int(confidence * 100))

            st.write("---")  # Separator line
            st.write("### Debug: Testing individual negative words")
            test_reviews = [
                "worst",
                "terrible", 
                "chaos",
                "bad",
                "worst airport chaos terrible"
            ]
            
            for test in test_reviews:
                proc = preprocess(test)
                X_test = tfidf.transform([proc])
                pred_test = svm.predict(X_test)[0]
                conf_test = compute_confidence(svm, X_test)
                st.write(f"'{test}' → Processed: '{proc}' → Prediction: **{pred_test}** ({conf_test*100:.2f}%)")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
