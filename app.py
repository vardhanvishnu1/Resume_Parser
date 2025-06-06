import streamlit as st

st.set_page_config(page_title="Resume Parser & Job Classifier", layout="wide")

import os
import io
import docx
from PyPDF2 import PdfReader
import textract
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter

# NLTK & spaCy setup
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------ Skill keywords for matching ------------------
SKILLS = ['python', 'java', 'c++', 'sql', 'excel', 'machine learning', 'deep learning', 'communication', 'html', 'css']

# ------------------ Extractor Functions ------------------
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    return "Not found"

def extract_email(text):
    match = re.search(r'\S+@\S+', text)
    return match.group() if match else "Not found"

def extract_phone(text):
    match = re.search(r'(\+?\d[\d\-\(\) ]{8,}\d)', text)
    return match.group() if match else "Not found"

def extract_skills(text):
    words = nltk.word_tokenize(text.lower())
    found = list(set([word for word in words if word in SKILLS]))
    return found if found else ["Not found"]

# ------------------ ML Model Load ------------------
try:
    import joblib
    model = joblib.load('logistic_regression_model.pkl') 
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') 
    st.success("Trained model and TF-IDF vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please train and save them.")
    st.stop()

# ------------------ Preprocessing ------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ------------------ Extract Text from Resume ------------------
def extract_text_from_resume(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    text = ""
    temp_file_path = f"temp_resume.{file_extension}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        if file_extension == 'pdf':
            pdf_reader = PdfReader(temp_file_path)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file_extension == 'docx':
            doc = docx.Document(temp_file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension == 'txt':
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = textract.process(temp_file_path).decode('utf-8')
    except Exception as e:
        st.error(f"Error reading file: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return text.strip()

# ------------------ Match Resume to Job ------------------
def match_resume_to_job(resume_text, model, tfidf_vectorizer):
    if not resume_text:
        return "No resume text to classify."
    processed = preprocess(resume_text)
    features = tfidf_vectorizer.transform([processed])
    predicted = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = prob[model.classes_ == predicted][0]
    return f"**Predicted Job Category:** {predicted} (Confidence: {confidence:.2f})"

# ------------------ Summarize Resume ------------------
def summarize_resume(text, num_sentences=5):
    from nltk.tokenize import sent_tokenize, word_tokenize
    if not text:
        return "No text to summarize."
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
    freq = Counter(words)
    scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                scores[i] = scores.get(i, 0) + freq[word]
    top_idxs = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    return " ".join([sentences[i] for i in sorted(top_idxs)])

# ------------------ Streamlit UI ------------------

st.title("📄 Resume Parser & Job Classifier")
st.markdown("Upload a resume to extract structured details, generate a summary, and classify job role.")

st.header("1. Upload Resume")
resume_file = st.file_uploader("Upload your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

st.markdown("---")
if st.button("Process Resume & Classify Job"):
    if resume_file is not None:
        with st.spinner("Extracting text and analyzing..."):
            extracted_text = extract_text_from_resume(resume_file)

            if extracted_text:
                # Show extracted text
                st.subheader("Extracted Resume Text")
                st.write(extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text)

                # Structured Info
                st.subheader("Structured Resume Info")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Name:** {extract_name(extracted_text)}")
                    st.markdown(f"**Email:** {extract_email(extracted_text)}")
                with col2:
                    st.markdown(f"**Phone:** {extract_phone(extracted_text)}")
                    st.markdown(f"**Skills:** {', '.join(extract_skills(extracted_text))}")

                # Summary
                st.subheader("Resume Summary")
                summary = summarize_resume(extracted_text, num_sentences=7)
                st.info(summary)

                # Job Classification
                st.subheader("Job Classification Result")
                classification = match_resume_to_job(extracted_text, model, tfidf_vectorizer)
                st.success(classification)
            else:
                st.warning("Could not extract text from the resume.")
    else:
        st.warning("Please upload a resume file.")

st.markdown("---")
st.caption("Developed by Vardhan Bharathula")
