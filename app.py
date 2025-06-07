import streamlit as st
import os
import docx
import fitz
import re
import phonenumbers
import spacy
import nltk
from collections import Counter
import textract
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import tempfile
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

SKILLS = ['python', 'c++', 'java', 'flask', 'streamlit', 'pandas', 'numpy',
          'scikit-learn', 'html', 'css', 'git', 'github', 'linux', 'windows',
          'oop', 'jupyter', 'machine learning', 'data analysis', 'sql', 'tableau',
          'power bi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'tensorflow',
          'pytorch', 'r', 'javascript', 'react', 'angular', 'vue.js', 'node.js']

try:
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as the script.")
    st.info("You'll need to train your machine learning model and save these files first. Refer to the project documentation for training instructions.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model/vectorizer: {e}")
    st.stop()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Returns the extracted text string.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        text = ""
    return text.strip()

def extract_text(file_upload_object):
    """
    Main function to extract text from uploaded resume files.
    Handles PDF, DOCX, and TXT formats.
    Returns the extracted text string.
    """
    ext = file_upload_object.name.split('.')[-1].lower()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    file_upload_object.seek(0)
    temp_file.write(file_upload_object.read())
    temp_file.close()
    temp_path = temp_file.name

    text = ""

    try:
        if ext == 'pdf':
            text = extract_text_from_pdf(temp_path)
            
        elif ext == 'docx':
            doc = docx.Document(temp_path)
            text = '\n'.join([p.text for p in doc.paragraphs])

        elif ext == 'txt':
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()

        else:
            text = textract.process(temp_path).decode('utf-8')
            
    except Exception as e:
        st.error(f"Error processing {ext} file: {e}")
        text = ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return text.strip()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1])

def extract_name(text):
    """
    Improved name extraction focusing on the top of the resume and spaCy's PERSON entities.
    """
    lines = text.split('\n')
    top_lines = [line.strip() for line in lines[:8] if line.strip()]

    for line in top_lines:
        words = line.split()
        if 2 <= len(words) <= 4 and all(word[0].isupper() or not word.isalpha() for word in words):
            if '@' not in line and not re.search(r'\d{5,}', line):
                return line.title()

    doc = nlp(text)
    
    potential_names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name_words = ent.text.split()
            if 2 <= len(name_words) <= 4 and all(word[0].isupper() for word in name_words if word.isalpha()):
                potential_names.append(ent.text)

    if potential_names:
        potential_names.sort(key=lambda x: text.find(x))
        return potential_names[0].title()

    for line in top_lines:
        if line.isupper() and 2 <= len(line.split()) <= 4:
            return line.title()

    return "Not found"


def extract_email(text):
    matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return matches[0] if matches else "Not found"

def extract_phone(text):
    matches = re.findall(r'(?:\+?(\d{1,3})[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if matches:
        full_number = ""
        for match in matches:
            country_code = match[0]
            if country_code:
                full_number += "+" + country_code + " "
            full_number += "".join(filter(None, match[1:]))
            return full_number
    
    try:
        for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
            return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    except Exception:
        pass

    return "Not found"

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for s in SKILLS:
        if re.search(r'\b' + re.escape(s) + r'\b', text):
            found_skills.append(s)
    return found_skills or ["Not found"]

def extract_sections(text, keywords):
    sections_content = {}
    current_section = None
    # Ensure 'education' and related terms are always considered for section extraction
    all_keywords = set(keywords + ["education", "academic qualifications", "academics"])
    section_patterns = {kw: re.compile(r'^\s*' + re.escape(kw) + r'(?:\s*[:.\-]?\s*[\r\n]|\s*$)', re.IGNORECASE | re.MULTILINE) for kw in all_keywords}

    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        found_section = False
        for keyword in all_keywords: # Iterate over all relevant keywords
            if section_patterns[keyword].match(line_lower):
                current_section = keyword
                sections_content[current_section] = []
                found_section = True
                break
        
        if not found_section and current_section is not None:
            is_new_section_start = False
            for kw in all_keywords: # Check if the current line starts a new known section
                if section_patterns[kw].match(line_lower):
                    is_new_section_start = True
                    break
            
            if not is_new_section_start and line.strip():
                sections_content[current_section].append(line.strip())
            elif not line.strip() and sections_content[current_section]:
                pass
            
    return {k: "\n".join(v) for k, v in sections_content.items()}

def summarize_text(text, num_sentences=5):
    # This function is now mostly for general summary if needed, not for specific bullet point formatting
    if not text.strip():
        return ""
    
    sents = sent_tokenize(text)
    if len(sents) <= num_sentences:
        return text
    
    word_freq = Counter([w.lower() for w in word_tokenize(text) if w.isalnum()])
    
    sent_scores = {}
    for i, s in enumerate(sents):
        sent_scores[i] = sum(word_freq[w.lower()] for w in word_tokenize(s) if w.isalnum())
        
    top_sents_indices = sorted(sent_scores, key=sent_scores.get, reverse=True)[:num_sentences]
    
    # Reconstruct summary in original sentence order
    sorted_sents = sorted(top_sents_indices)
    return ' '.join([sents[i] for i in sorted_sents])

def extract_project_tech_stack(project_text, skills_list):
    """
    Extracts tech stack keywords from project description text.
    """
    if not project_text.strip():
        return "No Projects Done."
    
    text_lower = project_text.lower()
    found_tech = []
    
    for skill in skills_list:
        # Use word boundaries to match whole words
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_tech.append(skill)
            
    if found_tech:
        return "Key Technologies: " + ", ".join(sorted(list(set(found_tech)))) + "."
    else:
        return "Tech stack not explicitly mentioned or recognized."

def format_achievements(achievements_text, max_bullets=5): # Default max_bullets for general achievements
    """
    Extracts top sentences from achievement text and formats them as a bulleted list.
    Aims for conciseness by selecting a limited number of high-scoring sentences.
    """
    if not achievements_text.strip():
        return "Not found"
    
    # Tokenize the achievements text into sentences
    sents = sent_tokenize(achievements_text.strip())
    
    # Filter out very short or non-descriptive sentences (e.g., less than 4 words)
    meaningful_sents = [s.strip() for s in sents if len(s.split()) > 3] 
    
    if not meaningful_sents:
        return "Not found"

    # Calculate word frequency for scoring
    word_freq = Counter([w.lower() for w in word_tokenize(achievements_text) if w.isalnum()])
    
    sent_scores = {}
    for i, s in enumerate(meaningful_sents):
        score = sum(word_freq[w.lower()] for w in word_tokenize(s) if w.isalnum())
        sent_scores[i] = score
        
    # Get indices of top 'max_bullets' scoring sentences
    # If fewer meaningful sentences than max_bullets, take all
    top_sents_indices = sorted(sent_scores, key=sent_scores.get, reverse=True)[:min(len(meaningful_sents), max_bullets)]
    
    # Sort these indices to display achievements in their original order
    sorted_sents_indices = sorted(top_sents_indices)
    
    formatted_achievements_list = []
    for i in sorted_sents_indices:
        formatted_achievements_list.append(f"<li>{meaningful_sents[i]}</li>")
            
    if formatted_achievements_list:
        return "<ul>" + "".join(formatted_achievements_list) + "</ul>"
    else:
        return "Not found"


def get_achievements_projects(text):
    general_achievements_keywords = ["achievements", "awards", "honors", "accomplishments", "recognition"]
    projects_keywords = ["projects", "portfolio", "key projects", "major projects", "work experience", "experience"]

    all_sections = extract_sections(text, general_achievements_keywords + projects_keywords)

    extracted_general_achievements_text = ""
    for kw in general_achievements_keywords:
        if all_sections.get(kw):
            extracted_general_achievements_text = all_sections[kw]
            break

    extracted_projects_text = ""
    for kw in projects_keywords:
        if all_sections.get(kw):
            extracted_projects_text = all_sections[kw]
            break

    achievements_formatted = format_achievements(extracted_general_achievements_text) 
    projects_summary = extract_project_tech_stack(extracted_projects_text, SKILLS)
    
    return achievements_formatted, projects_summary 

def extract_cpi(text):
    """
    Extracts B.Tech CPI/CGPA/GPA specifically from the resume text.
    """
    education_keywords = ["education", "academic qualifications", "academics"]
    sections = extract_sections(text, education_keywords)
    education_text = ""
    for kw in education_keywords:
        if sections.get(kw):
            education_text = sections[kw]
            break

    if not education_text.strip():
        return "Not found"

    # Keywords to identify B.Tech or equivalent degree sections
    btech_keywords = [
        r'bachelor\s*of\s*technology',
        r'b\.?tech',
        r'engineering',
        r'b\.?e\b', # Bachelor of Engineering
        r'undergraduate'
    ]

    lines = education_text.split('\n')
    btech_start_index = -1
    
    # Find the start index of the B.Tech block
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(re.search(kw, line_lower) for kw in btech_keywords):
            btech_start_index = i
            break # Found the start of B.Tech block

    search_text_for_cpi = ""
    if btech_start_index != -1:
        btech_end_index = -1
        # Now find the end of this block. It ends when a new degree/board entry begins
        # or a very long empty sequence occurs.
        for i in range(btech_start_index + 1, len(lines)):
            line_lower = lines[i].lower()
            # General heuristic for new degree/board:
            # - Contains "Board" or "Class" or "X", "XII", "10th", "12th" or "SSC" but not "b.tech" or "engineering" (to avoid matching multiple B.Tech degrees if present)
            # - Or if it's a new post-graduate degree (Master, Ph.D.)
            # - Or if it's a short capitalized line indicating a new degree/board, not a year.
            if (re.search(r'\b(cbse|board|class|x|xii|10th|12th|ssc)\b', line_lower) and not any(re.search(kw, line_lower) for kw in btech_keywords)) or \
               (re.search(r'(master|ph\.?d|post\s*graduate)', line_lower) and len(line_lower.split()) < 6) or \
               (len(lines[i].strip().split()) <= 4 and lines[i].strip() and lines[i].strip()[0].isupper() and not re.search(r'\d{4}', lines[i].strip())): 
                btech_end_index = i
                break
        
        if btech_end_index == -1: # If no explicit end found, take till end of education text
            btech_end_index = len(lines)
        
        # Now, `btech_related_text_lines` will be the relevant block
        btech_related_text_lines = lines[btech_start_index:btech_end_index]
        search_text_for_cpi = "\n".join(btech_related_text_lines).strip()
    else: # If B.Tech keywords were not found, search the entire education text as a fallback
        search_text_for_cpi = education_text

    if not search_text_for_cpi:
        return "Not found"

    # Regex patterns for CGPA/GPA (e.g., 8.5, 9.25/10, 8.9 out of 10)
    cpi_gpa_patterns = [
        re.compile(r'(\d(?:\.\d{1,2})?)\s*(?:CPI|CGPA|GPA|SGPA)(?:\s*\/10|\s*out of 10)?', re.IGNORECASE),
        re.compile(r'(?:CPI|CGPA|GPA|SGPA)\s*[:=]?\s*(\d(?:\.\d{1,2})?)(?:\s*\/10|\s*out of 10)?', re.IGNORECASE),
        re.compile(r'scored\s*(\d(?:\.\d{1,2})?)(?:\s*\/10|\s*out of 10)', re.IGNORECASE),
        re.compile(r'\b(?:aggregate|overall)\s*(\d(?:\.\d{1,2})?)(?:\s*\/10|\s*out of 10)\b', re.IGNORECASE),
        re.compile(r'(\d(?:\.\d{1,2})?)\s*\/\s*10(?:\.0)?', re.IGNORECASE),
        re.compile(r'\b(\d{1}\.\d{1,2})\b', re.IGNORECASE), # Matches single digit.decimal.1-2 digits like 6.93
        re.compile(r'\b(10(?:\.0{1,2})?)\b', re.IGNORECASE) # Matches perfect 10 or 10.0, 10.00
    ]

    # Search for CGPA/GPA within the B.Tech related text
    for pattern in cpi_gpa_patterns:
        matches = pattern.findall(search_text_for_cpi)
        for match in matches:
            try:
                # Ensure match is a string, not a tuple from multiple capture groups if pattern has them
                score_str = match if isinstance(match, str) else match[0] # Take the first group if it's a tuple
                val = float(score_str)
                if 0.0 <= val <= 10.0: # Valid range for typical CGPA/GPA
                    # Format to 2 decimal places if it was a float, for consistency
                    # or as an integer if it's a whole number like 8, 9, 10
                    if '.' in score_str:
                        return f"{val:.2f}/10"
                    else:
                        return f"{int(val)}/10"
            except ValueError:
                continue

    # If no specific B.Tech CGPA/GPA found, return "Not found"
    return "Not found"


def generate_summary(name, email, phone, skills, cpi, projects):
    summary = f"""
    <div style="background-color:#F0F2F6; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 20px;">
        <h4 style="color:#0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; margin-top: 0;">Resume Summary</h4>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Email:</strong> {email}</p>
        <p><strong>Phone:</strong> {phone}</p>
        <p><strong>Skills:</strong> {', '.join(skills)}</p>
        <p><strong>B.Tech Academic Score (CPI/CGPA/GPA):</strong> {cpi}</p> 
        <p><strong>Tech stack used in Projects:</strong> {projects}</p>
    </div>
    """
    return summary

def classify_job(text):
    clean = preprocess(text)
    if not clean.strip():
        return "Unknown", 0.0

    features = vectorizer.transform([clean])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = prob[model.classes_ == pred][0]
    return pred, confidence

def calculate_ats_score(resume_text, job_description_text):
    if not resume_text or not job_description_text:
        return 0.0

    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(job_description_text)
    
    if not processed_resume or not processed_jd:
        return 0.0

    resume_vector = vectorizer.transform([processed_resume])
    jd_vector = vectorizer.transform([processed_jd])

    similarity = cosine_similarity(resume_vector, jd_vector)[0][0]

    ats_score = similarity * 100
    return round(ats_score, 2)

st.set_page_config(page_title="Resume Parser & Job Classifier with ATS Score", layout="wide")

# Custom CSS for a white background and improved aesthetics
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF; /* White background for the entire app */
    color: #333333; /* Default text color */
}
/* Target Streamlit's main content area which often has a dark background by default */
.st-emotion-cache-fis6y9, .st-emotion-cache-1wv7q08, .st-emotion-cache-1kenbb8 {
    background-color: #FFFFFF; /* Ensure main content area is also white */
}

/* Styling for specific Streamlit components to give them a modern look */
.st-emotion-cache-1pbsqon, .st-emotion-cache-1wmy9hq { /* Common classes for headers/titles */
    color: #0056b3; /* A nice blue for titles/headers */
    font-weight: bold;
}

/* Style for input elements and text areas */
.stTextArea > label, .stFileUploader > label {
    font-weight: bold;
    color: #0056b3;
    margin-bottom: 5px;
    display: block;
}
.stTextArea textarea {
    border: 1px solid #cccccc;
    border-radius: 8px;
    padding: 10px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
.stFileUploader {
    border: 2px dashed #cccccc;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    background-color: #f9f9f9;
}
.stFileUploader:hover {
    border-color: #007bff;
}

/* Button Styling */
.stButton>button {
    background-color: #007bff; /* Primary blue */
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: background-color 0.3s ease, transform 0.2s ease;
    margin-top: 15px; /* Add some spacing */
}
.stButton>button:hover {
    background-color: #0056b3; /* Darker blue on hover */
    transform: translateY(-2px);
}
.stButton>button:active {
    transform: translateY(0);
}

/* Info, Success, Warning boxes */
.stAlert {
    border-radius: 8px;
    padding: 15px 20px;
    margin-top: 15px;
    margin-bottom: 15px;
    font-size: 1rem;
}
.stAlert.info {
    background-color: #e0f2f7; /* Light blue */
    color: #007bff;
    border-left: 5px solid #007bff;
}
.stAlert.success {
    background-color: #e6ffe6; /* Light green */
    color: #28a745;
    border-left: 5px solid #28a745;
}
.stAlert.warning {
    background-color: #fff3e0; /* Light orange */
    color: #ffc107;
    border-left: 5px solid #ffc107;
}

/* General text styling */
p {
    line-height: 1.6;
}
strong {
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

st.title("Resume Parser & Job Classifier with ATS Score")
st.markdown("Upload your resume and paste a job description to extract details, generate a summary, classify job role, and get an ATS match score.")

# Added a separator for better visual structure
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Resume Upload") # Added sub-header
    file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])
    if st.session_state.get('processed_file', None) is not None:
        if st.button("Clear Processed Data", key="clear_data"):
            st.session_state['processed_file'] = None
            st.experimental_rerun()

with col2:
    st.header("Job Description") # Added sub-header
    job_description = st.text_area("Paste Job Description Here", height=300, 
                                   help="Copy and paste the full job description text here for ATS matching. This helps the ATS score calculation.")


if st.button("Process Resume & Calculate ATS Score", key="process_button"):
    if file is None:
        st.warning("Please upload a resume file to process.")
    else:
        st.session_state['processed_file'] = file.name
        with st.spinner("Processing..."):
            text = extract_text(file) 

            if not text:
                st.warning("Failed to extract text from the resume. Please try a different file or format.")

            name = extract_name(text)
            email = extract_email(text)
            phone = extract_phone(text)
            skills = extract_skills(text)
            cpi = extract_cpi(text) 

            achievements_formatted, projects_summary = get_achievements_projects(text) 

            st.markdown("---") 
            st.header("Analysis Results")

            if job_description:
                ats_score = calculate_ats_score(text, job_description)
                st.markdown(f"<h3 style='color:#0056b3;'>🎯 ATS Match Score: <span style='color:#28a745;'>**{ats_score}%**</span></h3>", unsafe_allow_html=True)
                if ats_score < 50:
                    st.info("Consider tailoring your resume more closely to the job description's keywords.")
                elif ats_score < 75:
                    st.info("Good match! Review the job description for more specific keywords to improve further.")
                else:
                    st.success("Excellent match! Your resume aligns well with the job description.")
            else:
                st.info("Paste a Job Description to get an ATS Match Score.")

            st.subheader("Resume Summary")
            st.markdown(generate_summary(name, email, phone, skills, cpi, projects_summary), unsafe_allow_html=True)

            job, conf = classify_job(text)
            st.subheader("Predicted Job Role")
            st.info(f"**{job}** with confidence **{conf*100:.2f}%**")

st.markdown("---")
st.caption("Developed by **Vardhan Bharathula**")