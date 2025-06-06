import streamlit as st
import os
import docx
import fitz # PyMuPDF for PDF hyperlink extraction
import re
import phonenumbers
import spacy
import nltk
from collections import Counter
# from PyPDF2 import PdfReader # Not strictly needed if fitz handles all PDF text/link extraction
import textract
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import zipfile
import xml.etree.ElementTree as ET


# --- Setup NLTK & spaCy ---
try:
    nltk.data.find('corpora/stopwords')
except Exception: # Catch a more general Exception if stopwords are not found
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except Exception: # Catch a more general Exception if wordnet is not found
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except Exception: # Catch a more general Exception if punkt is not found
    nltk.download('punkt')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a list of common skills. Expand this based on your needs.
SKILLS = ['python', 'c++', 'java', 'flask', 'streamlit', 'pandas', 'numpy',
          'scikit-learn', 'html', 'css', 'git', 'github', 'linux', 'windows',
          'oop', 'jupyter', 'machine learning', 'data analysis', 'sql', 'tableau',
          'power bi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'tensorflow',
          'pytorch', 'r', 'javascript', 'react', 'angular', 'vue.js', 'node.js']

# Load ML model and vectorizer
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

# --- Text Extraction Helpers ---

def extract_text_and_hyperlinks_from_pdf(pdf_path):
    """
    Extracts text and hyperlinks from a PDF file using PyMuPDF (fitz).
    Returns a tuple: (extracted_text_string, list_of_urls).
    """
    text = ""
    urls = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text() # Extract visible text
            
            # Extract links/annotations
            page_links = page.get_links()
            for link in page_links:
                if link['kind'] == fitz.LINK_URI: # Check if it's a URI link
                    urls.append(link['uri'])
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text or links from PDF: {e}")
        text = "" # Ensure text is empty if extraction fails
        urls = []
    return text.strip(), list(set(urls))

def extract_hyperlinks_from_docx(file_path):
    """
    Reads raw XML for hyperlinks from docx file using zipfile and xml.etree.
    Returns a list of URLs.
    """
    urls = []
    try:
        with zipfile.ZipFile(file_path) as docx_zip:
            if 'word/_rels/document.xml.rels' in docx_zip.namelist():
                rels_content = docx_zip.read('word/_rels/document.xml.rels').decode('utf-8')
                tree = ET.fromstring(rels_content)
                ns = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}
                for rel in tree.findall('r:Relationship', ns):
                    url = rel.attrib.get('Target')
                    # Filter for external URLs (http/https)
                    if url and url.startswith(('http://', 'https://')):
                        urls.append(url)
    except Exception as e:
        # st.warning(f"Could not extract hyperlinks from DOCX: {e}") # Optional: show debug warning
        pass # Silently fail if docx is malformed or no rels found
    return list(set(urls)) # Return unique hyperlinks

# --- Main Text Extraction Function ---
def extract_text(file_upload_object):
    """
    Main function to extract text and file-specific URLs from uploaded resume files.
    Handles PDF, DOCX, and TXT formats.
    Returns a tuple: (extracted_text_string, list_of_urls_found_in_file).
    """
    ext = file_upload_object.name.split('.')[-1].lower()
    temp_path = f"temp_resume_upload.{ext}" # Use a distinct temp file name
    
    # Ensure the file pointer is at the beginning and save to temp file
    file_upload_object.seek(0)
    with open(temp_path, "wb") as f:
        f.write(file_upload_object.read())

    text = ""
    file_specific_urls = [] # List to capture URLs found directly by file parser (PDF/DOCX)

    try:
        if ext == 'pdf':
            text, file_specific_urls = extract_text_and_hyperlinks_from_pdf(temp_path)
            
        elif ext == 'docx':
            doc = docx.Document(temp_path)
            text = '\n'.join([p.text for p in doc.paragraphs])
            file_specific_urls = extract_hyperlinks_from_docx(temp_path) # Call helper for DOCX links

        elif ext == 'txt':
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # For TXT, explicit hyperlinks are rare; regex will catch them later

        else:
            # Fallback for other types using textract
            text = textract.process(temp_path).decode('utf-8')
            
    except Exception as e:
        st.error(f"Error processing {ext} file: {e}")
        text = ""
        file_specific_urls = [] # Ensure URLs are also cleared on error
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return text.strip(), list(set(file_specific_urls)) # Return unique URLs found during extraction

# --- Preprocessing ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1])

# --- Structured Info Extraction ---

def extract_name(text):
    """
    Improved name extraction focusing on the top of the resume and spaCy's PERSON entities.
    """
    # 1. Prioritize lines at the very top of the resume
    lines = text.split('\n')
    top_lines = [line.strip() for line in lines[:8] if line.strip()] # Consider top 8 non-empty lines

    # Try to find a single, prominent capitalized name in the top few lines
    for line in top_lines:
        words = line.split()
        # Look for lines that are mostly capitalized, and have 2-4 words (common for names)
        if 2 <= len(words) <= 4 and all(word[0].isupper() or not word.isalpha() for word in words):
            # Check if it's not likely an email/phone/address line
            if '@' not in line and not re.search(r'\d{5,}', line):
                return line.title() # Return title case for consistency

    # 2. Use spaCy's PERSON entity recognition
    doc = nlp(text)
    
    # Collect all PERSON entities that look like a full name
    potential_names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name_words = ent.text.split()
            # Heuristic: A name should have at least two words, and not too many
            # and generally be capitalized or Title Case
            if 2 <= len(name_words) <= 4 and all(word[0].isupper() for word in name_words if word.isalpha()):
                potential_names.append(ent.text)

    # If multiple potential names, try to pick the one closest to the top of the document
    if potential_names:
        # Sort by their appearance order in the original text
        potential_names.sort(key=lambda x: text.find(x))
        # Often the first valid PERSON entity is the name
        return potential_names[0].title()

    # 3. Fallback: Look for lines that are entirely uppercase in the top section
    for line in top_lines:
        if line.isupper() and 2 <= len(line.split()) <= 4:
            return line.title()

    return "Not found"


def extract_email(text):
    matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return matches[0] if matches else "Not found"

def extract_phone(text):
    # This pattern is more flexible for Indian numbers and handles various formats
    # It looks for 10-15 digits, optionally with + country code, spaces, or hyphens
    matches = re.findall(r'(?:\+?(\d{1,3})[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if matches:
        # Reconstruct the number for consistency, e.g., +91 9876543210
        full_number = ""
        for match in matches:
            country_code = match[0]
            if country_code:
                full_number += "+" + country_code + " "
            full_number += "".join(filter(None, match[1:])) # Join other parts, filtering empty
            return full_number # Return the first found number
    
    # Fallback to phonenumbers library if regex fails, good for international numbers
    try:
        import phonenumbers # Re-import if not at top level
        for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
            return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    except ImportError:
        pass # st.warning("phonenumbers library not installed. Phone extraction might be limited.")
    except Exception:
        pass # Ignore other errors from phonenumbers if parsing fails

    return "Not found"

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for s in SKILLS:
        # Use word boundaries to avoid partial matches (e.g., 'java' in 'javascript')
        # This regex ensures ' java ' or 'java.' or 'java,' etc.
        if re.search(r'\b' + re.escape(s) + r'\b', text):
            found_skills.append(s)
    return found_skills or ["Not found"]

# --- Extract URLs from raw text (regex-based) ---
def extract_urls(text):
    # Regex to catch HTTP/HTTPS URLs explicitly written in text
    url_pattern = r'(https?://[^\s)>\]]+)'
    urls = re.findall(url_pattern, text)
    return list(set(urls)) # Return unique URLs

# --- Summary Generator (Conceptual for Achievements/Projects) ---
def extract_sections(text, keywords):
    sections_content = {}
    current_section = None
    # Compile regex for section headers, case-insensitive, starts with optional whitespace
    # and ends with optional colon, newline or just end of line
    section_patterns = {kw: re.compile(r'^\s*' + re.escape(kw) + r'(?:\s*[:.\-]?\s*[\r\n]|\s*$)', re.IGNORECASE | re.MULTILINE) for kw in keywords}

    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        found_section = False
        for keyword in keywords:
            if section_patterns[keyword].match(line_lower):
                current_section = keyword
                sections_content[current_section] = []
                found_section = True
                break
        
        if not found_section and current_section is not None:
            # Append line to current section until a new section header is found
            # or it's a completely empty line that might delineate sections
            is_new_section_start = False
            for kw in keywords:
                if section_patterns[kw].match(line_lower):
                    is_new_section_start = True
                    break
            
            if not is_new_section_start and line.strip(): # Add non-empty lines
                sections_content[current_section].append(line.strip())
            elif not line.strip() and sections_content[current_section]: # If line is empty, and section already has content, it might be end of section
                 # This is a heuristic and might need fine-tuning
                pass
            
    return {k: "\n".join(v) for k, v in sections_content.items()}


def get_achievements_projects(text):
    # Keywords for identifying sections. Order matters for preference.
    achievements_keywords = ["achievements", "awards", "honors", "accomplishments", "recognition"]
    projects_keywords = ["projects", "portfolio", "key projects", "major projects", "work experience", "experience"] # 'experience' as a fallback

    sections = extract_sections(text, achievements_keywords + projects_keywords)

    extracted_achievements_text = ""
    for kw in achievements_keywords:
        if sections.get(kw):
            extracted_achievements_text = sections[kw]
            break

    extracted_projects_text = ""
    for kw in projects_keywords:
        if sections.get(kw):
            extracted_projects_text = sections[kw]
            break

    # Summarize if content is found, otherwise return 'Not found'
    achievements_summary = summarize_text(extracted_achievements_text, num_sentences=3) if extracted_achievements_text else "Not found"
    projects_summary = summarize_text(extracted_projects_text, num_sentences=5) if extracted_projects_text else "Not found"
    
    return achievements_summary, projects_summary

def generate_summary(name, email, phone, skills, achievements, projects):
    summary = f"""
    ### Resume Summary

    **Name:** {name}  
    **Email:** {email}  
    **Phone:** {phone}  

    **Skills:** {', '.join(skills)}  

    **Achievements:** {achievements}  

    **Projects:** {projects}
    """
    return summary

# --- Resume Text Summarizer ---
def summarize_text(text, num_sentences=5):
    from nltk.tokenize import sent_tokenize, word_tokenize
    if not text.strip(): # Handle empty text
        return ""
    
    sents = sent_tokenize(text)
    if len(sents) <= num_sentences:
        return text # Return original if fewer sentences than requested
    
    # Filter out non-alphanumeric words for frequency counting
    word_freq = Counter([w.lower() for w in word_tokenize(text) if w.isalnum()])
    
    sent_scores = {}
    for i, s in enumerate(sents):
        # Score each sentence based on the frequency of its alphanumeric words
        sent_scores[i] = sum(word_freq[w.lower()] for w in word_tokenize(s) if w.isalnum())
        
    # Get indices of top sentences based on score
    top_sents_indices = sorted(sent_scores, key=sent_scores.get, reverse=True)[:num_sentences]
    
    # Reconstruct summary in original sentence order
    return ' '.join([sents[i] for i in sorted(top_sents_indices)])

# --- ML Prediction ---
def classify_job(text):
    clean = preprocess(text)
    if not clean.strip(): # Handle case where preprocessing results in empty string
        return "Unknown", 0.0

    features = vectorizer.transform([clean])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = prob[model.classes_ == pred][0]
    return pred, confidence

# --- Coding Profile Scrapers ---
# Add a timeout to all requests for better robustness
REQUEST_TIMEOUT = 10 # seconds

def scrape_codechef(url):
    try:
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        res.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        soup = BeautifulSoup(res.text, 'html.parser')
        rating_div = soup.find('div', class_='rating-number')
        stars_div = soup.find('span', class_='rating-star') # Corrected for typical structure

        rating_val = rating_div.text.strip() if rating_div else "N/A"
        stars_val = stars_div.text.strip() if stars_div else "N/A" # Often a single span
        return f"CodeChef Rating: {rating_val}, Stars: {stars_val}"
    except requests.exceptions.Timeout:
        return "CodeChef: Profile not reachable (Connection timed out)."
    except requests.exceptions.RequestException as e:
        return f"CodeChef: Profile not reachable (Network error: {e})."
    except Exception as e:
        return f"CodeChef: Error parsing profile data: {e}"

def scrape_leetcode(url):
    try:
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        # LeetCode's public profile data is often loaded dynamically via JavaScript
        # Simple scraping might only get static elements like the username from URL
        username = url.rstrip('/').split('/')[-1]
        
        # Attempt to find common static elements if available, e.g., profile name
        profile_name_tag = soup.find('div', class_='text-label-1') # This might vary
        profile_name = profile_name_tag.text.strip() if profile_name_tag else "N/A"
        
        return f"LeetCode Profile: {profile_name} (@{username}) (detailed stats often require dynamic rendering)"
    except requests.exceptions.Timeout:
        return "LeetCode: Profile not reachable (Connection timed out)."
    except requests.exceptions.RequestException as e:
        return f"LeetCode: Profile not reachable (Network error: {e})."
    except Exception as e:
        return f"LeetCode: Error parsing profile data: {e}"

def scrape_codeforces(url):
    try:
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Find rating and max rating. These can be tricky due to varying HTML
        rating_tag = soup.select_one('.user-rank') # Current rating
        max_rating_tag = soup.select_one('.info .user-rank-legend') # Max rating is often in a specific legend class

        rating_val = rating_tag.text.strip() if rating_tag else "N/A"
        max_rating_val = ""
        if max_rating_tag:
            # Extract just the rating value from a string like " (max. 1500, Expert)"
            match = re.search(r'\(max\.\s*(\d+)', max_rating_tag.text)
            if match:
                max_rating_val = f", Max Rating: {match.group(1)}"
            else:
                max_rating_val = f", {max_rating_tag.text.strip()}" # Fallback to full text if no match

        return f"Codeforces Rating: {rating_val}{max_rating_val}"
    except requests.exceptions.Timeout:
        return "Codeforces: Profile not reachable (Connection timed out)."
    except requests.exceptions.RequestException as e:
        return f"Codeforces: Profile not reachable (Network error: {e})."
    except Exception as e:
        return f"Codeforces: Error parsing profile data: {e}"

def scrape_hackerrank(url):
    try:
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        # HackerRank's public profiles are notoriously hard to scrape for detailed stats.
        # Often, only the username is easily accessible from the URL.
        username = url.rstrip('/').split('/')[-1]
        
        # Try to find user's name if available in a meta tag or header
        user_name_tag = soup.find('h1', class_='profile-title') # This class may vary
        user_name = user_name_tag.text.strip() if user_name_tag else "N/A"

        return f"HackerRank Profile: {user_name} (@{username}) (detailed stats limited publicly)"
    except requests.exceptions.Timeout:
        return "HackerRank: Profile not reachable (Connection timed out)."
    except requests.exceptions.RequestException as e:
        return f"HackerRank: Profile not reachable (Network error: {e})."
    except Exception as e:
        return f"HackerRank: Error parsing profile data: {e}"

def scrape_github(url):
    try:
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Number of public repositories
        repo_count_tag = soup.find('span', class_='Counter', attrs={'title': re.compile(r'\d+ public')}) # Find the counter specifically for public repos
        repos = repo_count_tag.text.strip() if repo_count_tag else "N/A"
        
        # Number of followers
        followers_tag = soup.find('a', class_='Link--secondary', href=re.compile(r'/followers'))
        followers_count = ""
        if followers_tag:
            span_count = followers_tag.find('span', class_='text-bold')
            followers_count = span_count.text.strip() if span_count else "N/A"
        
        return f"GitHub: {repos} public repos, {followers_count} followers"
    except requests.exceptions.Timeout:
        return "GitHub: Profile not reachable (Connection timed out)."
    except requests.exceptions.RequestException as e:
        return f"GitHub: Profile not reachable (Network error: {e})."
    except Exception as e:
        return f"GitHub: Error parsing profile data: {e}"

# --- Detect coding platform from URL ---
def detect_platform(url):
    domain = urlparse(url).netloc.lower()
    if 'codechef.com' in domain:
        return 'codechef'
    if 'leetcode.com' in domain:
        return 'leetcode'
    if 'codeforces.com' in domain:
        return 'codeforces'
    if 'hackerrank.com' in domain:
        return 'hackerrank'
    if 'github.com' in domain:
        return 'github'
    if 'linkedin.com' in domain: # Added LinkedIn detection
        return 'linkedin'
    if 'geeksforgeeks.org' in domain: # Added GFG detection
        return 'gfg'
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Parser & Job Classifier with Coding Profiles", layout="wide")

st.title("Resume Parser & Job Classifier with Coding Profiles")
st.markdown("Upload your resume to extract details, generate a summary, classify job role, and fetch coding profile stats.")

file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

# Add a clear button
if st.session_state.get('processed_file', None) is not None:
    if st.button("Clear Processed Data"):
        st.session_state['processed_file'] = None
        st.experimental_rerun() # Rerun to clear the display

if st.button("Process Resume"):
    if file is not None:
        st.session_state['processed_file'] = file.name # Store the file name to check if something is processed
        with st.spinner("Processing..."):
            # CALL THE MODIFIED EXTRACT_TEXT HERE
            text, file_extracted_urls = extract_text(file) # Capture both text and URLs

            if not text:
                st.warning("Failed to extract text from the resume. Please try a different file or format.")
                st.stop()

            # Extract structured info (using the 'text' variable)
            name = extract_name(text)
            email = extract_email(text)
            phone = extract_phone(text)
            skills = extract_skills(text)

            # Extract URLs from the general text content (e.g., if URLs are just typed out)
            urls_from_text_regex = extract_urls(text)
            
            # Combine all collected URLs and get unique ones
            all_urls = list(set(file_extracted_urls + urls_from_text_regex))
            
            # Filter coding profile URLs
            coding_urls = [u for u in all_urls if detect_platform(u)]
            
            # Separate out LinkedIn and GFG as they are not "coding profiles" in the same sense as competitive programming sites
            linkedin_urls = [u for u in coding_urls if detect_platform(u) == 'linkedin']
            gfg_urls = [u for u  in coding_urls if detect_platform(u) == 'gfg']
            # Remove them from coding_urls if you want to display them separately
            coding_urls = [u for u in coding_urls if detect_platform(u) not in ['linkedin', 'gfg']]


            # Dynamically extract achievements and projects
            achievements, projects = get_achievements_projects(text)

            # Display resume summary
            st.subheader("Resume Summary")
            st.markdown(generate_summary(name, email, phone, skills, achievements, projects))

            # Classify job role
            job, conf = classify_job(text)
            st.subheader("Predicted Job Role")
            st.info(f"**{job}** with confidence **{conf*100:.2f}%**")

            # Show coding profiles info
            if coding_urls or linkedin_urls or gfg_urls:
                st.subheader("Detected Online Profiles & Stats")
                
                # Display Competitive Programming Profiles
                if coding_urls:
                    st.markdown("##### Competitive Programming & Code Hosting")
                    for url in coding_urls:
                        platform = detect_platform(url)
                        st.markdown(f"**{platform.title()} Profile:** [{url}]({url})")
                        if platform == 'codechef':
                            st.write(scrape_codechef(url))
                        elif platform == 'leetcode':
                            st.write(scrape_leetcode(url))
                        elif platform == 'codeforces':
                            st.write(scrape_codeforces(url))
                        elif platform == 'hackerrank':
                            st.write(scrape_hackerrank(url))
                        elif platform == 'github':
                            st.write(scrape_github(url))
                        else:
                            st.write("No specific scraper available for this platform.")
                
                # Display LinkedIn Profiles
                if linkedin_urls:
                    st.markdown("##### Professional Networking")
                    for url in linkedin_urls:
                        st.markdown(f"**LinkedIn Profile:** [{url}]({url})")
                        # You could add a simple LinkedIn scraper if you wish,
                        # but it's usually very complex due to login requirements and dynamic content.
                        st.write("LinkedIn profile detected.")
                
                # Display GFG Profiles
                if gfg_urls:
                    st.markdown("##### Learning & Coding Resources")
                    for url in gfg_urls:
                        st.markdown(f"**GeeksforGeeks Profile:** [{url}]({url})")
                        # You could add a GFG scraper here similar to others.
                        st.write("GeeksforGeeks profile detected.")
            else:
                st.info("No common coding profile URLs (CodeChef, LeetCode, Codeforces, HackerRank, GitHub, LinkedIn, GeeksforGeeks) detected in the resume.")
    else:
        st.warning("Please upload a resume file to process.")

st.markdown("---")
st.caption("Developed by Vardhan Bharathula")