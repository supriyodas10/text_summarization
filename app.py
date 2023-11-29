from flask import Flask, render_template, request
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from bs4 import BeautifulSoup
import urllib.request

app = Flask(__name__)

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Function to preprocess and tokenize text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Function to summarize text using TF-IDF algorithm
def summarize_text(text, num_sentences=5):
    preprocessed_text = preprocess_text(text)

    # Calculate TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get TF-IDF scores for each word
    tfidf_scores = tfidf_matrix[0].toarray()

    # Sort words based on their TF-IDF scores
    sorted_indices = tfidf_scores.argsort()[0][::-1]

    # Get the top words
    top_words = [feature_names[idx] for idx in sorted_indices]

    # Get the sentences that contain the top words
    sentences = [sentence.text for sentence in nlp(preprocessed_text).sents]
    relevant_sentences = []
    for sentence in sentences:
        if any(word in sentence.lower() for word in top_words):
            relevant_sentences.append(sentence)

    # Take the specified number of sentences for the summary
    summary = ' '.join(relevant_sentences[:num_sentences])

    return summary

# Function to read text from different sources
def read_text(source, request_data):
    try:
        if source == 1:
            return request_data['text_input']
        elif source == 2:
            file_content = request.files['file_input'].read().decode('utf-8')
            return file_content
        elif source == 3:
            url = request_data['url_input']
            return read_wikipedia(url)
        else:
            return ""
    except Exception as e:
        print(f"Error reading in text file: {e}")
        return ""    

# Function to read text from a PDF 
def read_pdf(file_content):
    text = ""
    pdf_reader = PyPDF2.PdfFileReader(file_content)
    num_pages = pdf_reader.numPages
    for page_num in range(num_pages):
        page = pdf_reader.getPage(page_num)
        text += page.extractText()
    return text

# Function to read text from a Wikipedia page
def read_wikipedia(url):
    try:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
        return text
    except Exception as e:
        print(f"Error reading Wikipedia page: {e}")
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None

    if request.method == 'POST':
        source_choice = int(request.form['source_choice'])
        input_text = read_text(source_choice, request.form)

        if input_text:
            summary = summarize_text(input_text)

    return render_template('index.html', summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
