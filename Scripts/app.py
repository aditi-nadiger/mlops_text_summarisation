import streamlit as st
import math
import bs4 as bs
import urllib.request
import re
import PyPDF2
import nltk
import spacy
from nltk.stem import WordNetLemmatizer 

# Download NLTK resources
nltk.download('wordnet')

# Initialize NLP tools
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

st.title("🧠 Text Summarizer using TF-IDF")
st.write("Upload a `.txt`, `.pdf`, enter text manually, or provide a Wikipedia URL.")

# Read txt file
def file_text(uploaded_file):
    return uploaded_file.read().decode('utf-8').replace("\n", '')

# Read PDF file
def pdfReader(uploaded_file):
    pdfReader = PyPDF2.PdfFileReader(uploaded_file)
    text = ''
    for i in range(pdfReader.numPages):
        page = pdfReader.getPage(i)
        text += page.extractText()
    return text

# Scrape Wikipedia text
def wiki_text(url):
    scrap_data = urllib.request.urlopen(url)
    article = scrap_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = "".join([re.sub(r'\[[0-9]*\]', '', p.text) for p in paragraphs])
    return article_text

# Frequency Matrix
def frequency_matrix(sentences):
    freq_matrix = {}
    stopWords = nlp.Defaults.stop_words

    for sent in sentences:
        freq_table = {}
        words = [word.text.lower() for word in sent if word.text.isalnum()]
        for word in words:
            word = lemmatizer.lemmatize(word)
            if word not in stopWords:
                freq_table[word] = freq_table.get(word, 0) + 1
        freq_matrix[sent[:15]] = freq_table
    return freq_matrix

def tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        tf_table = {}
        total_words = len(freq_table)
        for word, count in freq_table.items():
            tf_table[word] = count / total_words
        tf_matrix[sent] = tf_table
    return tf_matrix

def sentences_per_words(freq_matrix):
    word_count = {}
    for f_table in freq_matrix.values():
        for word in f_table:
            word_count[word] = word_count.get(word, 0) + 1
    return word_count

def idf_matrix(freq_matrix, word_count, total_sentences):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table:
            idf_table[word] = math.log10(total_sentences / float(word_count[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix

def tf_idf_matrix(tf_matrix, idf_matrix):
    tfidf = {}
    for (sent, tf_table), (_, idf_table) in zip(tf_matrix.items(), idf_matrix.items()):
        tfidf_table = {}
        for word in tf_table:
            tfidf_table[word] = tf_table[word] * idf_table.get(word, 0)
        tfidf[sent] = tfidf_table
    return tfidf

def score_sentences(tf_idf_matrix):
    sentence_score = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score = sum(f_table.values())
        total_words = len(f_table)
        if total_words > 0:
            sentence_score[sent] = total_score / total_words
    return sentence_score

def average_score(score_dict):
    return sum(score_dict.values()) / len(score_dict)

def create_summary(sentences, sentence_score, threshold):
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentence_score and sentence_score[sentence[:15]] >= threshold:
            summary += " " + sentence.text
    return summary

# Input section
option = st.radio("Choose input method:", 
    ('Type Text', 'Upload .txt File', 'Upload .pdf File', 'Wikipedia URL'))

text = ""
if option == 'Type Text':
    text = st.text_area("Enter your text below:")
elif option == 'Upload .txt File':
    uploaded_file = st.file_uploader("Upload a .txt file", type=['txt'])
    if uploaded_file is not None:
        text = file_text(uploaded_file)
elif option == 'Upload .pdf File':
    uploaded_file = st.file_uploader("Upload a .pdf file", type=['pdf'])
    if uploaded_file is not None:
        text = pdfReader(uploaded_file)
elif option == 'Wikipedia URL':
    url = st.text_input("Enter Wikipedia URL")
    if url:
        try:
            text = wiki_text(url)
        except:
            st.error("Failed to fetch article. Check the URL.")

if text:
    st.subheader("Original Text")
    st.write(text[:1000] + "..." if len(text) > 1000 else text)
    
    original_words = [w for w in text.split() if w.isalnum()]
    num_words_in_original = len(original_words)
    
    doc = nlp(text)
    sentences = list(doc.sents)
    total_sentences = len(sentences)

    freq_matrix_ = frequency_matrix(sentences)
    tf_matrix_ = tf_matrix(freq_matrix_)
    word_counts = sentences_per_words(freq_matrix_)
    idf_matrix_ = idf_matrix(freq_matrix_, word_counts, total_sentences)
    tfidf_matrix = tf_idf_matrix(tf_matrix_, idf_matrix_)
    sentence_scores = score_sentences(tfidf_matrix)
    avg_score = average_score(sentence_scores)

    threshold = 1.3 * avg_score
    summary = create_summary(sentences, sentence_scores, threshold)
    
    st.subheader("🔍 Summary")
    st.success(summary if summary else "Summary is empty. Try reducing threshold or checking input quality.")
    
    st.subheader("📊 Stats")
    st.markdown(f"- Total words in original text: **{num_words_in_original}**")
    st.markdown(f"- Total words in summary: **{len(summary.split())}**")
