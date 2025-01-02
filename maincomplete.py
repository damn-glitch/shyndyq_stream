import streamlit as st
import collections as coll
import math
import pickle
import string
import matplotlib.pyplot as plt
import nltk
import numpy as np
from PyPDF2 import PdfReader
from matplotlib import style
from nltk.corpus import cmudict, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nltk.download('cmudict')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')  # Ensure the punkt tokenizer is available

style.use("ggplot")
cmuDictionary = cmudict.dict()


def slidingWindow(sequence, winSize, step=1):
    sequence = sent_tokenize(sequence)
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)
    return [" ".join(sequence[i:i + winSize]) for i in range(0, numOfChunks * step, step)], sequence


def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''.join(page.extract_text() for page in pdf_reader.pages)
    return text


def syllable_count(word):
    global cmuDictionary
    d = cmuDictionary
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syl = syllable_count_manual(word)
    return syl


def syllable_count_manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count


def avg_word_length(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    return np.mean([len(word) for word in words])


def avg_sent_length_by_char(text):
    tokens = sent_tokenize(text)
    return np.mean([len(token) for token in tokens])


def avg_sent_length_by_word(text):
    tokens = sent_tokenize(text)
    return np.mean([len(token.split()) for token in tokens])


def avg_syllable_per_word(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    syllables = [syllable_count(word) for word in words]
    return np.mean(syllables)


def count_special_character(text):
    special_chars = "#$%&()*+-/<=>@[]^_`{|}~\t\n"
    return sum(1 for char in text if char in special_chars) / len(text)


def count_punctuation(text):
    punctuation = ",.!'\";?:"
    return sum(1 for char in text if char in punctuation) / len(text)


def feature_extraction(text, winSize, step):
    chunks, full_text = slidingWindow(text, winSize, step)
    features = []
    for chunk in chunks:
        feature = [
            avg_word_length(chunk),
            avg_sent_length_by_char(chunk),
            avg_sent_length_by_word(chunk),
            avg_syllable_per_word(chunk),
            count_special_character(chunk),
            count_punctuation(chunk),
        ]
        features.append(feature)
    return features, chunks, full_text


def elbow_method(data):
    distorsions = []
    for k in range(1, min(10, len(data))):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distorsions.append(kmeans.inertia_)
    plt.figure(figsize=(15, 5))
    plt.plot(range(1, min(10, len(data))), distorsions, 'bo-')
    plt.grid(True)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.title('Elbow Curve')
    st.pyplot(plt)


def analysis(vector, chunks, K=2):
    arr = np.array(vector)
    sc = StandardScaler()
    x = sc.fit_transform(arr)
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=min(K, len(components)))
    kmeans.fit_transform(components)
    labels = kmeans.labels_
    percentage_of_others = (np.sum(labels == 1) / len(labels)) * 100
    centers = kmeans.cluster_centers_
    colors = ["r.", "g.", "b.", "y.", "c."]

    plt.figure()
    for i in range(len(components)):
        plt.plot(components[i][0], components[i][1], colors[labels[i]], markersize=10)
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.title("Styles Clusters")
    st.pyplot(plt)

    return percentage_of_others, labels, chunks


def highlight_text(full_text, labels, chunks):
    highlighted_text = ""
    chunk_start = 0
    for i, chunk in enumerate(chunks):
        chunk_end = chunk_start + len(chunk)
        color = "red" if labels[i] == 1 else "black"
        highlighted_text += f"<span style='color:{color}'>{full_text[chunk_start:chunk_end]}</span> "
        chunk_start = chunk_end
    return highlighted_text


st.title("Document Plagiarism Checker")
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.pdf'):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.write("Original Document:")
    st.write(text)

    with st.spinner('Analyzing the document...'):
        vector, chunks, full_text = feature_extraction(text, winSize=10, step=10)
        if len(vector) == 0:
            st.error("The document does not contain enough text for analysis.")
        else:
            st.write("Feature vector shape:", np.array(vector).shape)
            elbow_method(vector)
            result, labels, chunks = analysis(vector, chunks, K=min(3, len(vector)))
            highlighted_text = highlight_text(full_text, labels, chunks)

            st.markdown(f"### Analysis complete! Percentage of plagiarized work: {result:.2f}%")
            st.markdown(highlighted_text, unsafe_allow_html=True)
