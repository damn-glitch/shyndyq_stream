import collections as coll
import math
import pickle
import string
import PyPDF2

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
from PyPDF2 import PdfReader
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import requests
import json
import pyfiglet

nltk.download('cmudict')
nltk.download('stopwords')

style.use("ggplot")
cmuDictionary = None
"""
Implementation way
1) Our system takes a document.
2) Divides it into chunks of 10 sentences.
3) Computes stylometric features for each chunk.
4) Then uses the elbow method on these vectors to identify the value of centroids K.
5) The value of K corresponds to the number of different writing styles the document had.
6) In order to visualize the clusters, PCA is used to convert the high dimensional features vector to a 2D one and then the chunks are plotted.
7) The chunks with same style are grouped under one centroid with same color, hence implying the number of writing styles implied in that document.

The heart of our approach is correctly extracting the style from the chunk which is successfully achieved using a mix of
different categories of linguistic features like lexical, vocabulary richness and readability scores.
Our method is repeated every time for a new document. Since tt identifies the different writing styles in that document,
hence our approach can also be used to detect plagiarism.
"""


# takes a paragraph of text and divides it into chunks of specified number of sentences
def slidingWindow(sequence, winSize, step=1):
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    sequence = sent_tokenize(sequence)

    # Pre-compute number of chunks to omit
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)

    l = []
    # Do the work
    for i in range(0, numOfChunks * step, step):
        l.append(" ".join(sequence[i:i + winSize]))

    return l

# ---------------------------------------------------------------------

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ''
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extractText()
    return text


# ---------------------------------------------------------------------

def syllable_count_Manual(word):
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


# ---------------------------------------------------------------------
# COUNTS NUMBER OF SYLLABLES

def syllable_count(word):
    global cmuDictionary
    d = cmuDictionary
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syl = syllable_count_Manual(word)
    return syl

    # ----------------------------------------------------------------------------


# removing stop words plus punctuation.
def Avg_wordLength(str):
    str.translate(string.punctuation)
    tokens = word_tokenize(str, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    return np.average([len(word) for word in words])


# ----------------------------------------------------------------------------


# returns avg number of characters in a sentence
def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


# ----------------------------------------------------------------------------

# returns avg number of words in a sentence
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])


# ----------------------------------------------------------------------------


# GIVES NUMBER OF SYLLABLES PER WORD
def Avg_Syllable_per_Word(text):
    tokens = word_tokenize(text, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    syllabls = [syllable_count(word) for word in words]
    p = (" ".join(words))
    return sum(syllabls) / max(1, len(words))


# -----------------------------------------------------------------------------

# COUNTS SPECIAL CHARACTERS NORMALIZED OVER LENGTH OF CHUNK
def CountSpecialCharacter(text):
    st = ["#", "$", "%", "&", "(", ")", "*", "+", "-", "/", "<", "=", '>',
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return count / len(text)


# ----------------------------------------------------------------------------

def CountPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return float(count) / float(len(text))


# ----------------------------------------------------------------------------
# RETURNS NORMALIZED COUNT OF FUNCTIONAL WORDS FROM A Framework for
# Authorship Identification of Online Messages: Writing-Style Features and Classification Techniques

def CountFunctionalWords(text):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    functional_words = functional_words.split()
    words = RemoveSpecialCHs(text)
    count = 0

    for i in text:
        if i in functional_words:
            count += 1

    return count / len(words)


# ---------------------------------------------------------------------------

# also returns Honore Measure R
def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
    h = V1 / N
    return R, h


# ---------------------------------------------------------------------------

def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    count = 0
    # Collections as coll Counter takes an iterable collapse duplicate and counts as
    # a dictionary how many equivelant items has been entered
    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))
    return S, h


# ---------------------------------------------------------------------------

# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


# --------------------------------------------------------------------------
# TYPE TOKEN RATIO NO OF DIFFERENT WORDS / NO OF WORDS
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)


# --------------------------------------------------------------------------
# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# we can convert into log because we are only comparing different texts
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    B = (V - a) / (math.log(N))
    return B


# ------------------------------------------------------------------------
def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


# -------------------------------------------------------------------------
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# -------------------------------------------------------------------------


# -1*sigma(pi*lnpi)
# Shannon and sympsons index are basically diversity indices for any community
def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
    return H


# ------------------------------------------------------------------
# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return D


# ------------------------------------------------------------------

def FleschReadingEase(text, NoOfsentences):
    words = RemoveSpecialCHs(text)
    l = float(len(words))
    scount = 0
    for word in words:
        scount += syllable_count(word)

    I = 206.835 - 1.015 * (l / float(NoOfsentences)) - 84.6 * (scount / float(l))
    return I


# -------------------------------------------------------------------
def FleschCincadeGradeLevel(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    scount = 0
    for word in words:
        scount += syllable_count(word)

    l = len(words)
    F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / float(l)) - 15.59
    return F


# -----------------------------------------------------------------
def dale_chall_readability_formula(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)
    try:
        with open('C:\\Users\\tairk\\OneDrive\\Desktop\\Shyndyq\\complete\\Code\\dale-chall.pkl', 'rb') as f:
            fimiliarWords = pickle.load(f)
            
        print(fimiliarWords)
    except pickle.UnpicklingError as e:
        print(f"Error loading pickled data: {e}")
        fimiliarWords = {}  # Use a default or empty object here
    for word in words:
        if word not in fimiliarWords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if percent > 5:
        adjusted = 3.6365
    D = 0.1579 * percent + 0.0496 * (NoOfWords / NoOfSentences) + adjusted
    return D


# ------------------------------------------------------------------
def GunningFoxIndex(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    NoOFWords = float(len(words))
    complexWords = 0
    for word in words:
        if (syllable_count(word) > 2):
            complexWords += 1

    G = 0.4 * ((NoOFWords / NoOfSentences) + 100 * (complexWords / NoOFWords))
    return G


def PrepareData(text1, text2, Winsize):
    chunks1 = slidingWindow(text1, Winsize, Winsize)
    chunks2 = slidingWindow(text2, Winsize, Winsize)
    return " ".join(str(chunk1) + str(chunk2) for chunk1, chunk2 in zip(chunks1, chunks2))


# ------------------------------------------------------------------

# returns a feature vector of text
def FeatureExtration(text, winSize, step):
    # cmu dictionary for syllables
    global cmuDictionary
    cmuDictionary = cmudict.dict()

    chunks = slidingWindow(text, winSize, step)
    vector = []
    for chunk in chunks:
        feature = []

        # LEXICAL FEATURES
        meanwl = (Avg_wordLength(chunk))
        feature.append(meanwl)

        meansl = (Avg_SentLenghtByCh(chunk))
        feature.append(meansl)

        mean = (Avg_SentLenghtByWord(chunk))
        feature.append(mean)

        meanSyllable = Avg_Syllable_per_Word(chunk)
        feature.append(meanSyllable)

        means = CountSpecialCharacter(chunk)
        feature.append(means)

        p = CountPuncuation(chunk)
        feature.append(p)

        f = CountFunctionalWords(text)
        feature.append(f)

        # VOCABULARY RICHNESS FEATURES

        TTratio = typeTokenRatio(chunk)
        feature.append(TTratio)

        HonoreMeasureR, hapax = hapaxLegemena(chunk)
        feature.append(hapax)
        feature.append(HonoreMeasureR)

        SichelesMeasureS, dihapax = hapaxDisLegemena(chunk)
        feature.append(dihapax)
        feature.append(SichelesMeasureS)

        YuleK = YulesCharacteristicK(chunk)
        feature.append(YuleK)

        S = SimpsonsIndex(chunk)
        feature.append(S)

        B = BrunetsMeasureW(chunk)
        feature.append(B)

        Shannon = ShannonEntropy(text)
        feature.append(Shannon)

        # READIBILTY FEATURES
        FR = FleschReadingEase(chunk, winSize)
        feature.append(FR)

        FC = FleschCincadeGradeLevel(chunk, winSize)
        feature.append(FC)

        # also quite a different
        D = dale_chall_readability_formula(chunk, winSize)
        feature.append(D)

        # quite a difference
        G = GunningFoxIndex(chunk, winSize)
        feature.append(G)

        vector.append(feature)

    return vector


# -----------------------------------------------------------------------------------------
# ELBOW METHOD
def ElbowMethod(data):
    X = data  # <your_data>
    distorsions = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(1, 10), distorsions, 'bo-')
    plt.grid(True)
    plt.ylabel("Square Root Error")
    plt.xlabel("Number of Clusters")
    plt.title('Elbow curve')
    plt.savefig("ElbowCurve.png")
    plt.show()


# -----------------------------------------------------------------------------------------
# ANALYSIS PART

# Using the graph shown in Elbow Method, find the appropriate value of K and set it here.
def Analysis(vector, K=2):
    arr = (np.array(vector))

    # mean normalization of the data . converting into normal distribution having mean=0 , -0.1<x<0.1
    sc = StandardScaler()
    x = sc.fit_transform(arr)

    # Breaking into principle components
    pca = PCA(n_components=2)
    components = (pca.fit_transform(x))
    # Applying kmeans algorithm for finding centroids

    kmeans = KMeans(n_clusters=K)
    kmeans.fit_transform(components)
    print("labels: ", kmeans.labels_)
    labels = kmeans.labels_
    total_count = len(labels)
    count_of_ones = np.count_nonzero(labels == 1)

    percentage_of_ones = (count_of_ones / total_count) * 100
    print(f"Percentage of written by other person: {percentage_of_ones:.2f}%")
    centers = kmeans.cluster_centers_

    # lables are assigned by the algorithm if 2 clusters then lables would be 0 or 1
    lables = kmeans.labels_
    colors = ["r.", "g.", "b.", "y.", "c."]
    colors = colors[:K + 1]

    for i in range(len(components)):
        plt.plot(components[i][0], components[i][1], colors[lables[i]], markersize=10)

    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150, linewidths=10, zorder=15)
    plt.xlabel("1st Principle Component")
    plt.ylabel("2nd Principle Component")
    title = "Styles Clusters"
    plt.title(title)
    plt.savefig("Results" + ".png")
    plt.show()
    
    return percentage_of_ones

app = Flask(__name__)

UPLOAD_FOLDER = 'C:\\Users\\tairk\\OneDrive\\Desktop\\Shyndyq\\complete\\Code\\'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_text():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            if filename.endswith('.pdf'):
                with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as pdf_file:
                    file_content = extract_text_from_pdf(pdf_file)
                pass
            else:
                with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r') as uploaded_file:
                    file_content = uploaded_file.read()
                
                vector = FeatureExtration(file_content, winSize=10, step=10)
                ElbowMethod(np.array(vector))
                analysis_result = Analysis(vector)
                
                return render_template('results.html', results=analysis_result)

if __name__ == '__main__':
    banner = pyfiglet.figlet_format("Shyndyq")
    print(banner)

    filename_1 = "C:\\Users\\tairk\\OneDrive\\Desktop\\Shyndyq\\complete\\Code\\DocumentWithVariantWritingStyles.txt"

    try:
        with open(filename_1, 'r', encoding='utf-8') as file:
            text_to_check = file.read()
    except FileNotFoundError:
        print(f"File '{filename_1}' not found. Please ensure the file exists.")

    try:
        vector = FeatureExtration(text_to_check, winSize=10, step=10)
        ElbowMethod(np.array(vector))
        Analysis(vector)
    except NameError:
        print("Functions 'FeatureExtration', 'ElbowMethod', or 'Analysis' are not defined.")
        
    # Run the Flask app
    app.run(debug=False)
