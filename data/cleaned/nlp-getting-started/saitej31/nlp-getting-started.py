import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
train_data = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('data/input/nlp-getting-started/test.csv')
sample = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print(len(train_data))
print(len(test_data))
y = train_data.target
train_data = train_data.drop(['id', 'keyword', 'location', 'target'], axis=1)

def process_text(text):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    text = re.sub('\\$\\w*', '', text)
    text = re.sub('^RT[\\s]+', '', text)
    text = re.sub('https?:\\/\\/.*[\\r\\n]*', '', text)
    text = re.sub('#', '', text)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    texts_clean = []
    for word in text_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            texts_clean.append(stem_word)
    return texts_clean

def build_freqs(texts, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for (y, text) in zip(yslist, texts):
        for word in process_text(text):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
freqs = build_freqs(train_data['text'], y)
freqs

def extract_features(text, freqs):
    word_l = process_text(text)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1.0), 0)
        x[0, 2] += freqs.get((word, 0.0), 0)
    assert x.shape == (1, 3)
    return x
X = np.zeros((len(train_data), 3))
for i in range(len(train_data)):
    X[i, :] = extract_features(train_data.text[i], freqs)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score
y.value_counts()
len(X)
model = LogisticRegression(penalty='l2', random_state=42, max_iter=20, solver='liblinear', class_weight='balanced')