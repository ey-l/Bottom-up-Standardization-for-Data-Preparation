import numpy as np
import pandas as pd
import tqdm.notebook as tqdm
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
tqdm.tqdm_notebook.pandas()
pd.set_option('display.max_colwidth', None)
STOP_WORDS = stopwords.words('english')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
PUNCTUATIONS = '"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def clean_text(text):
    text = re.sub('^RT[\\s]+', '', text)
    text = re.sub('https?://[^\\s\\n\\r]+', '', text)
    text = re.sub('#', '', text)
    return text

def remove_stop_words_and_puntuation(tokenized_text):
    text_clean = []
    for word in tokenized_text:
        if word not in PUNCTUATIONS and word not in STOP_WORDS:
            text_clean.append(word)
    return text_clean

def stemm_text(tokenized_text):
    text_stemm = []
    stemmer = PorterStemmer()
    for word in tokenized_text:
        text_stemm.append(stemmer.stem(word))
    return text_stemm

def process_text(text):
    text = clean_text(text)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    text = tokenizer.tokenize(text)
    text = remove_stop_words_and_puntuation(text)
    text = stemm_text(text)
    return text

def create_frequency_map(data: pd.DataFrame):
    data_dict = data.to_dict()
    frequency_map = {}
    for idx in tqdm.tqdm_notebook(range(len(data_dict['id']))):
        target = data_dict['target'][idx]
        words = data_dict['text'][idx].split(' ')
        for word in words:
            target = data_dict['target'][idx]
            if (word.lower(), target) in frequency_map:
                frequency_map[word.lower(), target] += 1
            else:
                frequency_map[word.lower(), target] = 1
    return frequency_map

def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(x)
    J = 0
    for i in tqdm.tqdm_notebook(range(0, num_iters)):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1 / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        theta = theta - alpha / m * np.dot(x.T, h - y)
    J = float(J[0])
    return (J, theta)

def extract_features(text, freqs):
    word_l = process_text(text)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        if (word, 1.0) in freqs:
            x[0, 1] += freqs[word, 1.0]
        if (word, 0.0) in freqs:
            x[0, 2] += freqs[word, 0.0]
    return x

def predict_text(text, freqs, theta):
    x = extract_features(text, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    if y_pred[0] > 0.5:
        return 1
    else:
        return 0
FREQUENCY_MAP = create_frequency_map(_input1)
_input1.head()
X = np.zeros((len(_input1), 3))
for idx in range(len(X)):
    X[idx, :] = extract_features(_input1['text'][idx], FREQUENCY_MAP)
y = [[n] for n in _input1['target'].to_numpy()]
y = np.array(y)
(J, theta) = gradientDescent(X, y, np.zeros((3, 1)), 1e-09, 100000)
J
submission = {'id': [], 'target': []}
for n in tqdm.tqdm_notebook(range(len(_input0))):
    row = _input0.loc[n, :]
    submission['id'].append(row['id'])
    submission['target'].append(predict_text(row['text'], FREQUENCY_MAP, theta))
df_submission = pd.DataFrame(submission)
df_submission