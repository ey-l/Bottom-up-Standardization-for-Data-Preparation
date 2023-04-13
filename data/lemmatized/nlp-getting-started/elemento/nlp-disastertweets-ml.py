import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import vstack
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1.info()
_input1 = _input1.drop_duplicates(subset=['text', 'target'], keep='first', inplace=False)
sep = _input1.shape[0]
_input1.info()
Y = _input1['target']
_input1 = _input1.drop(['target'], axis=1, inplace=False)
print(_input1.shape, Y.shape)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0.info()
df = pd.concat([_input1, _input0], axis=0)
df = df.drop(['location'], axis=1, inplace=False)
df.info()
df.head()

def decontracted(phrase):
    phrase = re.sub("won't", 'will not', phrase)
    phrase = re.sub("can\\'t", 'can not', phrase)
    phrase = re.sub("n\\'t", ' not', phrase)
    phrase = re.sub("\\'re", ' are', phrase)
    phrase = re.sub("\\'s", ' is', phrase)
    phrase = re.sub("\\'d", ' would', phrase)
    phrase = re.sub("\\'ll", ' will', phrase)
    phrase = re.sub("\\'t", ' not', phrase)
    phrase = re.sub("\\'ve", ' have', phrase)
    phrase = re.sub("\\'m", ' am', phrase)
    return phrase
stopwords = set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
pre_text = []
for sen in tqdm(df['text'].values):
    sen = decontracted(sen)
    sen = re.sub('\\S*\\d\\S*', '', sen).strip()
    sen = re.sub('[^A-Za-z]+', ' ', sen)
    sen = ' '.join((e.lower() for e in sen.split() if e.lower() not in stopwords))
    pre_text.append(sen.strip())
print(pre_text[10])
print(pre_text[20])
print(pre_text[30])
tfidf_vect = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=1, norm='l2', sublinear_tf=True)
fea_train = tfidf_vect.fit_transform(pre_text[:sep])
print(fea_train.shape)
fea_test = tfidf_vect.transform(pre_text[sep:])
print(fea_test.shape)
fea_text = vstack([fea_train, fea_test])
fea_text.todense()
print(type(fea_text), fea_text.shape)
df = df.drop(['keyword', 'text'], axis=1, inplace=False)
fea_text = pd.DataFrame.sparse.from_spmatrix(fea_text)
df_fea = pd.concat([df, fea_text.reindex(df.index)], axis=1)
print(df.shape, fea_text.shape, df_fea.shape)
df_fea.head()
_input1 = df_fea.iloc[:sep, :]
_input0 = df_fea.iloc[sep:, :]
print(_input1.columns)
id_train = _input1['id']
_input1 = _input1.drop(['id'], inplace=False, axis=1)
id_test = _input0['id']
_input0 = _input0.drop(['id'], inplace=False, axis=1)
print(_input1.shape, Y.shape, id_train.shape, _input0.shape, id_test.shape)
n = 1
skf = StratifiedKFold(n_splits=4)
for (train_index, val_index) in skf.split(_input1, Y):
    (X_train, X_val) = (_input1.iloc[train_index], _input1.iloc[val_index])
    (y_train, y_val) = (Y.iloc[train_index], Y.iloc[val_index])
    model = LogisticRegression(max_iter=1000, C=3)