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
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
df_sub = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
df_train.info()
df_train.drop_duplicates(subset=['text', 'target'], keep='first', inplace=True)
sep = df_train.shape[0]
df_train.info()
Y = df_train['target']
df_train.drop(['target'], axis=1, inplace=True)
print(df_train.shape, Y.shape)
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
df_test.info()
df = pd.concat([df_train, df_test], axis=0)
df.drop(['location'], axis=1, inplace=True)
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
df.drop(['keyword', 'text'], axis=1, inplace=True)
fea_text = pd.DataFrame.sparse.from_spmatrix(fea_text)
df_fea = pd.concat([df, fea_text.reindex(df.index)], axis=1)
print(df.shape, fea_text.shape, df_fea.shape)
df_fea.head()
df_train = df_fea.iloc[:sep, :]
df_test = df_fea.iloc[sep:, :]
print(df_train.columns)
id_train = df_train['id']
df_train.drop(['id'], inplace=True, axis=1)
id_test = df_test['id']
df_test.drop(['id'], inplace=True, axis=1)
print(df_train.shape, Y.shape, id_train.shape, df_test.shape, id_test.shape)
n = 1
skf = StratifiedKFold(n_splits=4)
for (train_index, val_index) in skf.split(df_train, Y):
    (X_train, X_val) = (df_train.iloc[train_index], df_train.iloc[val_index])
    (y_train, y_val) = (Y.iloc[train_index], Y.iloc[val_index])
    model = LogisticRegression(max_iter=1000, C=3)