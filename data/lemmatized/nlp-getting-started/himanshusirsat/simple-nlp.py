import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
import nltk.corpus
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1 = _input1.drop(['keyword', 'location'], axis=1)
_input1
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input0 = _input0.drop(['keyword', 'location'], axis=1)
_input0
data = pd.DataFrame(_input1.text, columns=['text'])
data2 = pd.DataFrame(_input0.text, columns=['text'])
Data = pd.concat([data, data2], axis=0, ignore_index=True)
import re
import string

def remove_URL(text):
    url = re.compile('https?://\\S+|www\\.\\S+')
    return url.sub('', text)

def remove_html(text):
    html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)
Data['text'] = Data['text'].apply(lambda x: remove_URL(x))
Data['text'] = Data['text'].apply(lambda x: remove_html(x))
for i in range(0, 10876):
    Data['text'][i] = Data['text'][i].lower()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(stopwords.words('english'))
stop_wrds = np.array(stop_words)
import gensim
Data['tokenized_text'] = Data['text'].apply(gensim.utils.simple_preprocess)
Data

def remove_stp(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stop_words]
    return txt_clean
Data['no_stpword'] = Data['tokenized_text'].apply(lambda x: remove_stp(x))
Data
final = [0] * 10876
for i in range(0, 10876):
    str = ' '
    re = Data['no_stpword'][i]
    final[i] = str.join(re)
Data['final_txt'] = final
Data
Data.text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
vector = cv.fit_transform(Data['final_txt'])
X = vector[0:7613]
y = _input1.target
rf = MultinomialNB()