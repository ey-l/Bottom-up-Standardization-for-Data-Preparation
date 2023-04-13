import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col=0)
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col=0)
_input1 = _input1.fillna('', inplace=False)
_input0 = _input0.fillna('', inplace=False)
_input1.head()
_input0.head()
finaldata = _input1['keyword'] + ' ' + _input1['location'] + ' ' + _input1['text']
testdata = _input0['keyword'] + ' ' + _input0['location'] + ' ' + _input0['text']
finaltarget = _input1['target']
finaldata.head()
testdata.head()
finaltarget.head()
WNL = WordNetLemmatizer()

def text_process(data):
    msg = [c for c in _input1 if c not in string.punctuation]
    msg = ''.join(msg)
    msg = [word for word in msg.split() if word.lower() not in stopwords.words('english')]
    msg = [WNL.lemmatize(word) for word in msg]
    return msg
finaldata.head(10).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
CV = CountVectorizer(analyzer=text_process)