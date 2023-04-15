import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
data = pd.read_csv('data/input/nlp-getting-started/train.csv', index_col=0)
test = pd.read_csv('data/input/nlp-getting-started/test.csv', index_col=0)
data.fillna('', inplace=True)
test.fillna('', inplace=True)
data.head()
test.head()
finaldata = data['keyword'] + ' ' + data['location'] + ' ' + data['text']
testdata = test['keyword'] + ' ' + test['location'] + ' ' + test['text']
finaltarget = data['target']
finaldata.head()
testdata.head()
finaltarget.head()
WNL = WordNetLemmatizer()

def text_process(data):
    msg = [c for c in data if c not in string.punctuation]
    msg = ''.join(msg)
    msg = [word for word in msg.split() if word.lower() not in stopwords.words('english')]
    msg = [WNL.lemmatize(word) for word in msg]
    return msg
finaldata.head(10).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
CV = CountVectorizer(analyzer=text_process)