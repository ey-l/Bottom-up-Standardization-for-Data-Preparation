import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1
_input1['location'].isnull().sum()
_input1['keyword'].value_counts
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def preprocessing(data):
    documents = []
    stem = WordNetLemmatizer()
    for i in range(len(data)):
        text = str(data['text'][i])
        text = text.lower()
        document = re.sub('\\W', ' ', text)
        document = re.sub('\\s+[a-zA-Z]\\s+', ' ', document)
        document = re.sub('\\^[a-zA-Z]\\s+', ' ', document)
        document = re.sub('\\s+', ' ', document, flags=re.I)
        document = document.split(' ')
        stemDocement = [stem.lemmatize(i) for i in document]
        document = ' '.join(document)
        documents.append(document)
    return documents
x = preprocessing(_input1)
x2 = preprocessing(_input0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
tfidfObject = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))