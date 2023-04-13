import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
ax = sns.countplot(x='target', data=_input1)
sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z?.!,¿]+', ' ', text)
    text = re.sub('http\\S+', '', text)
    html = re.compile('<.*?>')
    text = html.sub('', text)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    return text
_input1['text'] = _input1['text'].apply(lambda x: clean_text(x))
_input1.head()
sample_corpora = _input1['text'].iloc[:2].values
sample_corpora
count_vectorizer = CountVectorizer()
wm = count_vectorizer.fit_transform(sample_corpora)
doc_names = ['Doc{:d}'.format(idx) for (idx, _) in enumerate(wm)]
feat_names = count_vectorizer.get_feature_names()
sample_df = pd.DataFrame(data=wm.toarray(), index=doc_names, columns=feat_names)
sample_df
(X_train, X_test, y_train, y_test) = train_test_split(_input1['text'].values, _input1['target'].values, test_size=0.2, random_state=123, stratify=_input1['target'].values)
tfidf_vectorizer = TfidfVectorizer()
tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_vectors = tfidf_vectorizer.transform(X_test)
classifier = RandomForestClassifier()