import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import string
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
stop = set(stopwords.words('english'))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
df = pd.concat([train, test], sort=False)
df.shape

def remove_URL(text):
    url = re.compile('https?://\\S+|www\\.\\S+')
    return url.sub('', text)

def remove_html(text):
    html = re.compile('<.*?>')
    return html.sub('', text)

def remove_emoji(text):
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)
df['text'] = df['text'].apply(lambda x: remove_URL(x))
df['text'] = df['text'].apply(lambda x: remove_html(x))
df['text'] = df['text'].apply(lambda x: remove_emoji(x))
df['text'] = df['text'].apply(lambda x: remove_punct(x))
df.head()
df_train = df[df['target'].notnull()]
df_train.head()
df_test = df[df['target'].isnull()]
df_test.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(df_train['text'], df_train['target'], test_size=0.3, random_state=101)
count_vectorizer = feature_extraction.text.CountVectorizer(analyzer='word', max_features=5000)