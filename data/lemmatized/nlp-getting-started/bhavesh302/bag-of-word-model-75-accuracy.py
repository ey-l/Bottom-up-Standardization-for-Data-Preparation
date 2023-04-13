import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_text_len = [len(row.split()) for row in _input1['text']]
test_text_len = [len(row.split()) for row in _input0['text']]
train_text_len = np.array(train_text_len) / max(train_text_len)
test_text_len = np.array(test_text_len) / max(test_text_len)
print('Train_data_dim:{}'.format(_input1.shape))
print('Test data_dim:{}'.format(_input0.shape))
_input1.head()
_input0.head()
sns.countplot(x='target', data=_input1)
target = _input1['target']
_input1 = _input1['text']
_input1
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string

def text_preprocessor(text):
    stop_free_text = ' '.join([word.lower() for word in word_tokenize(text) if word not in stop_words])
    digit_free_text = res = ''.join(filter(lambda x: not x.isdigit(), stop_free_text))
    punct_free_text = ' '.join([word for word in word_tokenize(digit_free_text) if word not in list(string.punctuation)])
    lemmatised_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(punct_free_text)])
    return lemmatised_text
train_df = pd.DataFrame([])
processed_text = []
for text_data in _input1:
    processed_text.append(text_preprocessor(text_data))
train_df['text'] = processed_text
train_df['text_len'] = train_text_len
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(train_df, target, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=500, min_df=1)
train_X = vectorizer.fit_transform(X_train['text'].tolist())
test_X = vectorizer.transform(X_test['text'].tolist())
train_X = pd.DataFrame(train_X.toarray())
test_X = pd.DataFrame(test_X.toarray())
train_X['text_len'] = list(X_train['text_len'])
test_X['text_len'] = list(X_test['text_len'])
print(vectorizer.get_feature_names())
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
model = MultinomialNB()
model = RandomForestClassifier(random_state=42)