import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head(10)
_input1.count()
_input0.count()
from nltk.corpus import stopwords
import re
import string

def change_contraction_verb(text):
    text = re.sub("n\\'t", ' not', text)
    text = re.sub("\\'re", ' are', text)
    text = re.sub("\\'s", ' is', text)
    text = re.sub("\\'d", ' would', text)
    text = re.sub("\\'ll", ' will', text)
    text = re.sub("\\'t", ' not', text)
    text = re.sub("\\'ve", ' have', text)
    text = re.sub("\\'m", ' am', text)
    text = re.sub("won\\'t", 'will not', text)
    text = re.sub("can\\'t", 'can not', text)
    return text
_input1['text'] = _input1['text'].apply(lambda x: change_contraction_verb(x))
_input0['text'] = _input0['text'].apply(lambda x: change_contraction_verb(x))
_input1['text'].head(10)

def custom_preprocessor(text):
    """
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    """
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text
_input1['text'] = _input1['text'].apply(lambda x: custom_preprocessor(x))
_input0['text'] = _input0['text'].apply(lambda x: custom_preprocessor(x))
_input1['text'].head(10)

def remove_emoji(text):
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    return emoji_pattern.sub('', text)
_input1['text'] = _input1['text'].apply(lambda x: remove_emoji(x))
_input0['text'] = _input0['text'].apply(lambda x: remove_emoji(x))
_input1.head(10)
_input0.head(10)
from sklearn.feature_extraction.text import CountVectorizer
stopwords = stopwords.words('english')
print(stopwords)
count_vectorizer = CountVectorizer(token_pattern='\\w{1,}', ngram_range=(1, 2), stop_words=stopwords)
train_vector = count_vectorizer.fit_transform(_input1['text'])
test_vector = count_vectorizer.transform(_input0['text'])
train_vector.toarray()
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
clf = LogisticRegression()
scores = model_selection.cross_val_score(clf, train_vector, _input1['target'], cv=5, scoring='accuracy')
print(scores)
print('Accuracy of Model with Cross Validation is: ', scores.mean() * 100)