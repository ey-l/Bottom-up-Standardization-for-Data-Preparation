import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train.head(10)
train.count()
test.count()
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
train['text'] = train['text'].apply(lambda x: change_contraction_verb(x))
test['text'] = test['text'].apply(lambda x: change_contraction_verb(x))
train['text'].head(10)

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
train['text'] = train['text'].apply(lambda x: custom_preprocessor(x))
test['text'] = test['text'].apply(lambda x: custom_preprocessor(x))
train['text'].head(10)

def remove_emoji(text):
    emoji_pattern = re.compile('[😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿✂-➰Ⓜ-🉑]+', flags=re.UNICODE)
    return emoji_pattern.sub('', text)
train['text'] = train['text'].apply(lambda x: remove_emoji(x))
test['text'] = test['text'].apply(lambda x: remove_emoji(x))
train.head(10)
test.head(10)
from sklearn.feature_extraction.text import CountVectorizer
stopwords = stopwords.words('english')
print(stopwords)
count_vectorizer = CountVectorizer(token_pattern='\\w{1,}', ngram_range=(1, 2), stop_words=stopwords)
train_vector = count_vectorizer.fit_transform(train['text'])
test_vector = count_vectorizer.transform(test['text'])
train_vector.toarray()
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
clf = LogisticRegression()
scores = model_selection.cross_val_score(clf, train_vector, train['target'], cv=5, scoring='accuracy')
print(scores)
print('Accuracy of Model with Cross Validation is: ', scores.mean() * 100)