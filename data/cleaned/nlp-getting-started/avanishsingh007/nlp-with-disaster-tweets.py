import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import codecs
path = 'data/input/nlp-getting-started/train.csv'
with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
    train_df = pd.read_csv(f)
train_df[0:2]
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import nltk
train_df.drop(['keyword'], axis=1, inplace=True)
train_df.drop(['location'], axis=1, inplace=True)
train_df.head(5)
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
tokenizer = RegexpTokenizer('\\w+')
ps = PorterStemmer()
en_stop = set(stopwords.words('english'))

def getCleanedText(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stop]
    stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
    clean_text = ' '.join(stemmed_tokens)
    return clean_text
train_df['text'] = train_df['text'].apply(getCleanedText)
train_df['text'].head(5)

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity
train_df['Subjectivity'] = train_df['text'].apply(getSubjectivity)
train_df['Polarity'] = train_df['text'].apply(getPolarity)
train_df

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
train_df['Analysis'] = train_df['Polarity'].apply(getAnalysis)
train_df
train_df['Analysis'].value_counts()
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
train_df['Analysis'].value_counts().plot(kind='bar')

import codecs
path = 'data/input/nlp-getting-started/test.csv'
with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
    test_df = pd.read_csv(f)
test_df[0:2]
test_df.drop(['keyword'], axis=1, inplace=True)
test_df.drop(['location'], axis=1, inplace=True)
test_df['text'] = test_df['text'].apply(getCleanedText)
test_df['text'].head(5)
test_df['Subjectivity'] = test_df['text'].apply(getSubjectivity)
test_df['Polarity'] = test_df['text'].apply(getPolarity)
test_df['Analysis'] = test_df['Polarity'].apply(getAnalysis)
test_df
frames = [train_df, test_df]
new_df = pd.concat(frames)
new_df
new_df.isna().sum()
new_df = new_df.fillna(1)
new_df.info
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1, 2))
X_cv = cv.fit_transform(new_df['text']).toarray()
X = X_cv
y = new_df['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=200)
RF = RandomForestClassifier()