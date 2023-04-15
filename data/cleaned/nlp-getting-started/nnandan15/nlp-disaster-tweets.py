import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sample_submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
train.drop(['keyword', 'location'], axis=1, inplace=True)
test.drop(['keyword', 'location'], axis=1, inplace=True)
train.head()
test.head()
import string
train['text'] = train['text'].str.lower()
test['text'] = test['text'].str.lower()
text = train['text']
text1 = test['text']

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
text_clean = text.apply(lambda text: remove_punctuation(text))
text_clean1 = text1.apply(lambda text1: remove_punctuation(text1))
text_clean.head()
text_clean1.head()
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def stopwords_(text):
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])
text_clean = text_clean.apply(lambda text: stopwords_(text))
text_clean1 = text_clean1.apply(lambda text1: stopwords_(text1))
text_clean.head()
text_clean1.head()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemma(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
text_clean = text_clean.apply(lambda text: lemma(text))
text_clean1 = text_clean1.apply(lambda text1: lemma(text1))

def remove_URL(text):
    url = re.compile('https?://\\S+|www\\.\\S+')
    return url.sub('', text)
import re
text_clean = text_clean.apply(lambda x: remove_URL(x))
text_clean1 = text_clean1.apply(lambda x: remove_URL(x))

def remove_html(text):
    html = re.compile('<.*?>')
    return html.sub('', text)
text_clean = text_clean.apply(lambda x: remove_html(x))
text_clean1 = text_clean1.apply(lambda x: remove_html(x))
text_clean.head()
text_clean1.head()
from wordcloud import WordCloud
all_words = ' '.join([text for text in text_clean])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(16, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

df = pd.DataFrame({'text': text_clean})
df.head()
train.update(df)
train.head()
df1 = pd.DataFrame({'text': text_clean1})
df1.head()
test.update(df1)
test.drop('id', axis=1, inplace=True)
test.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_all = pd.concat([train['text'], test['text']])
tfidf = TfidfVectorizer(stop_words='english')