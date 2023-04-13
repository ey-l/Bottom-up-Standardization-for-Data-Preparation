import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import pandas as pd
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1 = _input1.drop(['keyword', 'location'], axis=1, inplace=False)
_input0 = _input0.drop(['keyword', 'location'], axis=1, inplace=False)
print(_input1.head())
print(_input0.head())
import string
_input1['text'] = _input1['text'].str.lower()
_input0['text'] = _input0['text'].str.lower()
_input1.head()
textData = _input1['text']
text1Data = _input0['text']
textData.head()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
text_clean = textData.apply(lambda text: remove_punctuation(text))
text_clean1 = text1Data.apply(lambda text1: remove_punctuation(text1))
text_clean.head()
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def stopwords_(text):
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])
text_clean = text_clean.apply(lambda text: stopwords_(text))
text_clean1 = text_clean1.apply(lambda text1: stopwords_(text1))
text_clean.head()
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
from wordcloud import WordCloud
all_words = ' '.join([text for text in text_clean])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(16, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
df = pd.DataFrame({'text': text_clean})
df.head()
df.head()
_input1.update(df)
df1 = pd.DataFrame({'text': text_clean1})
df1.head()
_input0.update(df1)
_input0.head()
testData_text = _input0.drop('id', axis=1)
testData_text.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_all = pd.concat([_input1['text'], _input0['text']])
tfidf = TfidfVectorizer(stop_words='english')