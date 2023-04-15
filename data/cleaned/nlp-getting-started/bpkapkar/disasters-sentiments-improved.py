import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import pandas as pd
sample_submission = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
testData = pd.read_csv('data/input/nlp-getting-started/test.csv')
trainData = pd.read_csv('data/input/nlp-getting-started/train.csv')
trainData.drop(['keyword', 'location'], axis=1, inplace=True)
testData.drop(['keyword', 'location'], axis=1, inplace=True)
print(trainData.head())
print(testData.head())
import string
trainData['text'] = trainData['text'].str.lower()
testData['text'] = testData['text'].str.lower()
trainData.head()
textData = trainData['text']
text1Data = testData['text']
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
trainData.update(df)
df1 = pd.DataFrame({'text': text_clean1})
df1.head()
testData.update(df1)
testData.head()
testData_text = testData.drop('id', axis=1)
testData_text.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_all = pd.concat([trainData['text'], testData['text']])
tfidf = TfidfVectorizer(stop_words='english')