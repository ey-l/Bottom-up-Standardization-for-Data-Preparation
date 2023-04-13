import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
from nltk.stem.porter import PorterStemmer
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head(10)
_input1.info()
print('Shape of Data is: ', _input1.shape)
_input1.drop(['id', 'keyword', 'location'], axis=1)

def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str
_input1['text'] = _input1['text'].apply(punctuation_removal)
import re

def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\\d', i):
            list_text_new.append(i)
    return ''.join(list_text_new)
_input1['text'] = _input1['text'].apply(drop_numbers)
_input1['text'].head()
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(_input1['text'])
sum_words = words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for (word, idx) in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
wordcloud = WordCloud(background_color='black', width=2000, height=2000).generate_from_frequencies(dict(words_freq))
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(wordcloud)
plt.title('Vocabulary from text Reviews', fontsize=20)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
df = cv.fit_transform(_input1['text']).toarray()
x = df[:, [0]]
y = _input1['target'].values
print(x.shape)
print(y.shape)
print('Shape of X:', len(x))
print('Shape of Y:', len(y))
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion='entropy', random_state=1000)