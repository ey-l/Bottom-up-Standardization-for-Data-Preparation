import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
import en_core_web_sm
nlp = en_core_web_sm.load()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1.head(10)
_input0.head(10)
_input1['text'][152]
print('Shape of Training data:-', _input1.shape)
print('Shape of Test data:-', _input0.shape)
_input1.isnull().sum()
_input0.isnull().sum()
_input1['target'].value_counts()
_input1.describe()
plt.style.use('ggplot')
target_counts = _input1.target.value_counts()
sns.barplot(y=target_counts, x=target_counts.index)
plt.title('Counting the values in target column')
plt.ylabel('Sample')
plt.xlabel('Target')
my_labels = ['Non-Disaster', 'Disaster']
my_color = ['Blue', 'Green']
plt.figure(figsize=(15, 7))
plt.pie(_input1['target'].value_counts(), labels=my_labels, colors=my_color, autopct='%1.1f%%')
plt.legend()
my_disaster_tweets = _input1[_input1['target'] == 1]['text']
my_disaster_tweets[:10]
non_disaster_tweets = _input1[_input1['target'] == 0]['text']
non_disaster_tweets[:10]
plt.figure(figsize=(15, 10))
wc = WordCloud(max_words=500, background_color='White', width=1000, height=500, stopwords=STOPWORDS).generate(' '.join(_input1[_input1.target == 1].text))
plt.imshow(wc, interpolation='bilinear')
plt.figure(figsize=(15, 10))
wc = WordCloud(max_words=500, background_color='White', width=1000, height=500, stopwords=STOPWORDS).generate(' '.join(_input1[_input1.target == 0].text))
plt.imshow(wc, interpolation='bilinear')
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 5))
char_len = _input1[_input1['target'] == 1]['text'].str.len()
ax1.hist(char_len, color='#db680f', edgecolor='black')
ax1.set_title('Disaster Tweets')
char_len2 = _input1[_input1['target'] == 0]['text'].str.len()
ax2.hist(char_len2, color='#03639e', edgecolor='black')
ax2.set_title('Non-Disater Tweets')
plt.suptitle('Length of Characters in text', fontsize=20)
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 5))
char_len = _input1[_input1['target'] == 1]['text'].str.split().map(lambda x: len(x))
ax1.hist(char_len, color='#c40a0d', edgecolor='black')
ax1.set_title('Disaster Tweets')
char_len2 = _input1[_input1['target'] == 0]['text'].str.split().map(lambda x: len(x))
ax2.hist(char_len2, color='#0893a6', edgecolor='black')
ax2.set_title('Non-Disater Tweets')
plt.suptitle('Length of words in text', fontsize=20)
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 5))
char_len_dis = _input1[_input1['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.distplot(char_len_dis.map(lambda x: np.mean(x)), ax=ax1, color='green')
ax1.set_title('Disaster Tweets')
char_len_ndis = _input1[_input1['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.distplot(char_len_ndis.map(lambda x: np.mean(x)), ax=ax2, color='red')
ax2.set_title('Non-Disaster Tweets')
plt.suptitle('Average word counts', fontsize=20)

def sample_corpus(target):
    corpus = []
    for x in _input1[_input1['target'] == target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus
from collections import defaultdict

def stopwords_analysis(data, func, target):
    value_list = []
    for labels in range(0, len(target)):
        dic = defaultdict(int)
        corpus = func(target[labels])
        for words in corpus:
            dic[words] += 1
        top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:20]
        (x_items, y_values) = zip(*top)
        value_list.append(x_items)
        value_list.append(y_values)
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.barh(value_list[0], value_list[1], color='b')
    ax1.set_title('Non-Disaster Tweets')
    ax2.barh(value_list[2], value_list[3], color='red')
    ax2.set_title('Disaster Tweets')
    plt.suptitle('Top Stop words in text')
stopwords_analysis(_input1, sample_corpus, [0, 1])
import string

def punctuation_analysis(data, func, target):
    values_list = []
    special = string.punctuation
    for labels in range(0, len(target)):
        dic = defaultdict(int)
        corpus = func(target[labels])
        for i in corpus:
            if i in special:
                dic[i] += 1
        (x_items, y_values) = zip(*dic.items())
        values_list.append(x_items)
        values_list.append(y_values)
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.bar(values_list[0], values_list[1], color='b', linewidth=1.2)
    ax1.set_title('Non-Disaster Tweets')
    ax2.bar(values_list[2], values_list[3], color='red', edgecolor='black', linewidth=1.2)
    ax2.set_title('Disaster Tweets')
    plt.suptitle('Punctuations in text')
punctuation_analysis(_input1, sample_corpus, [0, 1])
missing_train = _input1.isnull().sum()
missing_test = _input0.isnull().sum()
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
missing_train = missing_train[missing_train > 0].sort_values()
ax1.pie(missing_train, autopct='%1.1f%%', startangle=30, explode=[0.9, 0], labels=['keyword', 'location'], colors=['red', '#afe84d'])
ax1.set_title('Null values present in Train Dataset')
missing_test = missing_test[missing_test > 0].sort_values()
ax2.pie(missing_test, autopct='%1.1f%%', startangle=30, explode=[0.9, 0], labels=['keyword', 'location'], colors=['Red', '#6c1985'])
ax2.set_title('Null values present in Test Dataset')
plt.suptitle('Distribution of Null Values in Dataset')
plt.tight_layout()
stop_words = nltk.corpus.stopwords.words('english')
i = 0
import contractions
from nltk.stem import SnowballStemmer
nltk.download('wordnet')
nltk.download('punkt')
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
for doc in _input1.text:
    doc = re.sub('https?://\\S+|www\\.\\S+', '', doc)
    doc = re.sub('<.*?>', '', doc)
    doc = re.sub('[^a-zA-Z\\s]', '', doc, re.I | re.A)
    doc = ' '.join([wnl.lemmatize(i) for i in doc.lower().split()])
    doc = contractions.fix(doc)
    tokens = nltk.word_tokenize(doc)
    filtered = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered)
    _input1.text[i] = doc
    i += 1
i = 0
for doc in _input0.text:
    doc = re.sub('https?://\\S+|www\\.\\S+', '', doc)
    doc = re.sub('<.*?>', '', doc)
    doc = re.sub('[^a-zA-Z\\s]', '', doc, re.I | re.A)
    doc = ' '.join([wnl.lemmatize(i) for i in doc.lower().split()])
    doc = contractions.fix(doc)
    tokens = nltk.word_tokenize(doc)
    filtered = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered)
    _input0.text[i] = doc
    i += 1
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1, 1))
cv_matrix = cv.fit_transform(_input1.text).toarray()
train_df = pd.DataFrame(cv_matrix, columns=cv.get_feature_names())
test_df = pd.DataFrame(cv.transform(_input0.text).toarray(), columns=cv.get_feature_names())
train_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 1), use_idf=True)
mat = tfidf.fit_transform(_input1.text).toarray()
train_df = pd.DataFrame(mat, columns=tfidf.get_feature_names())
test_df = pd.DataFrame(tfidf.transform(_input0.text).toarray(), columns=tfidf.get_feature_names())
train_df.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
model = LogisticRegression()