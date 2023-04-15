import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.sample(5)
df.shape
df.info()
df.isna().sum() / df.shape[0] * 100
df.drop(columns=['id', 'location'], inplace=True)
df.dropna(inplace=True)
df.shape
df.duplicated().sum()
df.drop_duplicates(keep='first', inplace=True)
df['content'] = df['keyword'] + ' ' + df['text']
df.head()
df.iloc[2, 3]
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import re

def text_preprocessing(text):
    text = text.lower()
    text = re.sub('(?i)\\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:\'".,<>?«»“”‘’]))', ' ', text)
    tokenize_text = nltk.word_tokenize(text)
    text_without_stopwords = [i for i in tokenize_text if i not in stopwords.words('english')]
    text_without_punc = [i for i in text_without_stopwords if i not in string.punctuation]
    transformed_text = [ps.stem(i) for i in text_without_punc if i.isalnum() == True]
    return ' '.join(transformed_text)
df['transformed_content'] = df['content'].apply(text_preprocessing)
final_df = df.drop(['text', 'keyword', 'content'], axis=1)
from wordcloud import WordCloud
wc = WordCloud(background_color='white', min_font_size=10, width=500, height=500)
true_news_wc = wc.generate(final_df[final_df['target'] == 0]['transformed_content'].str.cat(sep=' '))
plt.figure(figsize=(8, 6))
plt.imshow(true_news_wc)

fake_news_wc = wc.generate(final_df[final_df['target'] == 1]['transformed_content'].str.cat(sep=' '))
plt.figure(figsize=(8, 6))
plt.imshow(fake_news_wc)

from collections import Counter
true_news_words_list = final_df[final_df['target'] == 0]['transformed_content'].str.cat(sep=' ').split()
true_news_words_df = pd.DataFrame(Counter(true_news_words_list).most_common(20))
sns.barplot(x=true_news_words_df[0], y=true_news_words_df[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('True News Words Count')

fake_news_words_list = final_df[final_df['target'] == 1]['transformed_content'].str.cat(sep=' ').split()
fake_news_words_df = pd.DataFrame(Counter(fake_news_words_list).most_common(20))
sns.barplot(x=fake_news_words_df[0], y=fake_news_words_df[1])
plt.xticks(rotation='vertical')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('Fake News Words Count')

X = final_df['transformed_content'].values
y = final_df['target'].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cf = CountVectorizer(max_features=5000)
X_trf = cf.fit_transform(X).toarray()
X_train = cf.fit_transform(X_train).toarray()
X_test = cf.transform(X_test).toarray()
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

def check_model(X, y):
    algos = {'lgr': {'model': LogisticRegression(), 'params': {'C': [0.1, 0.01, 1, 0.5, 2, 10, 20]}}, 'mnb': {'model': MultinomialNB(), 'params': {}}, 'bnb': {'model': BernoulliNB(), 'params': {}}, 'gnb': {'model': GaussianNB(), 'params': {}}}
    score = []
    for (model_name, config) in algos.items():
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        gd = GridSearchCV(estimator=config['model'], param_grid=config['params'], n_jobs=-1, cv=cv)