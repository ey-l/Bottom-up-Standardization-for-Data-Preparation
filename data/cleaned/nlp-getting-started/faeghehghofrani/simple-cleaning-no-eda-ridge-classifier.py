import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.stem import PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
nltk.download('wordnet')
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('data/input/nlp-getting-started/test.csv')
train_df.head()

def Info_dataFrame(df):
    name = [x for x in globals() if globals()[x] is df][0]
    print('informaion of {}:'.format(name))
    print('--' * 20)
    print(df.info())
    print('==' * 20)
    print('informaion about count of Null in {}:'.format(name))
    print('--' * 20)
    print(df.isnull().sum())
    print('==' * 20)
Info_dataFrame(train_df)
Info_dataFrame(test_df)

def clean_text_Simple(df):
    for i in range(len(df)):
        sentence = sent_tokenize(df['text'][i])
        corpus = ''
        for sent in sentence:
            review = re.sub('[^a-zA-Z]', ' ', sent)
            review = review.lower()
            review = review.split()
            review = ' '.join(review)
            corpus += review
        df['text'][i] = corpus
    return df
train_df_m1 = clean_text_Simple(train_df)
train_df_m1
test_df_m1 = clean_text_Simple(test_df)
test_df_m1

def clean_text_Porter(df):
    for i in range(len(df)):
        sentence = sent_tokenize(df['text'][i])
        stemmer = PorterStemmer()
        corpus = ''
        for sent in sentence:
            review = re.sub('[^a-zA-Z]', ' ', sent)
            review = review.lower()
            review = review.split()
            review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus += review
        df['text'][i] = corpus
    return df
train_df_mp = clean_text_Porter(train_df)
train_df_mp
test_df_mp = clean_text_Porter(test_df)
test_df_mp

def split_train(df, split, model):
    X = df['text']
    y = df['target'].values
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train = split.fit_transform(X_train).toarray()
    X_test = split.transform(X_test).toarray()