import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1.head()
_input1.info()
_input1.isnull().sum()
_input1.isnull().sum().plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Columns')
plt.title('Count of Missing Values')
(_input1.isnull().sum() / len(_input1) * 100).plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Columns')
plt.title('Percentage of Missing Values')
plt.figure(figsize=(6, 6))
_input1['target'].value_counts().plot.pie(explode=[0.05, 0.05], labels=['False Disaster', 'True Disaster'], shadow=True, autopct='%1.1f%%', textprops={'fontsize': 10})
_input1['keyword'].value_counts()[0:10]
plt.figure(figsize=(7, 5))
_input1['keyword'].value_counts(normalize=True)[0:10].plot(kind='bar')
plt.title('Top 10 Keywords')
plt.xlabel('Keywords')
plt.ylabel('%')
_input1['location'].value_counts()[0:10]
plt.figure(figsize=(7, 5))
_input1['location'].value_counts(normalize=True)[0:10].plot(kind='bar')
plt.title('Top 10 Locations')
plt.ylabel('%')
plt.xlabel('Locations')
df = _input1.copy()
df = df.fillna(' ', inplace=False)
df['text_merged'] = df['keyword'] + df['location'] + df['text']
df.head()
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def text_cleaner(text):
    clean_words = []
    text = ''.join([s for s in text if not s in string.punctuation])
    list = text.split()
    for word in list:
        word = word.lower().strip()
        if word.isalpha():
            if len(word) > 3:
                if word not in stopwords:
                    clean_words.append(word)
                else:
                    continue
    return clean_words
tfidf = TfidfVectorizer(analyzer=text_cleaner)
X = df['text_merged']
y = df['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=20)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

def classify(model, XX, yy):
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=20)
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)