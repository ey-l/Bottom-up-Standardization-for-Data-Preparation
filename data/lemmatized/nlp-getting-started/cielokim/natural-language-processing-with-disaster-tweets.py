import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.info()
_input1.isna().sum() / len(_input1) * 100
_input1['keyword'].value_counts(normalize=True)[:10].plot(kind='bar')
plt.title('Top 10 Keywords')
plt.ylabel('%')
plt.xticks(rotation=55)
(_input1['location'].value_counts(normalize=True)[:10] * 100).plot(kind='bar')
plt.title('Top 10 Locations')
plt.ylabel('%')
plt.xticks(rotation=55)
(_input1['target'].value_counts(normalize=True) * 100).plot(kind='bar')
plt.title('Ratio of Target Data')
plt.xticks(ticks=[0, 1], labels=['False Disaster', 'True Disaster'], rotation=0)
_input1[_input1['target'] == 1].groupby('keyword')['target'].sum().sort_values(ascending=False)[:10].plot(kind='bar')
_input1[_input1['target'] == 1].groupby('location')['target'].sum().sort_values(ascending=False)[:10].plot(kind='bar')
new_df = _input1.copy()
new_df = new_df.fillna(' ', inplace=False)
new_df
new_df['text_merged'] = new_df['keyword'] + new_df['location'] + new_df['text']
new_df.sample(5)
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords[:10]

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
new_df['text_merged']
(X_train, X_test, y_train, y_test) = train_test_split(new_df['text_merged'], new_df['target'], test_size=0.3, stratify=new_df['target'], random_state=20)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
clf = LogisticRegression(random_state=20)