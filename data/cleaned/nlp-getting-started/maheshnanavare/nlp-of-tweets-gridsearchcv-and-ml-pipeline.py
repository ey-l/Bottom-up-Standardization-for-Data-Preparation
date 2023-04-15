import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = 'data/input/nlp-getting-started/'
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
train_df.head()
df = train_df[['target', 'text']]
df.sample()
ax = train_df.target.value_counts().plot(kind='pie')
ax.set_title('Distribution of Tweets')
X = df['text']
y = df['target']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', max_df=0.1, min_df=10)), ('clf', SVC())])
from sklearn import set_config
set_config(display='diagram')
text_clf_lsvc
text_clf_lsvc.get_params().keys()
param_grid = [{'clf__C': [1, 10, 100], 'clf__gamma': [1, 0.1, 0.01], 'clf__kernel': ['linear', 'rbf']}]
grid_search = GridSearchCV(text_clf_lsvc, param_grid, cv=4, verbose=1)