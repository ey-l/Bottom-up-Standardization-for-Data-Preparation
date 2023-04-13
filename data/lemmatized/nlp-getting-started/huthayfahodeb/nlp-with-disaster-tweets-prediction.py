import pandas as pd
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
features = 'text'
target = 'target'
_input1['text'].isnull().sum()
_input1[target].value_counts()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(_input1[features], _input1[target], test_size=0.33, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
svc_params = {'clf': SVC(C=1, kernel='rbf', random_state=42), 'tfidf': TfidfVectorizer(stop_words='english', max_df=0.8), 'tfidf__min_df': 0.05, 'tfidf__max_features': 5000, 'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf', 'poly'], 'clf__gamma': ['scale', 'auto']}
classifier = Pipeline([('tfidf', TfidfVectorizer()), ('clf', svc_params['clf'])])