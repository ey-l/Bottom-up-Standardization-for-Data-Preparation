import numpy as np
import pandas as pd
import statistics as st
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
data.info()
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data['Outcome'])
features = data.drop('Outcome', axis=1)
target = data['Outcome']
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

def get_voting():
    model = list()
    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier()
    model3 = SVC(probability=True, kernel='poly', degree=2)
    model4 = LogisticRegression(max_iter=500)
    ensemble = VotingClassifier(estimators=[('lr', model4), ('knn', model2), ('svc', model3)], voting='hard')
    return ensemble

def get_models():
    models = dict()
    models['Decision Tree'] = DecisionTreeClassifier()
    models['KNeighbours'] = KNeighborsClassifier()
    models['Logistic Regression'] = LogisticRegression(max_iter=500)
    models['SVC'] = SVC(probability=True, kernel='poly', degree=2)
    models['hard_voting'] = get_voting()
    return models

def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
models = get_models()
(results, names) = (list(), list())
for (name, model) in models.items():
    scores = evaluate_model(model, features, target)
    results.append(scores)
    names.append(name)
    print('>%s , accuracy-%.3f ' % (name, np.mean(scores)))
plt.figure(figsize=(10, 5))
plt.boxplot(results, labels=names, showmeans=True)

from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(features, target, random_state=42)
model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3 = LogisticRegression(max_iter=500)