import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
import matplotlib.pyplot as plt
df.hist(figsize=(15, 15))

labels = []
for (i, df_visualize) in enumerate(df.groupby(['Outcome'])):
    labels.append(df_visualize[0])
    plt.bar(i, df_visualize[1].count(), label=df_visualize[0])
plt.xticks(range(len(labels)), labels)
plt.legend()

df.corr()
import seaborn as sb
plt.figure(figsize=(12, 10))
sb.heatmap(df.corr(), annot=True)
df[df.columns[1:]].corr()['Outcome'][:].sort_values(ascending=False)
features = df.drop(['Outcome'], axis='columns')
labels = df.Outcome
from sklearn.preprocessing import MinMaxScaler
features_scaler = MinMaxScaler()
features = features_scaler.fit_transform(features)
labels.value_counts()
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
(x_sm, y_sm) = smote.fit_resample(features, labels)
y_sm.value_counts()
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
model_params = {'svm': {'model': SVC(gamma='auto'), 'params': {'C': [1, 10, 20, 30, 50], 'kernel': ['rbf', 'linear', 'poly']}}, 'random_forest': {'model': RandomForestClassifier(), 'params': {'n_estimators': [10, 50, 100]}}, 'logistic_regression': {'model': LogisticRegression(solver='liblinear', multi_class='auto'), 'params': {'C': [1, 5, 10]}}, 'KNN': {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [3, 7, 11, 13]}}}
from sklearn.model_selection import GridSearchCV
scores = []
for (model_name, mp) in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)