import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input1.info()
_input1.isnull().sum()
_input1 = _input1.fillna(_input1.median())
_input1.isnull().sum()
_input1 = _input1.dropna()
_input1.isnull().sum()
df2 = _input1.groupby('HomePlanet')[['Transported']].count().reset_index()
fig = px.bar(df2, x='HomePlanet', y='Transported', height=400, text_auto=True)
fig.show()
df2 = _input1.groupby('Destination')[['Transported']].count().reset_index().sort_values(['Transported'], ascending=False)
fig = px.bar(df2, x='Destination', y='Transported', height=400, text_auto=True, color_discrete_sequence=px.colors.qualitative.Dark24)
fig.show()
df2 = _input1.groupby(['Transported'])[['Age']].mean().reset_index()
fig = px.pie(df2, values='Age', names='Transported', height=400)
fig.show()
df2 = _input1.groupby(['HomePlanet'])[['RoomService']].mean().reset_index().sort_values(['RoomService'], ascending=False)
fig = px.bar(df2, x='HomePlanet', y='RoomService', height=400, text_auto=True)
fig.show()
df2 = _input1.groupby(['HomePlanet'])[['ShoppingMall']].mean().reset_index().sort_values(['ShoppingMall'], ascending=False)
fig = px.bar(df2, x='HomePlanet', y='ShoppingMall', height=400, text_auto=True, color_discrete_sequence=px.colors.qualitative.Bold)
fig.show()
df2 = _input1.groupby(['HomePlanet'])[['Spa']].mean().reset_index().sort_values(['Spa'], ascending=False)
fig = px.bar(df2, x='HomePlanet', y='Spa', height=400, text_auto=True, color_discrete_sequence=px.colors.qualitative.Dark2)
fig.show()
df2 = _input1.groupby(['HomePlanet'])[['VRDeck']].mean().reset_index().sort_values(['VRDeck'], ascending=False)
fig = px.bar(df2, x='HomePlanet', y='VRDeck', height=400, text_auto=True, color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()
fig = px.histogram(_input1, x='Age', color_discrete_sequence=px.colors.qualitative.T10)
fig.show()
_input1[['first_name', 'last_name']] = _input1['Name'].str.split(pat=' ', expand=True)
_input1
df2 = _input1.groupby('first_name')[['Transported']].count().reset_index().sort_values(['Transported'], ascending=False).head(10)
fig = px.bar(df2, x='first_name', y='Transported', color_discrete_sequence=px.colors.qualitative.Pastel2, text_auto=True)
fig.show()
df2 = _input1.groupby('last_name')[['Transported']].count().reset_index().sort_values(['Transported'], ascending=False).head(10)
fig = px.bar(df2, x='last_name', y='Transported', color_discrete_sequence=px.colors.qualitative.Pastel1, text_auto=True)
fig.show()
_input1 = _input1.drop(['Name', 'first_name', 'last_name'], axis=1)
_input1.select_dtypes(exclude=np.number).columns
for coluna in _input1.select_dtypes(exclude=np.number).columns:
    print(dict(enumerate(_input1[coluna].astype('category').cat.categories)))
    _input1[coluna] = _input1[coluna].astype('category').cat.codes
classifiers = {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(), 'Decision Tree': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier(), 'AdaBoost': AdaBoostClassifier()}
samplers = {'Random_under_sampler': RandomUnderSampler(), 'Random_over_sampler': RandomOverSampler()}

def df_split(df, target='TARGET'):
    _input1 = _input1.fillna(999)
    x = _input1.drop('Transported', axis=1)
    y = _input1['Transported']
    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    return (x_train, x_test, y_train, y_test)

def train_clfs(df, classifiers, samplers):
    (x_train, x_test, y_train, y_test) = df_split(_input1)
    names_samplers = []
    names_clfs = []
    results_train_cv_roc_auc = []
    results_train_cv_recall = []
    results_train_cv_accuracy = []
    results_test_roc_auc = []
    results_test_recall = []
    results_test_accuracy = []
    modelos = []
    for (name_sampler, sampler) in samplers.items():
        print(f'Sampler: {name_sampler}\n')
        for (name_clf, clf) in classifiers.items():
            print(f'Classifier: {name_clf}\n')
            pipeline = Pipeline([('sampler', sampler), ('clf', clf)])
            cv_auc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='roc_auc')
            cv_rec = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='recall')
            cv_acc = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='accuracy')