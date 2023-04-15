import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
df.info()

def split_cabin(x):
    if len(str(x).split('/')) < 3:
        return ['Missing', 'Missing', 'Missing']
    else:
        return str(x).split('/')
import random

def preprocessing(df):
    df['HomePlanet'].fillna('Earth', inplace=True)
    df['CryoSleep'].fillna(random.choice([True, False]), inplace=True)
    df['TempCabin'] = df['Cabin'].apply(lambda x: split_cabin(x))
    df['Deck'] = df['TempCabin'].apply(lambda x: x[0])
    df['Side'] = df['TempCabin'].apply(lambda x: x[2])
    df.drop(['TempCabin', 'Cabin'], axis=1, inplace=True)
    df['Destination'].fillna('TRAPPIST-1e', inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['VIP'].fillna(False, inplace=True)
    df['RoomService'].fillna(0, inplace=True)
    df['FoodCourt'].fillna(0, inplace=True)
    df['ShoppingMall'].fillna(0, inplace=True)
    df['Spa'].fillna(0, inplace=True)
    df['VRDeck'].fillna(0, inplace=True)
    df.drop('Name', axis=1, inplace=True)
abt = df.copy()
preprocessing(abt)
abt.dropna(inplace=True)
abt.info()
abt.head()
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
X = abt.drop(['Transported', 'PassengerId'], axis=1)
X = pd.get_dummies(X)
y = abt['Transported']
y
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipelines = {'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1234)), 'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1234)), 'xgb': make_pipeline(StandardScaler(), XGBClassifier(random_state=1234)), 'svc': make_pipeline(StandardScaler(), SVC())}
pipelines.items()
grid = {'rf': {'randomforestclassifier__n_estimators': [100, 200, 300]}, 'gb': {'gradientboostingclassifier__n_estimators': [100, 200, 300]}, 'xgb': {'xgbclassifier__n_estimators': [100, 200, 300]}, 'svc': {'svc__C': [0.1, 1, 10, 100, 1000]}}
fit_models = {}
for (algo, pipeline) in pipelines.items():
    print(f'training {algo}')
    model = GridSearchCV(pipeline, grid[algo], cv=10)