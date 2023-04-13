import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()

def split_cabin(x):
    if len(str(x).split('/')) < 3:
        return ['Missing', 'Missing', 'Missing']
    else:
        return str(x).split('/')
import random

def preprocessing(df):
    _input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
    _input1['CryoSleep'] = _input1['CryoSleep'].fillna(random.choice([True, False]), inplace=False)
    _input1['TempCabin'] = _input1['Cabin'].apply(lambda x: split_cabin(x))
    _input1['Deck'] = _input1['TempCabin'].apply(lambda x: x[0])
    _input1['Side'] = _input1['TempCabin'].apply(lambda x: x[2])
    _input1 = _input1.drop(['TempCabin', 'Cabin'], axis=1, inplace=False)
    _input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
    _input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
    _input1['VIP'] = _input1['VIP'].fillna(False, inplace=False)
    _input1['RoomService'] = _input1['RoomService'].fillna(0, inplace=False)
    _input1['FoodCourt'] = _input1['FoodCourt'].fillna(0, inplace=False)
    _input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0, inplace=False)
    _input1['Spa'] = _input1['Spa'].fillna(0, inplace=False)
    _input1['VRDeck'] = _input1['VRDeck'].fillna(0, inplace=False)
    _input1 = _input1.drop('Name', axis=1, inplace=False)
abt = _input1.copy()
preprocessing(abt)
abt = abt.dropna(inplace=False)
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