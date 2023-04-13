import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import random
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
scl = MinMaxScaler()
scl1 = StandardScaler()
le = LabelEncoder()

def derive_columns(df):
    _input1.loc[_input1.Cabin.isna() == True, 'Cabin'] = 'Missing/Missing/Missing'
    _input1.Name.fillna('Missing Missing')
    _input1['Deck'] = _input1.Cabin.apply(lambda x: x.split('/')[0])
    _input1['Deck_number'] = _input1.Cabin.apply(lambda x: x.split('/')[1])
    _input1['Port'] = _input1.Cabin.apply(lambda x: x.split('/')[2])
    _input1['Surname'] = _input1.Name.apply(lambda x: str(x).split(' ')[-1])
    _input1['Groups'] = _input1.PassengerId.apply(lambda x: str(x).split('_')[0])
    _input1 = _input1.drop(columns=['Name', 'Cabin', 'PassengerId'], inplace=False)
    return _input1
derive_columns(_input1)

def fill_columns(df):
    _input1.loc[_input1.Spa.isna() == True, 'Spa'] = 0
    _input1.loc[_input1.ShoppingMall.isna() == True, 'ShoppingMall'] = 0
    _input1.loc[_input1.VRDeck.isna() == True, 'VRDeck'] = 0
    _input1.loc[_input1.FoodCourt.isna() == True, 'FoodCourt'] = 0
    _input1.loc[_input1.RoomService.isna() == True, 'RoomService'] = 0
    _input1.VIP = _input1.VIP.fillna('False', inplace=False)
    _input1.Age = _input1.Age.fillna(_input1.Age.mean(), inplace=False)
    weight_HP = [_input1.HomePlanet.value_counts().values[0] / len(_input1.HomePlanet), _input1.HomePlanet.value_counts().values[1] / len(_input1.HomePlanet), _input1.HomePlanet.value_counts().values[2] / len(_input1.HomePlanet)]
    fillings_HP = random.choices(_input1.HomePlanet.unique()[:-1], weights=weight_HP, k=_input1[_input1.HomePlanet.isna() == True].shape[0])
    _input1.loc[_input1.HomePlanet.isna() == True, 'HomePlanet'] = fillings_HP
    weight_CR = [_input1.CryoSleep.value_counts().values[0] / len(_input1.CryoSleep), _input1.CryoSleep.value_counts().values[1] / len(_input1.CryoSleep)]
    fillings_CR = random.choices(_input1.CryoSleep.unique()[:-1], weights=weight_CR, k=_input1[_input1.CryoSleep.isna() == True].shape[0])
    _input1.loc[_input1.CryoSleep.isna() == True, 'CryoSleep'] = fillings_CR
    weight_dest = [_input1.Destination.value_counts().values[0] / len(_input1.Destination), _input1.Destination.value_counts().values[1] / len(_input1.Destination), _input1.Destination.value_counts().values[2] / len(_input1.Destination)]
    fillings_dest = random.choices(_input1.Destination.unique()[:-1], weights=weight_dest, k=_input1[_input1.Destination.isna() == True].shape[0])
    _input1.loc[_input1.Destination.isna() == True, 'Destination'] = fillings_dest
    _input1.loc[_input1.Deck_number == 'Missing', 'Deck_number'] = random.choices(np.arange(1, 900, 1), k=_input1[_input1.Deck_number == 'Missing'].shape[0])
    _input1 = _input1.drop(columns=['Deck_number'], inplace=False)
    return _input1
fill_columns(_input1)

def transform_columns(df):
    _input1.Destination = le.fit_transform(_input1.Destination)
    _input1.CryoSleep = le.fit_transform(_input1.CryoSleep)
    _input1.HomePlanet = le.fit_transform(_input1.HomePlanet)
    _input1.Deck = le.fit_transform(_input1.Deck)
    _input1.Port = le.fit_transform(_input1.Port)
    _input1.Surname = le.fit_transform(_input1.Surname)
    _input1.VIP = _input1.VIP.apply(lambda x: 1 if x == True else 0)
    return _input1
transform_columns(_input1)
x = _input1.drop(columns=['Transported', 'HomePlanet'])
y = _input1.Transported
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=1234)
pipelines = {'rf': make_pipeline(StandardScaler(), RandomForestClassifier()), 'adaboost': make_pipeline(StandardScaler(), AdaBoostClassifier()), 'rf1': make_pipeline(MinMaxScaler(), RandomForestClassifier()), 'rf2': make_pipeline(MinMaxScaler(), StandardScaler(), RandomForestClassifier()), 'adaboost1': make_pipeline(MinMaxScaler(), AdaBoostClassifier()), 'svc': make_pipeline(MinMaxScaler(), StandardScaler(), SVC())}
gridx = {'rf': {'randomforestclassifier__n_estimators': [300, 400, 450]}, 'rf1': {'randomforestclassifier__n_estimators': [300, 400, 450]}, 'rf2': {'randomforestclassifier__n_estimators': [350, 400, 450], 'randomforestclassifier__criterion': ['entropy'], 'randomforestclassifier__max_depth': [12, 14, 16]}, 'adaboost': {'adaboostclassifier__n_estimators': [80, 100, 120]}, 'adaboost1': {'adaboostclassifier__n_estimators': [80, 100, 120]}, 'svc': {'svc__degree': [3, 5, 2], 'svc__C': [10, 5, 1, 1.2]}}
for model in pipelines.keys():
    model = GridSearchCV(pipelines[model], param_grid=gridx[model], n_jobs=-1, cv=5, scoring='accuracy')