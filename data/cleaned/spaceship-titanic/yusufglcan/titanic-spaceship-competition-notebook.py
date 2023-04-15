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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df1 = pd.read_csv('data/input/spaceship-titanic/test.csv')
df2 = pd.read_csv('data/input/spaceship-titanic/test.csv')
scl = MinMaxScaler()
scl1 = StandardScaler()
le = LabelEncoder()

def derive_columns(df):
    df.loc[df.Cabin.isna() == True, 'Cabin'] = 'Missing/Missing/Missing'
    df.Name.fillna('Missing Missing')
    df['Deck'] = df.Cabin.apply(lambda x: x.split('/')[0])
    df['Deck_number'] = df.Cabin.apply(lambda x: x.split('/')[1])
    df['Port'] = df.Cabin.apply(lambda x: x.split('/')[2])
    df['Surname'] = df.Name.apply(lambda x: str(x).split(' ')[-1])
    df['Groups'] = df.PassengerId.apply(lambda x: str(x).split('_')[0])
    df.drop(columns=['Name', 'Cabin', 'PassengerId'], inplace=True)
    return df
derive_columns(df)

def fill_columns(df):
    df.loc[df.Spa.isna() == True, 'Spa'] = 0
    df.loc[df.ShoppingMall.isna() == True, 'ShoppingMall'] = 0
    df.loc[df.VRDeck.isna() == True, 'VRDeck'] = 0
    df.loc[df.FoodCourt.isna() == True, 'FoodCourt'] = 0
    df.loc[df.RoomService.isna() == True, 'RoomService'] = 0
    df.VIP.fillna('False', inplace=True)
    df.Age.fillna(df.Age.mean(), inplace=True)
    weight_HP = [df.HomePlanet.value_counts().values[0] / len(df.HomePlanet), df.HomePlanet.value_counts().values[1] / len(df.HomePlanet), df.HomePlanet.value_counts().values[2] / len(df.HomePlanet)]
    fillings_HP = random.choices(df.HomePlanet.unique()[:-1], weights=weight_HP, k=df[df.HomePlanet.isna() == True].shape[0])
    df.loc[df.HomePlanet.isna() == True, 'HomePlanet'] = fillings_HP
    weight_CR = [df.CryoSleep.value_counts().values[0] / len(df.CryoSleep), df.CryoSleep.value_counts().values[1] / len(df.CryoSleep)]
    fillings_CR = random.choices(df.CryoSleep.unique()[:-1], weights=weight_CR, k=df[df.CryoSleep.isna() == True].shape[0])
    df.loc[df.CryoSleep.isna() == True, 'CryoSleep'] = fillings_CR
    weight_dest = [df.Destination.value_counts().values[0] / len(df.Destination), df.Destination.value_counts().values[1] / len(df.Destination), df.Destination.value_counts().values[2] / len(df.Destination)]
    fillings_dest = random.choices(df.Destination.unique()[:-1], weights=weight_dest, k=df[df.Destination.isna() == True].shape[0])
    df.loc[df.Destination.isna() == True, 'Destination'] = fillings_dest
    df.loc[df.Deck_number == 'Missing', 'Deck_number'] = random.choices(np.arange(1, 900, 1), k=df[df.Deck_number == 'Missing'].shape[0])
    df.drop(columns=['Deck_number'], inplace=True)
    return df
fill_columns(df)

def transform_columns(df):
    df.Destination = le.fit_transform(df.Destination)
    df.CryoSleep = le.fit_transform(df.CryoSleep)
    df.HomePlanet = le.fit_transform(df.HomePlanet)
    df.Deck = le.fit_transform(df.Deck)
    df.Port = le.fit_transform(df.Port)
    df.Surname = le.fit_transform(df.Surname)
    df.VIP = df.VIP.apply(lambda x: 1 if x == True else 0)
    return df
transform_columns(df)
x = df.drop(columns=['Transported', 'HomePlanet'])
y = df.Transported
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=1234)
pipelines = {'rf': make_pipeline(StandardScaler(), RandomForestClassifier()), 'adaboost': make_pipeline(StandardScaler(), AdaBoostClassifier()), 'rf1': make_pipeline(MinMaxScaler(), RandomForestClassifier()), 'rf2': make_pipeline(MinMaxScaler(), StandardScaler(), RandomForestClassifier()), 'adaboost1': make_pipeline(MinMaxScaler(), AdaBoostClassifier()), 'svc': make_pipeline(MinMaxScaler(), StandardScaler(), SVC())}
gridx = {'rf': {'randomforestclassifier__n_estimators': [300, 400, 450]}, 'rf1': {'randomforestclassifier__n_estimators': [300, 400, 450]}, 'rf2': {'randomforestclassifier__n_estimators': [350, 400, 450], 'randomforestclassifier__criterion': ['entropy'], 'randomforestclassifier__max_depth': [12, 14, 16]}, 'adaboost': {'adaboostclassifier__n_estimators': [80, 100, 120]}, 'adaboost1': {'adaboostclassifier__n_estimators': [80, 100, 120]}, 'svc': {'svc__degree': [3, 5, 2], 'svc__C': [10, 5, 1, 1.2]}}
for model in pipelines.keys():
    model = GridSearchCV(pipelines[model], param_grid=gridx[model], n_jobs=-1, cv=5, scoring='accuracy')