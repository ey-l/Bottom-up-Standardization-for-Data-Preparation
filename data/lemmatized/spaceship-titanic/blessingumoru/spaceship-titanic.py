import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv', index_col='PassengerId')
_input1.isna().sum()
_input1.info()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('None', inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna('None', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('None', inplace=False)

def fixNumVal(data):
    for cols in data.columns:
        if data[cols].dtype == 'float64' or data[cols].dtype == 'int64':
            data[cols] = data[cols].fillna(data[cols].mean(), inplace=False)
        else:
            data[cols] = data[cols].fillna(data[cols].mode()[0], inplace=False)
fixNumVal(_input1)
fixNumVal(_input0)
_input1.isna().sum()
from sklearn.preprocessing import LabelEncoder
objectColumns = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'Name', 'VIP']
encoder = LabelEncoder()

def enconding(dataset):
    for col in objectColumns:
        dataset[col] = encoder.fit_transform(dataset[col])
enconding(_input1)
enconding(_input0)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error as mae
model = LogisticRegression()
kfold = KFold(n_splits=10, random_state=None)
y = _input1.Transported
X = _input1.drop(['Transported', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Cabin'], axis=1)
score = cross_val_score(model, X, y, cv=kfold)
score.mean()
X_test = _input0.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Cabin'], axis=1)