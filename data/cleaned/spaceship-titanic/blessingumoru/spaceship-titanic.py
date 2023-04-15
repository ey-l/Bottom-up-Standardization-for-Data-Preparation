import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
trainDf = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
testDf = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
submit = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv', index_col='PassengerId')
trainDf.isna().sum()
trainDf.info()
trainDf['HomePlanet'].fillna('None', inplace=True)
trainDf['Cabin'].fillna('None', inplace=True)
trainDf['Destination'].fillna('None', inplace=True)

def fixNumVal(data):
    for cols in data.columns:
        if data[cols].dtype == 'float64' or data[cols].dtype == 'int64':
            data[cols].fillna(data[cols].mean(), inplace=True)
        else:
            data[cols].fillna(data[cols].mode()[0], inplace=True)
fixNumVal(trainDf)
fixNumVal(testDf)
trainDf.isna().sum()
from sklearn.preprocessing import LabelEncoder
objectColumns = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'Name', 'VIP']
encoder = LabelEncoder()

def enconding(dataset):
    for col in objectColumns:
        dataset[col] = encoder.fit_transform(dataset[col])
enconding(trainDf)
enconding(testDf)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error as mae
model = LogisticRegression()
kfold = KFold(n_splits=10, random_state=None)
y = trainDf.Transported
X = trainDf.drop(['Transported', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Cabin'], axis=1)
score = cross_val_score(model, X, y, cv=kfold)
score.mean()
X_test = testDf.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Cabin'], axis=1)