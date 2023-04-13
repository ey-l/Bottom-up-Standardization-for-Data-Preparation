import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
y = _input1['Transported']
_input1 = _input1.drop(columns='Transported')
dfX = pd.concat([_input1, _input0], axis=0)
for x in ['PassengerId', 'Cabin', 'Name']:
    dfX = dfX.drop(columns=x)
for x in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
    dfX[x] = dfX[x].astype('category').cat.codes
for x in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    dfX[x] = dfX[x].fillna(dfX[x].mean())
dfX_scaled = MinMaxScaler().fit_transform(dfX)
X1 = dfX_scaled[:_input1.shape[0], :]
X_sub1 = dfX_scaled[_input1.shape[0]:, :]
y1 = y
r = train_test_split(X1, y1, test_size=0.25, random_state=2022)
(X_train, X_test, y_train, y_test) = r