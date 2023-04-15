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
X = pd.read_csv('data/input/spaceship-titanic/train.csv')
X_sub = pd.read_csv('data/input/spaceship-titanic/test.csv')
y = X['Transported']
X = X.drop(columns='Transported')
dfX = pd.concat([X, X_sub], axis=0)
for x in ['PassengerId', 'Cabin', 'Name']:
    dfX = dfX.drop(columns=x)
for x in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
    dfX[x] = dfX[x].astype('category').cat.codes
for x in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    dfX[x] = dfX[x].fillna(dfX[x].mean())
dfX_scaled = MinMaxScaler().fit_transform(dfX)
X1 = dfX_scaled[:X.shape[0], :]
X_sub1 = dfX_scaled[X.shape[0]:, :]
y1 = y
r = train_test_split(X1, y1, test_size=0.25, random_state=2022)
(X_train, X_test, y_train, y_test) = r