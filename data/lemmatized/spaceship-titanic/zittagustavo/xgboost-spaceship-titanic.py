import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1 = _input1.set_index('PassengerId', inplace=False)
_input1.head()
_input1.info()
_input1.describe(include=('object', 'bool'))
_input1.isna().sum()
cols = _input1.columns
for col in cols:
    if _input1[col].dtype != 'float64':
        print(_input1[col].value_counts())
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].median())
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].median())
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].median())
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].median())
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].median())
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input1['VIP'] = _input1['VIP'].fillna(False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input1['Cabin'] = _input1['Cabin'].fillna('G/0/S')
_input1.head()
_input1['Total_spending'] = _input1['RoomService'] + _input1['FoodCourt']
+_input1['ShoppingMall'] + _input1['VRDeck'] + _input1['Spa']
_input1['Cabin_Side'] = _input1['Cabin'].str.split('/').str[2]
_input1['Cabin_Deck'] = _input1['Cabin'].str.split('/').str[0]
_input1 = _input1.drop('Cabin', axis=1)
_input1 = _input1.dropna()
_input1 = _input1.drop('Name', axis='columns', inplace=False)
_input1.head()
_input1['HomePlanet'] = _input1['HomePlanet'].astype('category')
_input1['Destination'] = _input1['Destination'].astype('category')
_input1['Cabin_Deck'] = _input1['Cabin_Deck'].astype('category')
_input1['Cabin_Side'] = _input1['Cabin_Side'].astype('category')
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side'])
_input1.columns
_input1.head()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 'max_depth': [3, 4, 5, 6, 8, 10, 12, 15], 'min_child_weight': [1, 3, 5, 7], 'gamma': [0.0, 0.1, 0.2, 0.3, 0.4], 'colsample_bytree': [0.3, 0.4, 0.5, 0.7]}
import xgboost
classifier = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)