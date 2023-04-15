import pandas as pd
import numpy as np
dados_treino = pd.read_csv('data/input/spaceship-titanic/train.csv')
dados_treino.set_index('PassengerId', inplace=True)
dados_treino.head()
dados_treino.info()
dados_treino.describe(include=('object', 'bool'))
dados_treino.isna().sum()
cols = dados_treino.columns
for col in cols:
    if dados_treino[col].dtype != 'float64':
        print(dados_treino[col].value_counts())
dados_treino['RoomService'] = dados_treino['RoomService'].fillna(dados_treino['RoomService'].median())
dados_treino['FoodCourt'] = dados_treino['FoodCourt'].fillna(dados_treino['FoodCourt'].median())
dados_treino['ShoppingMall'] = dados_treino['ShoppingMall'].fillna(dados_treino['ShoppingMall'].median())
dados_treino['VRDeck'] = dados_treino['VRDeck'].fillna(dados_treino['VRDeck'].median())
dados_treino['Spa'] = dados_treino['Spa'].fillna(dados_treino['Spa'].median())
dados_treino['HomePlanet'] = dados_treino['HomePlanet'].fillna('Earth')
dados_treino['Destination'] = dados_treino['Destination'].fillna('TRAPPIST-1e')
dados_treino['CryoSleep'] = dados_treino['CryoSleep'].fillna(False)
dados_treino['VIP'] = dados_treino['VIP'].fillna(False)
dados_treino['Age'] = dados_treino['Age'].fillna(dados_treino['Age'].mean())
dados_treino['Cabin'] = dados_treino['Cabin'].fillna('G/0/S')
dados_treino.head()
dados_treino['Total_spending'] = dados_treino['RoomService'] + dados_treino['FoodCourt']
+dados_treino['ShoppingMall'] + dados_treino['VRDeck'] + dados_treino['Spa']
dados_treino['Cabin_Side'] = dados_treino['Cabin'].str.split('/').str[2]
dados_treino['Cabin_Deck'] = dados_treino['Cabin'].str.split('/').str[0]
dados_treino = dados_treino.drop('Cabin', axis=1)
dados_treino = dados_treino.dropna()
dados_treino.drop('Name', axis='columns', inplace=True)
dados_treino.head()
dados_treino['HomePlanet'] = dados_treino['HomePlanet'].astype('category')
dados_treino['Destination'] = dados_treino['Destination'].astype('category')
dados_treino['Cabin_Deck'] = dados_treino['Cabin_Deck'].astype('category')
dados_treino['Cabin_Side'] = dados_treino['Cabin_Side'].astype('category')
dados_treino = pd.get_dummies(dados_treino, columns=['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side'])
dados_treino.columns
dados_treino.head()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 'max_depth': [3, 4, 5, 6, 8, 10, 12, 15], 'min_child_weight': [1, 3, 5, 7], 'gamma': [0.0, 0.1, 0.2, 0.3, 0.4], 'colsample_bytree': [0.3, 0.4, 0.5, 0.7]}
import xgboost
classifier = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)