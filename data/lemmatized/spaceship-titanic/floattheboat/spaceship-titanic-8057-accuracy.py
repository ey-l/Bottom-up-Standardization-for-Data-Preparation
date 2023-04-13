import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0['Total_Spend'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input0[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = _input0['Cabin'].str.split('/', expand=True)
_input0['Cabin_Num'] = _input0['Cabin_Num'].astype(np.float64)
_input0.head()
_input1[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = _input1['Cabin'].str.split('/', expand=True)
_input1['Total_Spend'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1.head()
_input1['Cabin_Num'] = _input1['Cabin_Num'].astype(np.float64)
num_attribs = _input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Spend', 'Cabin_Num']]
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
num_attribs = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Spend', 'Cabin_Num']
ohe_attribs = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Side']
ord_attribs = ['Cabin_Deck']
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), ('ohe', OneHotEncoder(sparse=False), ohe_attribs), ('ord', OrdinalEncoder(encoded_missing_value=-1), ord_attribs)])
spaceship_prepared = full_pipeline.fit_transform(_input1)
test_prepared = full_pipeline.transform(_input0)
spaceship_prepared
y = _input1['Transported']
X = spaceship_prepared
booster = xgb.XGBClassifier(max_depth=2, n_estimators=43)
rf = RandomForestClassifier()
kneigh = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree', leaf_size=25)
voting_clf = VotingClassifier(estimators=[('booster', booster), ('rf', rf), ('kneigh', kneigh)], voting='soft')