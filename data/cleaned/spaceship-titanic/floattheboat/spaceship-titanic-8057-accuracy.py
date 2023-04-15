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
raw = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test['Total_Spend'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']
test[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = test['Cabin'].str.split('/', expand=True)
test['Cabin_Num'] = test['Cabin_Num'].astype(np.float64)
test.head()
raw[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = raw['Cabin'].str.split('/', expand=True)
raw['Total_Spend'] = raw['RoomService'] + raw['FoodCourt'] + raw['ShoppingMall'] + raw['Spa'] + raw['VRDeck']
raw.head()
raw['Cabin_Num'] = raw['Cabin_Num'].astype(np.float64)
num_attribs = raw[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Spend', 'Cabin_Num']]
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
num_attribs = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Spend', 'Cabin_Num']
ohe_attribs = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Side']
ord_attribs = ['Cabin_Deck']
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), ('ohe', OneHotEncoder(sparse=False), ohe_attribs), ('ord', OrdinalEncoder(encoded_missing_value=-1), ord_attribs)])
spaceship_prepared = full_pipeline.fit_transform(raw)
test_prepared = full_pipeline.transform(test)
spaceship_prepared
y = raw['Transported']
X = spaceship_prepared
booster = xgb.XGBClassifier(max_depth=2, n_estimators=43)
rf = RandomForestClassifier()
kneigh = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree', leaf_size=25)
voting_clf = VotingClassifier(estimators=[('booster', booster), ('rf', rf), ('kneigh', kneigh)], voting='soft')