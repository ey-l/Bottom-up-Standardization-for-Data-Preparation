import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
combine = [_input1, _input0]
print(_input1.shape)
print(_input0.shape)
print(_input1.columns.values)
print(_input0.columns.values)
_input1.head(10)
_input0.head(10)
_input1.isnull().sum()
_input0.isnull().sum()
for n in _input1.columns:
    print(str(n) + ':' + str(_input1[n].unique()) + '\n')
_input1.describe()
_input1.describe(include=['O', bool])
_input1['Expenses'] = _input1.iloc[:, 7:12].sum(axis=1)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('Unknown', inplace=False)
_input0['Expenses'] = _input0.iloc[:, 7:12].sum(axis=1)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna('Unknown', inplace=False)

def fill_cryosleep(x, y):
    if x == 'Unknown' and y == 0:
        return True
    elif x == 'Unknown' and y != 0:
        return False
    else:
        return x
_input1['CryoSleep'] = _input1[['CryoSleep', 'Expenses']].apply(lambda value: fill_cryosleep(value['CryoSleep'], value['Expenses']), axis=1)
_input0['CryoSleep'] = _input0[['CryoSleep', 'Expenses']].apply(lambda value: fill_cryosleep(value['CryoSleep'], value['Expenses']), axis=1)
RoomService_mean = _input1.loc[_input1.CryoSleep == 0]['RoomService'].mean()
FoodCourt_mean = _input1.loc[_input1.CryoSleep == 0]['FoodCourt'].mean()
ShoppingMall_mean = _input1.loc[_input1.CryoSleep == 0]['ShoppingMall'].mean()
Spa_mean = _input1.loc[_input1.CryoSleep == 0].Spa.mean()
Vrdeck_mean = _input1.loc[_input1.CryoSleep == 0].VRDeck.mean()
RoomService_mean_test = _input0.loc[_input0.CryoSleep == 0]['RoomService'].mean()
FoodCourt_mean_test = _input0.loc[_input0.CryoSleep == 0]['FoodCourt'].mean()
ShoppingMall_mean_test = _input0.loc[_input0.CryoSleep == 0]['ShoppingMall'].mean()
Spa_mean_test = _input0.loc[_input0.CryoSleep == 0].Spa.mean()
Vrdeck_mean_test = _input0.loc[_input0.CryoSleep == 0].VRDeck.mean()
_input1.iloc[:, 7:12] = _input1.iloc[:, 7:12].fillna('Unknown')

def fill_otherValues(x, y, mean):
    if x == True and y == 'Unknown':
        return 0
    elif x == False and y == 'Unknown':
        return mean
    else:
        return y
_input1.RoomService = _input1[['CryoSleep', 'RoomService']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['RoomService'], RoomService_mean), axis=1)
_input1.FoodCourt = _input1[['CryoSleep', 'FoodCourt']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['FoodCourt'], FoodCourt_mean), axis=1)
_input1.ShoppingMall = _input1[['CryoSleep', 'ShoppingMall']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['ShoppingMall'], ShoppingMall_mean), axis=1)
_input1.Spa = _input1[['CryoSleep', 'Spa']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['Spa'], Spa_mean), axis=1)
_input1.VRDeck = _input1[['CryoSleep', 'VRDeck']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['VRDeck'], Vrdeck_mean), axis=1)
_input0.iloc[:, 7:12] = _input0.iloc[:, 7:12].fillna('Unknown')
_input0.RoomService = _input0[['CryoSleep', 'RoomService']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['RoomService'], RoomService_mean_test), axis=1)
_input0.FoodCourt = _input0[['CryoSleep', 'FoodCourt']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['FoodCourt'], FoodCourt_mean_test), axis=1)
_input0.ShoppingMall = _input0[['CryoSleep', 'ShoppingMall']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['ShoppingMall'], ShoppingMall_mean_test), axis=1)
_input0.Spa = _input0[['CryoSleep', 'Spa']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['Spa'], Spa_mean_test), axis=1)
_input0.VRDeck = _input0[['CryoSleep', 'VRDeck']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['VRDeck'], Vrdeck_mean_test), axis=1)
_input0.isnull().sum()
_input1.isnull().sum()
imputer = SimpleImputer(strategy='most_frequent')
_input1['Cabin_side'] = _input1.Cabin.str.split('/', expand=True)[2]
_input1['Cabin_num'] = _input1.Cabin.str.split('/', expand=True)[1]
_input1['Cabin_deck'] = _input1.Cabin.str.split('/', expand=True)[0]
_input0['Cabin_side'] = _input0.Cabin.str.split('/', expand=True)[2]
_input0['Cabin_num'] = _input0.Cabin.str.split('/', expand=True)[1]
_input0['Cabin_deck'] = _input0.Cabin.str.split('/', expand=True)[0]
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input1[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
_input1[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
name = _input1['Name'].str.split(' ', expand=True)
_input1 = pd.concat([_input1, name], axis=1)
_input1 = _input1.rename(columns={0: 'Firstname', 1: 'FamilyName'}, inplace=False)
_input1 = _input1.drop(['Name', 'Firstname'], axis=1, inplace=False)
_input1.head()
name = _input0['Name'].str.split(' ', expand=True)
_input0 = pd.concat([_input0, name], axis=1)
_input0 = _input0.rename(columns={0: 'Firstname', 1: 'FamilyName'}, inplace=False)
_input0 = _input0.drop(['Name', 'Firstname'], axis=1, inplace=False)
_input0.head()
cols = _input1.columns
_input1 = imputer.fit_transform(_input1)
_input1 = pd.DataFrame(_input1, columns=cols)
cols = _input0.columns
_input0 = imputer.fit_transform(_input0)
_input0 = pd.DataFrame(_input0, columns=cols)
encoder = OrdinalEncoder()
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Expenses', 'Cabin_side', 'Cabin_num', 'Cabin_deck', 'FamilyName']
sample_encode = encoder.fit_transform(_input1[features])
_input1[features] = sample_encode
features_test = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenses', 'Cabin_side', 'Cabin_num', 'Cabin_deck', 'FamilyName']
sample_encode = encoder.fit_transform(_input0[features_test])
_input0[features_test] = sample_encode
scaling = MinMaxScaler()
sample_scale = scaling.fit_transform(_input1[features])
_input1[features] = sample_scale
scaling = MinMaxScaler()
scale = scaling.fit_transform(_input0[features_test])
_input0[features_test] = scale
_input1.head()
_input0.head()
X_train = _input1.drop(['PassengerId', 'Transported'], axis=1)
y_train = _input1['Transported']
(X_train.shape, y_train.shape)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
log = LogisticRegression()