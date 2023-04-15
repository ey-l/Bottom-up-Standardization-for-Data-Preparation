import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
combine = [train_data, test_data]
print(train_data.shape)
print(test_data.shape)
print(train_data.columns.values)
print(test_data.columns.values)
train_data.head(10)
test_data.head(10)
train_data.isnull().sum()
test_data.isnull().sum()
for n in train_data.columns:
    print(str(n) + ':' + str(train_data[n].unique()) + '\n')
train_data.describe()
train_data.describe(include=['O', bool])
train_data['Expenses'] = train_data.iloc[:, 7:12].sum(axis=1)
train_data['CryoSleep'].fillna('Unknown', inplace=True)
test_data['Expenses'] = test_data.iloc[:, 7:12].sum(axis=1)
test_data['CryoSleep'].fillna('Unknown', inplace=True)

def fill_cryosleep(x, y):
    if x == 'Unknown' and y == 0:
        return True
    elif x == 'Unknown' and y != 0:
        return False
    else:
        return x
train_data['CryoSleep'] = train_data[['CryoSleep', 'Expenses']].apply(lambda value: fill_cryosleep(value['CryoSleep'], value['Expenses']), axis=1)
test_data['CryoSleep'] = test_data[['CryoSleep', 'Expenses']].apply(lambda value: fill_cryosleep(value['CryoSleep'], value['Expenses']), axis=1)
RoomService_mean = train_data.loc[train_data.CryoSleep == 0]['RoomService'].mean()
FoodCourt_mean = train_data.loc[train_data.CryoSleep == 0]['FoodCourt'].mean()
ShoppingMall_mean = train_data.loc[train_data.CryoSleep == 0]['ShoppingMall'].mean()
Spa_mean = train_data.loc[train_data.CryoSleep == 0].Spa.mean()
Vrdeck_mean = train_data.loc[train_data.CryoSleep == 0].VRDeck.mean()
RoomService_mean_test = test_data.loc[test_data.CryoSleep == 0]['RoomService'].mean()
FoodCourt_mean_test = test_data.loc[test_data.CryoSleep == 0]['FoodCourt'].mean()
ShoppingMall_mean_test = test_data.loc[test_data.CryoSleep == 0]['ShoppingMall'].mean()
Spa_mean_test = test_data.loc[test_data.CryoSleep == 0].Spa.mean()
Vrdeck_mean_test = test_data.loc[test_data.CryoSleep == 0].VRDeck.mean()
train_data.iloc[:, 7:12] = train_data.iloc[:, 7:12].fillna('Unknown')

def fill_otherValues(x, y, mean):
    if x == True and y == 'Unknown':
        return 0
    elif x == False and y == 'Unknown':
        return mean
    else:
        return y
train_data.RoomService = train_data[['CryoSleep', 'RoomService']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['RoomService'], RoomService_mean), axis=1)
train_data.FoodCourt = train_data[['CryoSleep', 'FoodCourt']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['FoodCourt'], FoodCourt_mean), axis=1)
train_data.ShoppingMall = train_data[['CryoSleep', 'ShoppingMall']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['ShoppingMall'], ShoppingMall_mean), axis=1)
train_data.Spa = train_data[['CryoSleep', 'Spa']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['Spa'], Spa_mean), axis=1)
train_data.VRDeck = train_data[['CryoSleep', 'VRDeck']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['VRDeck'], Vrdeck_mean), axis=1)
test_data.iloc[:, 7:12] = test_data.iloc[:, 7:12].fillna('Unknown')
test_data.RoomService = test_data[['CryoSleep', 'RoomService']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['RoomService'], RoomService_mean_test), axis=1)
test_data.FoodCourt = test_data[['CryoSleep', 'FoodCourt']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['FoodCourt'], FoodCourt_mean_test), axis=1)
test_data.ShoppingMall = test_data[['CryoSleep', 'ShoppingMall']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['ShoppingMall'], ShoppingMall_mean_test), axis=1)
test_data.Spa = test_data[['CryoSleep', 'Spa']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['Spa'], Spa_mean_test), axis=1)
test_data.VRDeck = test_data[['CryoSleep', 'VRDeck']].apply(lambda room: fill_otherValues(room['CryoSleep'], room['VRDeck'], Vrdeck_mean_test), axis=1)
test_data.isnull().sum()
train_data.isnull().sum()
imputer = SimpleImputer(strategy='most_frequent')
train_data['Cabin_side'] = train_data.Cabin.str.split('/', expand=True)[2]
train_data['Cabin_num'] = train_data.Cabin.str.split('/', expand=True)[1]
train_data['Cabin_deck'] = train_data.Cabin.str.split('/', expand=True)[0]
test_data['Cabin_side'] = test_data.Cabin.str.split('/', expand=True)[2]
test_data['Cabin_num'] = test_data.Cabin.str.split('/', expand=True)[1]
test_data['Cabin_deck'] = test_data.Cabin.str.split('/', expand=True)[0]
train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
train_data[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_data[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_data[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
train_data[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False)
name = train_data['Name'].str.split(' ', expand=True)
train_data = pd.concat([train_data, name], axis=1)
train_data.rename(columns={0: 'Firstname', 1: 'FamilyName'}, inplace=True)
train_data.drop(['Name', 'Firstname'], axis=1, inplace=True)
train_data.head()
name = test_data['Name'].str.split(' ', expand=True)
test_data = pd.concat([test_data, name], axis=1)
test_data.rename(columns={0: 'Firstname', 1: 'FamilyName'}, inplace=True)
test_data.drop(['Name', 'Firstname'], axis=1, inplace=True)
test_data.head()
cols = train_data.columns
train_data = imputer.fit_transform(train_data)
train_data = pd.DataFrame(train_data, columns=cols)
cols = test_data.columns
test_data = imputer.fit_transform(test_data)
test_data = pd.DataFrame(test_data, columns=cols)
encoder = OrdinalEncoder()
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Expenses', 'Cabin_side', 'Cabin_num', 'Cabin_deck', 'FamilyName']
sample_encode = encoder.fit_transform(train_data[features])
train_data[features] = sample_encode
features_test = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenses', 'Cabin_side', 'Cabin_num', 'Cabin_deck', 'FamilyName']
sample_encode = encoder.fit_transform(test_data[features_test])
test_data[features_test] = sample_encode
scaling = MinMaxScaler()
sample_scale = scaling.fit_transform(train_data[features])
train_data[features] = sample_scale
scaling = MinMaxScaler()
scale = scaling.fit_transform(test_data[features_test])
test_data[features_test] = scale
train_data.head()
test_data.head()
X_train = train_data.drop(['PassengerId', 'Transported'], axis=1)
y_train = train_data['Transported']
(X_train.shape, y_train.shape)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
log = LogisticRegression()