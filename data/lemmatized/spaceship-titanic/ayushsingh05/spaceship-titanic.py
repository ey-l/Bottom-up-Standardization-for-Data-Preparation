import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(10)
_input1.shape
_input0.shape
_input1.info()
_input1.describe()
_input1.isnull().sum().plot.bar()
col = _input1.loc[:, 'RoomService':'VRDeck'].columns
_input1.groupby('VIP')[col].mean()
_input1.groupby('HomePlanet')[col].mean()
_input1.groupby('CryoSleep')[col].mean()
temp = _input1['CryoSleep'] == True
_input1.loc[temp, col] = 0.0
for c in col:
    for val in [True, False]:
        temp = _input1['VIP'] == val
        k = _input1[temp].mean()
        _input1.loc[temp, c] = _input1.loc[temp, c].fillna(k)
temp1 = _input0['CryoSleep'] == True
_input0.loc[temp1, col] = 0.0
for c in col:
    for val in [True, False]:
        temp1 = _input0['VIP'] == val
        k = _input0[temp1].mean()
        _input0.loc[temp1, c] = _input0.loc[temp1, c].fillna(k)
import seaborn as sb
sb.countplot(data=_input1, x='VIP', hue='HomePlanet')
sb.boxplot(_input1['Age'])
temp = _input1[_input1['Age'] < 63]['Age'].mean()
_input1['Age'] == _input1['Age'].fillna(temp)
sb.countplot(data=_input1, x='Transported', hue='CryoSleep')
_input1.isnull().sum()
_input0.isnull().sum()
for col in _input1.columns:
    if _input1[col].isnull().sum() == 0:
        continue
    if _input1[col].dtype == object or _input1[col].dtype == bool:
        _input1[col] = _input1[col].fillna(_input1[col].mode()[0])
    else:
        _input1[col] = _input1[col].fillna(_input1[col].mean())
_input1.isnull().sum().sum()
for col in _input0.columns:
    if _input0[col].isnull().sum() == 0:
        continue
    if _input0[col].dtype == object or _input0[col].dtype == bool:
        _input0[col] = _input0[col].fillna(_input0[col].mode()[0])
    else:
        _input0[col] = _input0[col].fillna(_input0[col].mean())
_input0.isnull().sum().sum()
_input0.shape
_input1.head()
new = _input1['PassengerId'].str.split('_', n=1, expand=True)
_input1['RoomNo'] = new[0].astype(int)
_input1['PassengerNo'] = new[1].astype(int)
_input1 = _input1.drop(['PassengerId', 'Name'], axis=1, inplace=False)
new1 = _input0['PassengerId'].str.split('_', n=1, expand=True)
_input0['RoomNo'] = new1[0].astype(int)
_input0['PassengerNo'] = new1[1].astype(int)
_input0.shape
new = _input1['RoomNo']
for i in range(_input1.shape[0]):
    temp = new == new[i]
    _input1['PassengerNo'][i] = temp.sum()
new1 = _input0['RoomNo']
for i in range(_input0.shape[0]):
    temp = new1 == new1[i]
    _input0['PassengerNo'][i] = temp.sum()
_input0.shape
_input1 = _input1.drop(['RoomNo'], axis=1, inplace=False)
sb.countplot(data=_input1, x='PassengerNo', hue='VIP')
_input0 = _input0.drop(['RoomNo'], axis=1, inplace=False)
sb.countplot(data=_input0, x='PassengerNo', hue='VIP')
new1 = _input1['Cabin'].str.split('/', n=2, expand=True)
new['F1'] = new[0]
_input1['F2'] = new1[1].astype(int)
_input1['F3'] = new1[2]
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
new1_test = _input0['Cabin'].str.split('/', n=2, expand=True)
new1['F1'] = new1[0]
_input0['F2'] = new1_test[1].astype(int)
_input0['F3'] = new1_test[2]
_input0 = _input0.drop(['Cabin'], axis=1, inplace=False)
_input1.head()
_input1.shape
_input0.head()
_input0.shape
_input1['LeasureBill'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['LeasureBill'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
LabelEncoder = preprocessing.LabelEncoder
for col in _input1.columns:
    if _input1[col].dtype == object:
        le = LabelEncoder()
        _input1[col] = le.fit_transform(_input1[col])
    if _input1[col].dtype == 'bool':
        _input1[col] = _input1[col].astype(int)
_input1.head()
for col in _input0.columns:
    if _input0[col].dtype == object:
        le = LabelEncoder()
        _input0[col] = le.fit_transform(_input0[col])
    if _input0[col].dtype == 'bool':
        _input0[col] = _input0[col].astype(int)
_input0.head()
features = _input1.drop(['Transported'], axis=1)
target = _input1.Transported
(X_train, X_val, Y_train, Y_val) = train_test_split(features, target, test_size=0.1, random_state=22)
(X_train.shape, X_val.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
from sklearn.metrics import roc_auc_score as ras
models = XGBClassifier()