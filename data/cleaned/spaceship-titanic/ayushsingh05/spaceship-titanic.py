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
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
data.head(10)
data.shape
test_data.shape
data.info()
data.describe()
data.isnull().sum().plot.bar()
col = data.loc[:, 'RoomService':'VRDeck'].columns
data.groupby('VIP')[col].mean()
data.groupby('HomePlanet')[col].mean()
data.groupby('CryoSleep')[col].mean()
temp = data['CryoSleep'] == True
data.loc[temp, col] = 0.0
for c in col:
    for val in [True, False]:
        temp = data['VIP'] == val
        k = data[temp].mean()
        data.loc[temp, c] = data.loc[temp, c].fillna(k)
temp1 = test_data['CryoSleep'] == True
test_data.loc[temp1, col] = 0.0
for c in col:
    for val in [True, False]:
        temp1 = test_data['VIP'] == val
        k = test_data[temp1].mean()
        test_data.loc[temp1, c] = test_data.loc[temp1, c].fillna(k)
import seaborn as sb
sb.countplot(data=data, x='VIP', hue='HomePlanet')

sb.boxplot(data['Age'])
temp = data[data['Age'] < 63]['Age'].mean()
data['Age'] == data['Age'].fillna(temp)
sb.countplot(data=data, x='Transported', hue='CryoSleep')
data.isnull().sum()
test_data.isnull().sum()
for col in data.columns:
    if data[col].isnull().sum() == 0:
        continue
    if data[col].dtype == object or data[col].dtype == bool:
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())
data.isnull().sum().sum()
for col in test_data.columns:
    if test_data[col].isnull().sum() == 0:
        continue
    if test_data[col].dtype == object or test_data[col].dtype == bool:
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
    else:
        test_data[col] = test_data[col].fillna(test_data[col].mean())
test_data.isnull().sum().sum()
test_data.shape
data.head()
new = data['PassengerId'].str.split('_', n=1, expand=True)
data['RoomNo'] = new[0].astype(int)
data['PassengerNo'] = new[1].astype(int)
data.drop(['PassengerId', 'Name'], axis=1, inplace=True)
new1 = test_data['PassengerId'].str.split('_', n=1, expand=True)
test_data['RoomNo'] = new1[0].astype(int)
test_data['PassengerNo'] = new1[1].astype(int)
test_data.shape
new = data['RoomNo']
for i in range(data.shape[0]):
    temp = new == new[i]
    data['PassengerNo'][i] = temp.sum()
new1 = test_data['RoomNo']
for i in range(test_data.shape[0]):
    temp = new1 == new1[i]
    test_data['PassengerNo'][i] = temp.sum()
test_data.shape
data.drop(['RoomNo'], axis=1, inplace=True)
sb.countplot(data=data, x='PassengerNo', hue='VIP')

test_data.drop(['RoomNo'], axis=1, inplace=True)
sb.countplot(data=test_data, x='PassengerNo', hue='VIP')

new1 = data['Cabin'].str.split('/', n=2, expand=True)
new['F1'] = new[0]
data['F2'] = new1[1].astype(int)
data['F3'] = new1[2]
data.drop(['Cabin'], axis=1, inplace=True)
new1_test = test_data['Cabin'].str.split('/', n=2, expand=True)
new1['F1'] = new1[0]
test_data['F2'] = new1_test[1].astype(int)
test_data['F3'] = new1_test[2]
test_data.drop(['Cabin'], axis=1, inplace=True)
data.head()
data.shape
test_data.head()
test_data.shape
data['LeasureBill'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
test_data['LeasureBill'] = test_data['RoomService'] + test_data['FoodCourt'] + test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck']
LabelEncoder = preprocessing.LabelEncoder
for col in data.columns:
    if data[col].dtype == object:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    if data[col].dtype == 'bool':
        data[col] = data[col].astype(int)
data.head()
for col in test_data.columns:
    if test_data[col].dtype == object:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col])
    if test_data[col].dtype == 'bool':
        test_data[col] = test_data[col].astype(int)
test_data.head()
features = data.drop(['Transported'], axis=1)
target = data.Transported
(X_train, X_val, Y_train, Y_val) = train_test_split(features, target, test_size=0.1, random_state=22)
(X_train.shape, X_val.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
from sklearn.metrics import roc_auc_score as ras
models = XGBClassifier()