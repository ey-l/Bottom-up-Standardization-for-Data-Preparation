import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.dtypes
_input1 = _input1.set_index('PassengerId')
_input1.shape[0]
_input1.isnull().sum()
_input1 = _input1.drop(columns=['Name'], axis=1)
import seaborn as sns
sns.histplot(x=_input1['Age'], kde=True)
_input1['Age'].skew()
corr = _input1.corr()[['Transported']]
sns.heatmap(corr, annot=True)
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(value=int(_input1['Age'].mean()), inplace=False)
_input1.isnull().sum()
_input1 = _input1.fillna(value='Missing')
_input1.isnull().sum()
_input1['Deck'] = ''
_input1['Side'] = ''
for rowNum in range(_input1.shape[0]):
    if _input1['Cabin'][rowNum] == 'Missing':
        _input1['Deck'][rowNum] = 'Missing'
        _input1['Side'][rowNum] = 'Missing'
    else:
        _input1['Deck'][rowNum] = _input1['Cabin'][rowNum][0]
        _input1['Side'][rowNum] = _input1['Cabin'][rowNum].split('/')[2]
_input1 = _input1.drop(columns=['Cabin'], axis=1)
_input1.head()
_input1['AgeBin'] = pd.cut(_input1['Age'], bins=8, labels=False)
_input1 = _input1.drop(columns=['Age'], axis=1)
_input1.head()
new_cols_order = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeBin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
_input1 = _input1.reindex(columns=new_cols_order)
_input1.head()
for colName in _input1.columns:
    if colName in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        next
    else:
        _input1[colName] = _input1[colName].astype('category')
_input1.dtypes
for i in _input1.columns:
    print(f'The column "{i}" has {len(_input1[i].value_counts())} unique values.')
_input1['Transported'] = _input1['Transported'].astype(int)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = sc.fit_transform(_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
_input1.head()
X_CategoricalFeatures = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeBin']
_input1 = pd.get_dummies(data=_input1, columns=X_CategoricalFeatures, drop_first=True)
_input1.head()
transportedColumn = _input1.pop('Transported')
_input1 = pd.concat([_input1, transportedColumn], axis=1)
_input1.head()
X = _input1.iloc[:, 0:-1].values
y = _input1.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier()