import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1['group'] = _input1.apply(lambda row: row['PassengerId'][-1], axis=1)
_input1['deck'] = _input1['Cabin'].str.split('/', expand=True)[0]
_input1['side'] = _input1['Cabin'].str.split('/', expand=True)[2]
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
from sklearn.impute import KNNImputer
object_features = _input1.columns[_input1.dtypes == 'object']
num_features = _input1.columns[_input1.dtypes != 'object']
for feature in object_features:
    _input1[feature] = _input1[feature].fillna(_input1[feature].mode()[0], inplace=False)
imputer = KNNImputer(n_neighbors=5)
_input1[num_features] = imputer.fit_transform(_input1[num_features])
_input1.isnull().sum()
_input1['expend'] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
_input1['Age_band'] = pd.qcut(_input1['Age'], q=10, labels=range(10)).astype('float')
_input1['VIP'] = _input1['VIP'].replace({False: 0, True: 1}, inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace({False: 0, True: 1}, inplace=False)
new_object_features = _input1.columns[_input1.dtypes == 'object']
for feature in new_object_features:
    _input1 = _input1.join(pd.get_dummies(_input1[feature]))
    _input1 = _input1.drop([feature], axis=1, inplace=False)

def data_preprocess(data):
    _input1['group'] = _input1.apply(lambda row: row['PassengerId'][-1], axis=1)
    _input1['deck'] = _input1['Cabin'].str.split('/', expand=True)[0]
    _input1['side'] = _input1['Cabin'].str.split('/', expand=True)[2]
    _input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
    object_features = _input1.columns[_input1.dtypes == 'object']
    num_features = _input1.columns[_input1.dtypes != 'object']
    for feature in object_features:
        _input1[feature] = _input1[feature].fillna(_input1[feature].mode()[0], inplace=False)
    imputer = KNNImputer(n_neighbors=5)
    _input1[num_features] = imputer.fit_transform(_input1[num_features])
    _input1['expend'] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    _input1['Age_band'] = pd.qcut(_input1['Age'], q=10, labels=range(10)).astype('float')
    _input1['VIP'] = _input1['VIP'].replace({False: 0, True: 1}, inplace=False)
    _input1['CryoSleep'] = _input1['CryoSleep'].replace({False: 0, True: 1}, inplace=False)
    new_object_features = _input1.columns[_input1.dtypes == 'object']
    for feature in new_object_features:
        _input1 = _input1.join(pd.get_dummies(_input1[feature]))
        _input1 = _input1.drop([feature], axis=1, inplace=False)
    return _input1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
(X_train, X_test, y_train, y_test) = train_test_split(_input1.drop(['Transported'], axis=1), _input1['Transported'], test_size=0.3, random_state=1, stratify=_input1['Transported'])
(X, Y) = (_input1.drop(['Transported'], axis=1), _input1['Transported'])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()