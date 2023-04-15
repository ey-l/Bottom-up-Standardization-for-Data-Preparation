import pandas as pd
import numpy as np
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
data['group'] = data.apply(lambda row: row['PassengerId'][-1], axis=1)
data['deck'] = data['Cabin'].str.split('/', expand=True)[0]
data['side'] = data['Cabin'].str.split('/', expand=True)[2]
data.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
from sklearn.impute import KNNImputer
object_features = data.columns[data.dtypes == 'object']
num_features = data.columns[data.dtypes != 'object']
for feature in object_features:
    data[feature].fillna(data[feature].mode()[0], inplace=True)
imputer = KNNImputer(n_neighbors=5)
data[num_features] = imputer.fit_transform(data[num_features])
data.isnull().sum()
data['expend'] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
data['Age_band'] = pd.qcut(data['Age'], q=10, labels=range(10)).astype('float')
data['VIP'].replace({False: 0, True: 1}, inplace=True)
data['CryoSleep'].replace({False: 0, True: 1}, inplace=True)
new_object_features = data.columns[data.dtypes == 'object']
for feature in new_object_features:
    data = data.join(pd.get_dummies(data[feature]))
    data.drop([feature], axis=1, inplace=True)

def data_preprocess(data):
    data['group'] = data.apply(lambda row: row['PassengerId'][-1], axis=1)
    data['deck'] = data['Cabin'].str.split('/', expand=True)[0]
    data['side'] = data['Cabin'].str.split('/', expand=True)[2]
    data.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
    object_features = data.columns[data.dtypes == 'object']
    num_features = data.columns[data.dtypes != 'object']
    for feature in object_features:
        data[feature].fillna(data[feature].mode()[0], inplace=True)
    imputer = KNNImputer(n_neighbors=5)
    data[num_features] = imputer.fit_transform(data[num_features])
    data['expend'] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    data['Age_band'] = pd.qcut(data['Age'], q=10, labels=range(10)).astype('float')
    data['VIP'].replace({False: 0, True: 1}, inplace=True)
    data['CryoSleep'].replace({False: 0, True: 1}, inplace=True)
    new_object_features = data.columns[data.dtypes == 'object']
    for feature in new_object_features:
        data = data.join(pd.get_dummies(data[feature]))
        data.drop([feature], axis=1, inplace=True)
    return data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
(X_train, X_test, y_train, y_test) = train_test_split(data.drop(['Transported'], axis=1), data['Transported'], test_size=0.3, random_state=1, stratify=data['Transported'])
(X, Y) = (data.drop(['Transported'], axis=1), data['Transported'])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()