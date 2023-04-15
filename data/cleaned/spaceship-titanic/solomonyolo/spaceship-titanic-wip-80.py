import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_csv = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(data_csv.head())
data_csv['FirstName'] = data_csv['Name'].str.split(' ', expand=True)[0]
data_csv['Family'] = data_csv['Name'].str.split(' ', expand=True)[1]
test_data['FirstName'] = test_data['Name'].str.split(' ', expand=True)[0]
test_data['Family'] = test_data['Name'].str.split(' ', expand=True)[1]
data_csv['CabinDeck'] = data_csv['Cabin'].str.split('/', expand=True)[0]
data_csv['CabinNum'] = data_csv['Cabin'].str.split('/', expand=True)[1]
data_csv['CabinSide'] = data_csv['Cabin'].str.split('/', expand=True)[2]
test_data['CabinDeck'] = test_data['Cabin'].str.split('/', expand=True)[0]
test_data['CabinNum'] = test_data['Cabin'].str.split('/', expand=True)[1]
test_data['CabinSide'] = test_data['Cabin'].str.split('/', expand=True)[2]
print(data_csv.head())
print(data_csv.isnull().sum())
data_csv = data_csv.drop(['Name', 'Cabin'], axis=1)
test_data = test_data.drop(['Name', 'Cabin'], axis=1)
print(data_csv.head())
data_csv.info()
data_csv['CabinNum'] = pd.to_numeric(data_csv['CabinNum'])
test_data['CabinNum'] = pd.to_numeric(test_data['CabinNum'])
data_csv.info()
numeric_data = [column for column in data_csv.select_dtypes(['int', 'float'])]
categoric_data = [column for column in data_csv.select_dtypes(exclude=['int', 'float'])]
categoric_data1 = [column for column in test_data.select_dtypes(exclude=['int', 'float'])]
print(categoric_data)
for i in numeric_data:
    data_csv[i].fillna(data_csv[i].median(), inplace=True)
    test_data[i].fillna(test_data[i].median(), inplace=True)
for i in categoric_data:
    data_csv[i].fillna(data_csv[i].value_counts().index[0], inplace=True)
for i in categoric_data1:
    test_data[i].fillna(test_data[i].value_counts().index[0], inplace=True)
print(data_csv.isnull().sum())
print(data_csv.corr())
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
data_csv[categoric_data] = encoder.fit_transform(data_csv[categoric_data])
test_data[categoric_data1] = encoder.fit_transform(test_data[categoric_data1])
print(data_csv[categoric_data])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_csv[numeric_data] = scaler.fit_transform(data_csv[numeric_data])
test_data[numeric_data] = scaler.fit_transform(test_data[numeric_data])
print(data_csv[numeric_data])
print(data_csv.corr())
X = data_csv.drop('Transported', axis=1)
Y = data_csv['Transported']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=0.1, max_depth=5, colsample_bytree=0.8, seed=27)