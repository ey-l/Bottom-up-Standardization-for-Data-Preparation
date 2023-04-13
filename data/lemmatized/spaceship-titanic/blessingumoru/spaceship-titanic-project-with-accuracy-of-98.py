import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv', index_col='PassengerId')
_input1.isna().sum()
_input1.info()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('None', inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna('None', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('None', inplace=False)

def fixNumVal(data):
    for cols in data.columns:
        if data[cols].dtype == 'float64' or data[cols].dtype == 'int64':
            data[cols] = data[cols].fillna(data[cols].mean(), inplace=False)
        else:
            data[cols] = data[cols].fillna(data[cols].mode()[0], inplace=False)
fixNumVal(_input1)
fixNumVal(_input0)
_input1.isna().sum()
from sklearn.preprocessing import LabelEncoder
objectColumns = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'Name', 'VIP']
encoder = LabelEncoder()

def enconding(dataset):
    for col in objectColumns:
        dataset[col] = encoder.fit_transform(dataset[col])
enconding(_input1)
enconding(_input0)
_input1['Transported'] = encoder.fit_transform(_input1['Transported'])
_input2['Transported'] = encoder.fit_transform(_input2['Transported'])
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer, StandardScaler
array = _input1.values
X_arr = array[:, 0:12]
Y = array[:, 12]
demo = MinMaxScaler(feature_range=(0, 1))
rescaledX = demo.fit_transform(X=X_arr)
rescaledXdf = pd.DataFrame(rescaledX)
rescaledXdf.columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name']
rescaledXdf['Transported'] = Y
rescaledXdf
array = rescaledXdf.values
X = array[0:, 0:12]
Y = array[:, 12]