import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
trainDf = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
testDf = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
submit = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv', index_col='PassengerId')
trainDf.isna().sum()
trainDf.info()
trainDf['HomePlanet'].fillna('None', inplace=True)
trainDf['Cabin'].fillna('None', inplace=True)
trainDf['Destination'].fillna('None', inplace=True)

def fixNumVal(data):
    for cols in data.columns:
        if data[cols].dtype == 'float64' or data[cols].dtype == 'int64':
            data[cols].fillna(data[cols].mean(), inplace=True)
        else:
            data[cols].fillna(data[cols].mode()[0], inplace=True)
fixNumVal(trainDf)
fixNumVal(testDf)
trainDf.isna().sum()
from sklearn.preprocessing import LabelEncoder
objectColumns = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'Name', 'VIP']
encoder = LabelEncoder()

def enconding(dataset):
    for col in objectColumns:
        dataset[col] = encoder.fit_transform(dataset[col])
enconding(trainDf)
enconding(testDf)
trainDf['Transported'] = encoder.fit_transform(trainDf['Transported'])
submit['Transported'] = encoder.fit_transform(submit['Transported'])
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer, StandardScaler
array = trainDf.values
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