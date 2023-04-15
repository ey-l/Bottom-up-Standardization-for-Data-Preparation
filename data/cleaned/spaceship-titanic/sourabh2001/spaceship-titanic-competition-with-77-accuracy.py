import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.columns
train.head()
train.info()
train = train.replace({True: 1, False: 0})
test = test.replace({True: 1, False: 0})
train.info()
test.info()
msn.bar(train, color='red')

def fill_missing_vals(train, fill_missing):
    for col in fill_missing:
        train[col].fillna(train[col].median(skipna=True), inplace=True)
    return train
fill_missing_vals(train, ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
train['HomePlanet'].fillna('Z', inplace=True)
train.info()
train['HomePlanet'].unique()
train['Transported'].isnull().sum()
from sklearn.preprocessing import LabelEncoder

def label_encode(df, col):
    train[col] = train[col].astype(str)
    train[col] = LabelEncoder().fit_transform(train[col])
    return train[col]
train['HomePlanet'] = label_encode(train, 'HomePlanet')
train['Destination'] = label_encode(train, 'Destination')
train['Cabin'] = label_encode(train, 'Cabin')
train.columns
X = train[['HomePlanet', 'Cabin', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
y = train['Transported']
X.info()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()