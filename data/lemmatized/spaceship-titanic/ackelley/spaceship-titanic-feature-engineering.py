import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input1.head())
_input1.describe()
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for feature in cat_features:
    data = _input1[str(feature)].value_counts()
    print(data)
plt.figure(figsize=(20, 20))
na_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
na_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
for (idx, feature) in enumerate(na_features):
    plt.subplot(2, 3, (idx + 1, idx + 1))
    sns.kdeplot(_input1[str(feature)], color=na_colors[idx])
print(_input1.isnull().sum())
print(_input1.info())
X = _input1.drop(labels=['PassengerId', 'Name', 'Transported'], axis=1)
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'IsAdult', 'Deck', 'Number', 'Side']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalAmenities', 'GroupSize']
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
cat_imputer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
num_imputer = SimpleImputer(strategy='median')
preprocessor = ColumnTransformer(transformers=[('num', num_imputer, num_cols), ('cat', cat_imputer, cat_cols)])
is_adult = np.where(X['Age'] < 18, 'Child', 'Adult')
X['IsAdult'] = is_adult
groups = X['Cabin'].value_counts().rename_axis('Cabin').reset_index(name='GroupSize')
X = X.merge(groups, on='Cabin', how='left')
cabins = X['Cabin'].str.split(pat='/', expand=True)
cabins = cabins.rename(columns={0: 'Deck', 1: 'Number', 2: 'Side'})
X = pd.concat([X, cabins], axis=1)
X = X.drop(labels=['Cabin'], axis=1)
X['Age'] = sklearn.preprocessing.minmax_scale(X['Age'])
plt.figure()
sns.kdeplot(X['Age'], color='red')
X['GroupSize'] = sklearn.preprocessing.minmax_scale(X['GroupSize'])
plt.figure()
sns.kdeplot(X['GroupSize'], color='red')
total_amenities = np.array(X['RoomService']) + np.array(X['FoodCourt']) + np.array(X['ShoppingMall']) + np.array(X['Spa']) + np.array(X['VRDeck'])
X['TotalAmenities'] = total_amenities
from sklearn.preprocessing import MaxAbsScaler
ma = MaxAbsScaler()
X.loc[:, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalAmenities']] = ma.fit_transform(X.loc[:, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalAmenities']])
amenity_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalAmenities']
plt.figure(figsize=(20, 20))
for (idx, feature) in enumerate(amenity_features):
    plt.subplot(2, 3, (idx + 1, idx + 1))
    sns.kdeplot(X[str(feature)], color=na_colors[idx])
X.head()
X = pd.concat([X.loc[:, num_cols], X.loc[:, cat_cols]], axis=1)
y = _input1['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])