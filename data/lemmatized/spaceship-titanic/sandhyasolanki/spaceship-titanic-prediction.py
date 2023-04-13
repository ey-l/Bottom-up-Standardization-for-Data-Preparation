import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.shape
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input0.shape
_input1.isnull().sum()
_input0.isnull().sum()
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
categorical_feature = [feature for feature in _input1.columns if _input1[feature].dtypes == 'O']
categorical_feature
_input1.info()
sns.countplot(x='Transported', data=_input1)
sns.countplot(x='Transported', hue='VIP', data=_input1)
sns.countplot(x='Transported', hue='Destination', data=_input1)
sns.countplot(x='Transported', hue='HomePlanet', data=_input1)
numerical_with_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1 and _input1[feature].dtypes != 'O' and (feature not in ['Transported'])]
for feature in numerical_with_nan:
    print(feature, np.round(_input1[feature].isnull().mean(), 4), '% missing values')
for feature in numerical_with_nan:
    median_values = _input1[feature].median()
    _input1[feature] = _input1[feature].fillna(median_values, inplace=False)
_input1[numerical_with_nan].isnull().sum()
categorical_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1 and _input1[feature].dtypes == 'O']
for feature in categorical_nan:
    print(feature, np.round(_input1[feature].isnull().mean(), 4), '% missing values')
_input1['HomePlanet'].value_counts()
_input1['CryoSleep'].value_counts()
_input1['Cabin'].value_counts()
_input1['Destination'].value_counts()
_input1['VIP'].value_counts()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input1['VIP'] = _input1['VIP'].fillna('False')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('False')
_input1.isnull().sum()
_input1['Cabin'].unique
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input1.isnull().sum()
passengerid = _input1['PassengerId']
passengerid
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
data_object = [feature for feature in _input1.columns if _input1[feature].dtypes == 'O']
print(data_object)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for features in data_object:
    _input1[features] = le.fit_transform(_input1[features].astype(str))
_input1.info()
numerical_with_nan = [feature for feature in _input0.columns if _input0[feature].isnull().sum() > 1 and _input0[feature].dtypes != 'O']
for feature in numerical_with_nan:
    print(feature, np.round(_input0[feature].isnull().mean(), 4), '% missing values')
for feature in numerical_with_nan:
    median_values = _input0[feature].median()
    _input0[feature] = _input0[feature].fillna(median_values, inplace=False)
_input0[numerical_with_nan].isnull().sum()
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e')
_input0['VIP'] = _input0['VIP'].fillna('False')
_input0['CryoSleep'] = _input0['CryoSleep'].fillna('False')
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input0.isnull().sum()
passengerid = _input0['PassengerId']
_input0 = _input0.drop('PassengerId', axis=1, inplace=False)
data_object = _input0.select_dtypes(include='object').columns
print(data_object)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for features in data_object:
    _input0[features] = le.fit_transform(_input0[features].astype(str))
_input0.info()
_input1.head(25)
feature_scale = [feature for feature in _input1.columns if feature not in ['PassengerId', 'Transported']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()