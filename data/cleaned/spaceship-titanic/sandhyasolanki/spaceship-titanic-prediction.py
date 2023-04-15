import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df.head()
train_df.shape
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df.head()
test_df.shape
train_df.isnull().sum()
test_df.isnull().sum()
train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)
categorical_feature = [feature for feature in train_df.columns if train_df[feature].dtypes == 'O']
categorical_feature
train_df.info()
sns.countplot(x='Transported', data=train_df)
sns.countplot(x='Transported', hue='VIP', data=train_df)
sns.countplot(x='Transported', hue='Destination', data=train_df)
sns.countplot(x='Transported', hue='HomePlanet', data=train_df)
numerical_with_nan = [feature for feature in train_df.columns if train_df[feature].isnull().sum() > 1 and train_df[feature].dtypes != 'O' and (feature not in ['Transported'])]
for feature in numerical_with_nan:
    print(feature, np.round(train_df[feature].isnull().mean(), 4), '% missing values')
for feature in numerical_with_nan:
    median_values = train_df[feature].median()
    train_df[feature].fillna(median_values, inplace=True)
train_df[numerical_with_nan].isnull().sum()
categorical_nan = [feature for feature in train_df.columns if train_df[feature].isnull().sum() > 1 and train_df[feature].dtypes == 'O']
for feature in categorical_nan:
    print(feature, np.round(train_df[feature].isnull().mean(), 4), '% missing values')
train_df['HomePlanet'].value_counts()
train_df['CryoSleep'].value_counts()
train_df['Cabin'].value_counts()
train_df['Destination'].value_counts()
train_df['VIP'].value_counts()
train_df['HomePlanet'] = train_df['HomePlanet'].fillna('Earth')
train_df['Destination'] = train_df['Destination'].fillna('TRAPPIST-1e')
train_df['VIP'] = train_df['VIP'].fillna('False')
train_df['CryoSleep'] = train_df['CryoSleep'].fillna('False')
train_df.isnull().sum()
train_df['Cabin'].unique
train_df.drop('Cabin', axis=1, inplace=True)
train_df.isnull().sum()
passengerid = train_df['PassengerId']
passengerid
train_df.drop('PassengerId', axis=1, inplace=True)
data_object = [feature for feature in train_df.columns if train_df[feature].dtypes == 'O']
print(data_object)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for features in data_object:
    train_df[features] = le.fit_transform(train_df[features].astype(str))
train_df.info()
numerical_with_nan = [feature for feature in test_df.columns if test_df[feature].isnull().sum() > 1 and test_df[feature].dtypes != 'O']
for feature in numerical_with_nan:
    print(feature, np.round(test_df[feature].isnull().mean(), 4), '% missing values')
for feature in numerical_with_nan:
    median_values = test_df[feature].median()
    test_df[feature].fillna(median_values, inplace=True)
test_df[numerical_with_nan].isnull().sum()
test_df['HomePlanet'] = test_df['HomePlanet'].fillna('Earth')
test_df['Destination'] = test_df['Destination'].fillna('TRAPPIST-1e')
test_df['VIP'] = test_df['VIP'].fillna('False')
test_df['CryoSleep'] = test_df['CryoSleep'].fillna('False')
test_df.drop('Cabin', axis=1, inplace=True)
test_df.isnull().sum()
passengerid = test_df['PassengerId']
test_df.drop('PassengerId', axis=1, inplace=True)
data_object = test_df.select_dtypes(include='object').columns
print(data_object)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for features in data_object:
    test_df[features] = le.fit_transform(test_df[features].astype(str))
test_df.info()
train_df.head(25)
feature_scale = [feature for feature in train_df.columns if feature not in ['PassengerId', 'Transported']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()