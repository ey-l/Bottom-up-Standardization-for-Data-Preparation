import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.tail()
_input1.shape
_input0.head()
_input0.isnull().sum()
pd.isnull(_input1).sum()
print(_input1.dtypes)
import missingno as msno
msno.matrix(_input1)
_input1.columns
count = (_input1[['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported']] == 0).sum()
print('Count of zeros in data: ', count)
_input1.shape
df = _input1.copy()
df_test = _input0.copy()
df = df.drop(df[['PassengerId', 'Name']], axis=1, inplace=False)
df_test = df_test.drop(df_test[['PassengerId', 'Name']], axis=1, inplace=False)
df_test.head()
df.head()
for i in list(df.columns):
    if df[i].dtypes == 'float64':
        df[i] = df[i].fillna(df[i].mean(), inplace=False)
    else:
        df[i] = df[i].fillna(df[i].mode()[0], inplace=False)
print(df.isnull().sum())
for i in list(df_test.columns):
    if df_test[i].dtypes == 'float64':
        df_test[i] = df_test[i].fillna(df_test[i].mean(), inplace=False)
    else:
        df_test[i] = df_test[i].fillna(df_test[i].mode()[0], inplace=False)
print(df_test.isnull().sum())
df.describe()
cont_table = pd.crosstab(df['Transported'], df['HomePlanet'])
cont_table
import scipy.stats
scipy.stats.chi2_contingency(cont_table, correction=True)
cont_table2 = pd.crosstab(df['Transported'], df['CryoSleep'])
cont_table2
import scipy.stats
scipy.stats.chi2_contingency(cont_table2, correction=True)
df.columns
import seaborn as sns
sns.set(style='whitegrid')
ax = sns.barplot(y='Age', x='HomePlanet', hue='Transported', data=df)
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
df['Transported'] = number.fit_transform(df['Transported'].astype('str'))
df['HomePlanet'] = number.fit_transform(df['HomePlanet'].astype('str'))
df_test['HomePlanet'] = number.fit_transform(df_test['HomePlanet'].astype('str'))
df['CryoSleep'] = number.fit_transform(df['CryoSleep'].astype('str'))
df_test['CryoSleep'] = number.fit_transform(df_test['CryoSleep'].astype('str'))
df['VIP'] = number.fit_transform(df['VIP'].astype('str'))
df_test['VIP'] = number.fit_transform(df_test['VIP'].astype('str'))
df['Destination'] = number.fit_transform(df['Destination'].astype('str'))
df_test['Destination'] = number.fit_transform(df_test['Destination'].astype('str'))
df['Cabin'] = number.fit_transform(df['Cabin'].astype('str'))
df_test['Cabin'] = number.fit_transform(df_test['Cabin'].astype('str'))
df_test.head()
df_test.shape
df.head()
df.shape
X = df.drop('Transported', axis=1)
y = df['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
(X_train.shape, X_test.shape)
import xgboost as xgb
XGBClassifier = xgb.XGBClassifier(max_depth=8, learning_rate=0.001, n_estimators=10000, objective='binary:logistic', gamma=0.64, max_delta_step=3, min_child_weight=7, subsample=0.7, colsample_bytree=0.8, n_jobs=-1)
import datetime
start = datetime.datetime.now()