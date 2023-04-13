import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', na_values='?')
_input1.shape
_input1.head()
_input1.describe()
_input1.nunique()
_input1.info()
_input1 = _input1.drop_duplicates(keep='first', inplace=False)
print(_input1.shape)
_input1.isna().sum()
_input1.dtypes
_input1.columns
_input1['PassengerId'] = _input1['PassengerId'].str.replace('_', '.')
_input1['PassengerId'] = _input1['PassengerId'].astype('float')
_input1['Transported'].value_counts()
sns.countplot(_input1['Transported'])
train_df1 = _input1.drop(['Cabin', 'Name'], axis=1)
train_df1.shape
train_df1.columns
cat_cols = train_df1.select_dtypes(include='object').columns
for col in cat_cols:
    train_df1[col] = train_df1[col].astype('category')
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
num_cols = ['PassengerId', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X = train_df1.drop(['Transported'], axis=1)
y = train_df1['Transported']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
y_train.value_counts(normalize=True) * 100
y_test.value_counts(normalize=True) * 100
df_cat_train = X_train[cat_cols]
df_cat_test = X_test[cat_cols]
print(df_cat_train.shape)
print(df_cat_test.shape)
df_num_train = X_train[num_cols]
df_num_test = X_test[num_cols]
print(df_num_train.shape)
print(df_num_test.shape)
from sklearn.impute import SimpleImputer
cat_imp = SimpleImputer(strategy='most_frequent')