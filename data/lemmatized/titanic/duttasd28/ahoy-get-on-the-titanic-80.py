from IPython import display
display.Image('https://raw.githubusercontent.com/Dutta-SD/Images_Unsplash/master/Kaggle/dorian-mongel-5Rgr_zI7pBw-unsplash.jpg', width=3000, height=500)
import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input0.head()
PassengerID = _input0.PassengerId
_input1 = _input1.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=False, axis=1)
_input0 = _input0.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=False, axis=1)
_input0.head()
print(_input1.isnull().any())
_input0.isnull().any()
y = _input1.Survived
X = _input1.drop(['Survived'], axis=1)
print(y.head())
print(X.head())
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x=y.unique(), y=y.value_counts())
sns.pairplot(data=_input1, corner=True, palette='summer')
(X_train, X_test, y_train, y_test) = (X, _input0, y, None)
X_train = X_train.reset_index(drop=True, inplace=False)
X_test = X_test.reset_index(drop=True, inplace=False)
y_train = y_train.reset_index(drop=True, inplace=False)
y_train = y_train.reset_index(drop=True, inplace=False)
X_train.info()
s = X_train.dtypes == 'object'
categorical_cols = list(s[s].index)
numerical_cols = [i for i in X_train.columns if not i in categorical_cols]
numerical_cols
from sklearn.impute import KNNImputer
nm_imputer = KNNImputer()
X_train_numerical = pd.DataFrame(nm_imputer.fit_transform(X_train[numerical_cols]), columns=numerical_cols)
X_test_numerical = pd.DataFrame(nm_imputer.transform(X_test[numerical_cols]), columns=numerical_cols)
X_train = X_train.drop(numerical_cols, axis=1)
X_test = X_test.drop(numerical_cols, axis=1)
X_train = X_train.join(X_train_numerical)
X_test = X_test.join(X_test_numerical)
X_train.isnull().any()
from sklearn.impute import SimpleImputer
nm_imputer = SimpleImputer(strategy='most_frequent')
X_train_numerical = pd.DataFrame(nm_imputer.fit_transform(X_train[categorical_cols]), columns=categorical_cols)
X_test_numerical = pd.DataFrame(nm_imputer.transform(X_test[categorical_cols]), columns=categorical_cols)
X_train = X_train.drop(categorical_cols, axis=1)
X_test = X_test.drop(categorical_cols, axis=1)
X_train = X_train.join(X_train_numerical)
X_test = X_test.join(X_test_numerical)
X_train.isnull().any()
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[categorical_cols]))
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index
num_X_train = X_train.drop(categorical_cols, axis=1)
num_X_test = X_test.drop(categorical_cols, axis=1)
X_train = num_X_train.join(OH_cols_train, how='left')
X_test = num_X_test.join(OH_cols_test, how='left')
X_train.head()
X_test.info()
X_train.info()
from sklearn.model_selection import train_test_split
(X_train_2, X_val, y_train_2, y_val) = train_test_split(X_train, y_train, test_size=0.2, random_state=10)
from tensorflow import keras
from keras import Sequential
from keras.layers import BatchNormalization, Dense
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(10,)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])