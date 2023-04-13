import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input0)
_input0.describe()
(_input0['Age'] == 0).sum()
_input0['Age'] = _input0['Age'].replace(0, np.nan)
(_input0['Age'] == 0).sum()
_input0.isnull().sum()
_input0.isnull().sum() / len(_input0)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(_input1)
_input1.describe()
(_input1['Age'] == 0).sum()
_input1['Age'] = _input1['Age'].replace(0, np.nan)
(_input1['Age'] == 0).sum()
_input1.isnull().sum()
_input1.isnull().sum() / len(_input1)
_input1 = _input1.drop(['Name'], axis=1)
_input1.head()
_input1.info()
_input0 = _input0.drop(['Name'], axis=1)
_input0.head()
_input0.info()
_input1[['Deck', 'DeckNum', 'Side']] = _input1.Cabin.str.split('/', expand=True)
_input0[['Deck', 'DeckNum', 'Side']] = _input0.Cabin.str.split('/', expand=True)
_input1 = _input1.drop(['Cabin'], axis=1)
_input0 = _input0.drop(['Cabin'], axis=1)
_input1.head()
_input0.head()
y = _input1.Transported
x_train_dataset = _input1.drop(['PassengerId', 'Transported'], axis=1)
x_test_dataset = _input0.drop(['PassengerId'], axis=1)
print(x_train_dataset.shape)
print(y.shape)
print(x_test_dataset.shape)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
x_train_dataset = x_train_dataset.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=5)
x_train_tra = imputer.fit_transform(x_train_dataset)
x_train_tra = pd.DataFrame(x_train_tra)
print(x_train_tra)
x_train_tra.columns = x_train_dataset.columns.values
x_train_tra.head()
x_train_tra.isnull().sum()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
x_test_dataset = x_test_dataset.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
imputer = KNNImputer(n_neighbors=3)
x_test_tra = imputer.fit_transform(x_test_dataset)
x_test_tra = pd.DataFrame(x_test_tra)
print(x_test_tra)
x_test_tra.isnull().sum()
x_test_tra.columns = x_test_dataset.columns.values
x_test_tra.head()
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
x_train_tra = scaler.fit_transform(x_train_tra)
x_test_tra = scaler.fit_transform(x_test_tra)
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
cv = KFold(n_splits=5, random_state=0, shuffle=True)
model = LogisticRegression()
scores = cross_val_score(model, x_train_tra, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
from sklearn.linear_model import LogisticRegressionCV