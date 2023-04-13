import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
_input1.info()
_input1 = _input1.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=False)
_input1
dest = _input1['Destination'].str.split('-', n=1, expand=True)
_input1['Destination_Planet'] = dest[0]
_input1['Platform'] = dest[1]
_input1 = _input1.drop('Destination', axis=1, inplace=False)
cat = []
num = []
for i in _input1.columns:
    if _input1[i].dtypes == object:
        cat.append(i)
        _input1[i].value_counts().plot(kind='pie', autopct='%.2f', cmap='rainbow_r')
    else:
        num.append(i)
        sns.distplot(_input1[i])
data_proc = pd.get_dummies(_input1, columns=cat, drop_first=True)
data_proc
data_proc = data_proc.replace(to_replace=False, value=0, inplace=False)
data_proc = data_proc.replace(to_replace=True, value=1, inplace=False)
data_proc
data_proc.isnull().values.any()
for i in data_proc.columns:
    if data_proc[i].isnull().values.any():
        data_proc[i] = data_proc[i].fillna(data_proc[i].mean(), axis=0, inplace=False)
_input0
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
dest = _input0['Destination'].str.split('-', n=1, expand=True)
_input0['Destination_Planet'] = dest[0]
_input0['Platform'] = dest[1]
_input0 = _input0.drop('Destination', axis=1, inplace=False)
cat_test = []
num_test = []
for i in _input0.columns:
    if _input0[i].dtypes == object:
        cat_test.append(i)
    else:
        num_test.append(i)
test_proc = pd.get_dummies(_input0, columns=cat_test, drop_first=True)
for i in test_proc.columns:
    if test_proc[i].isnull().values.any():
        test_proc[i] = test_proc[i].fillna(test_proc[i].mean(), axis=0, inplace=False)
data_proc['Transported'].value_counts()
(train, valid) = np.split(data_proc.sample(frac=1), [int(0.8 * len(data_proc))])

def pre(dataframe, test=False):
    sc = StandardScaler()
    if not _input0:
        x = dataframe.drop('Transported', axis=1)
        y = dataframe['Transported']
        X = sc.fit_transform(x)
        X = pd.DataFrame(X)
        return (X, y)
    else:
        x = dataframe
        X = sc.fit_transform(x)
        X = pd.DataFrame(X)
        return X
(X_train, Y_train) = pre(train)
(X_valid, Y_valid) = pre(valid)
test_proc = pre(test_proc, test=True)
model_0 = KNeighborsClassifier()