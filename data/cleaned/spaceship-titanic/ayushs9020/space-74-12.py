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
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
data
data.info()
data.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
data
dest = data['Destination'].str.split('-', n=1, expand=True)
data['Destination_Planet'] = dest[0]
data['Platform'] = dest[1]
data.drop('Destination', axis=1, inplace=True)
cat = []
num = []
for i in data.columns:
    if data[i].dtypes == object:
        cat.append(i)
        data[i].value_counts().plot(kind='pie', autopct='%.2f', cmap='rainbow_r')

    else:
        num.append(i)
        sns.distplot(data[i])

data_proc = pd.get_dummies(data, columns=cat, drop_first=True)
data_proc
data_proc.replace(to_replace=False, value=0, inplace=True)
data_proc.replace(to_replace=True, value=1, inplace=True)
data_proc
data_proc.isnull().values.any()
for i in data_proc.columns:
    if data_proc[i].isnull().values.any():
        data_proc[i].fillna(data_proc[i].mean(), axis=0, inplace=True)
test
test.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
dest = test['Destination'].str.split('-', n=1, expand=True)
test['Destination_Planet'] = dest[0]
test['Platform'] = dest[1]
test.drop('Destination', axis=1, inplace=True)
cat_test = []
num_test = []
for i in test.columns:
    if test[i].dtypes == object:
        cat_test.append(i)
    else:
        num_test.append(i)
test_proc = pd.get_dummies(test, columns=cat_test, drop_first=True)
for i in test_proc.columns:
    if test_proc[i].isnull().values.any():
        test_proc[i].fillna(test_proc[i].mean(), axis=0, inplace=True)
data_proc['Transported'].value_counts()
(train, valid) = np.split(data_proc.sample(frac=1), [int(0.8 * len(data_proc))])

def pre(dataframe, test=False):
    sc = StandardScaler()
    if not test:
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