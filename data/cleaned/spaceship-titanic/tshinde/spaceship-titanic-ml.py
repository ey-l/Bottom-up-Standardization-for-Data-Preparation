import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset
dataset.info()
dataset.isnull().sum()
x = dataset.iloc[:, :-1]
x
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
for D1 in dataset.columns:
    if dataset[D1].dtype == object:
        dataset[D1] = dataset[D1].astype(str)
        X1 = dataset[D1].unique()
        X2 = X1[-2]
        dataset[D1] = dataset[D1].fillna(X2)
        dataset[D1] = l1.fit_transform(dataset[D1])
    elif dataset[D1].dtype == float or dataset[D1].dtype == int:
        m1 = dataset[D1].mean()
        m1 = round(m1)
        dataset[D1] = dataset[D1].fillna(m1)
print(dataset)
del dataset['Transported']
dataset
data = pd.read_csv('data/input/spaceship-titanic/test.csv')
data
for D1 in data.columns:
    if data[D1].dtype == object:
        data[D1] = data[D1].astype(str)
        X1 = data[D1].unique()
        X2 = X1[-2]
        data[D1] = data[D1].fillna(X2)
        data[D1] = l1.fit_transform(data[D1])
    elif data[D1].dtype == float or data[D1].dtype == int:
        m1 = data[D1].mean()
        m1 = round(m1)
        data[D1] = data[D1].fillna(m1)
print(data)
data
dataset.info()
data.info()
dataset1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset1
from sklearn.preprocessing import LabelEncoder
l2 = LabelEncoder()
dataset1['Transported'] = l2.fit_transform(dataset1['Transported'])
dataset1['Transported']
x = dataset.iloc[:, :].values
y = dataset1.iloc[:, -1].values
x
y
data1 = dataset.iloc[:, :].values
from sklearn.preprocessing import StandardScaler
s1 = StandardScaler()
x = s1.fit_transform(x)
data1 = s1.fit_transform(data1)
data.shape
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=9)
from sklearn.linear_model import LogisticRegression
l3 = LogisticRegression()