import numpy as np
import pandas as pd
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_data
x_data = train_data.iloc[:, :-1]
x_data
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for col in train_data.columns:
    if train_data[col].dtype == object:
        train_data[col] = train_data[col].astype(str)
        d1 = train_data[col].unique()
        d2 = d1[-2]
        train_data[col] = train_data[col].fillna(d2)
        train_data[col] = lb.fit_transform(train_data[col])
    elif train_data[col].dtype == float or train_data[col].dtype == int:
        m1 = train_data[col].mean()
        m1 = round(m1)
        train_data[col] = train_data[col].fillna(m1)
print(train_data)
del train_data['Transported']
train_data
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_data
for col in test_data.columns:
    if test_data[col].dtype == object:
        test_data[col] = test_data[col].astype(str)
        d1 = test_data[col].unique()
        d2 = d1[-2]
        test_data[col] = test_data[col].fillna(d2)
        test_data[col] = lb.fit_transform(test_data[col])
    elif test_data[col].dtype == float or test_data[col].dtype == int:
        m1 = test_data[col].mean()
        m1 = round(m1)
        test_data[col] = test_data[col].fillna(m1)
print(test_data)
test_data
train_data.info()
test_data.info()
train_data1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_data1
from sklearn.preprocessing import LabelEncoder
lb1 = LabelEncoder()
train_data1['Transported'] = lb1.fit_transform(train_data1['Transported'])
train_data1['Transported']
x = train_data.iloc[:, :].values
y = train_data1.iloc[:, -1].values
y
test_std = test_data.iloc[:, :].values
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x = std.fit_transform(x)
test_std = std.fit_transform(test_std)
test_std.shape
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=11)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()