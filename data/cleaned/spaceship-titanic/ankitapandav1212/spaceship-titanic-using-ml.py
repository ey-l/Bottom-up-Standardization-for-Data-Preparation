import numpy as np
import pandas as pd
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
data1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
data1
data = data.iloc[:, :-1]
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].astype(str)
        d1 = data[col].unique()
        d2 = d1[-2]
        data[col] = data[col].fillna(d2)
        data[col] = lbl.fit_transform(data[col])
    elif data[col].dtype == float or data[col].dtype == int:
        m1 = data[col].mean()
        m1 = round(m1)
        data[col] = data[col].fillna(m1)
print(data)
data
for col in test_data.columns:
    if test_data[col].dtype == object:
        test_data[col] = test_data[col].astype(str)
        d1 = test_data[col].unique()
        d2 = d1[-2]
        test_data[col] = test_data[col].fillna(d2)
        test_data[col] = lbl.fit_transform(test_data[col])
    elif test_data[col].dtype == float or test_data[col].dtype == int:
        m1 = test_data[col].mean()
        m1 = round(m1)
        test_data[col] = test_data[col].fillna(m1)
print(test_data)
test_data
data.info()
test_data.info()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
data1['Transported'] = l1.fit_transform(data1['Transported'])
data1['Transported']
x = data.iloc[:, :].values
y = data1.iloc[:, -1].values
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