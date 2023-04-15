import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
dataset1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset1
dataset = dataset.iloc[:, :-1]
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
for col in dataset.columns:
    if dataset[col].dtype == object:
        dataset[col] = dataset[col].astype(str)
        d1 = dataset[col].unique()
        d2 = d1[-2]
        dataset[col] = dataset[col].fillna(d2)
        dataset[col] = lbl.fit_transform(dataset[col])
    elif dataset[col].dtype == float or dataset[col].dtype == int:
        m1 = dataset[col].mean()
        m1 = round(m1)
        dataset[col] = dataset[col].fillna(m1)
print(dataset)
dataset
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
dataset.info()
test_data.info()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
dataset1['Transported'] = l1.fit_transform(dataset1['Transported'])
dataset1['Transported']
x = dataset.iloc[:, :].values
y = dataset1.iloc[:, -1].values
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