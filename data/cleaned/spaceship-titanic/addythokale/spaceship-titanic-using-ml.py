import pandas as pd
import numpy as np
dataset = dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset
dataset.shape
dataset.info()
dataset.isnull().sum()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
for col in dataset.columns:
    if dataset[col].dtype == object:
        dataset[col] = dataset[col].astype(str)
        d1 = dataset[col].unique()
        d2 = d1[-2]
        dataset[col] = dataset[col].fillna(d2)
        dataset[col] = l1.fit_transform(dataset[col])
    elif dataset[col].dtype == float or dataset[col].dtype == int:
        m1 = dataset[col].mean()
        m1 = round(m1)
        dataset[col] = dataset[col].fillna(m1)
print(dataset)
dataset
dataset.info()
dataset['Transported'] = l1.fit_transform(dataset['Transported'])
dataset['Transported']
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x = std.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=30)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()