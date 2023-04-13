import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
_input0
_input1.head()
_input1.describe()
_input1.info()
_input1.isna().any()
_input1.isna().sum()
_input1.nunique()
_input1.shape
_input1['Transported'].value_counts()
sns.countplot(x='Transported', data=_input1)
_input1.isna().sum()
_input1 = _input1.drop(['Name', 'PassengerId', 'Cabin'], inplace=False, axis=1)
_input1['HomePlanet'].value_counts()
_input1.isna().sum()
_input1.dtypes
_input1['Destination'].value_counts()
_input1 = _input1.replace({False: 0, True: 1}, inplace=False)
_input1
_input1.dtypes
cat = _input1.select_dtypes(['object'])
cat
num = _input1.select_dtypes(['int64', 'float64'])
num
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
impnum = imp.fit_transform(num)
num = pd.DataFrame(impnum, columns=num.columns)
num.isna().sum()
cat.dtypes
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
impcat = imp.fit_transform(cat)
cat = pd.DataFrame(impcat, columns=cat.columns)
cat
cat.isna().sum()
cat = cat.astype('category')
cat.dtypes
from sklearn.preprocessing import OneHotEncoder
oe = OneHotEncoder()
oe.fit_transform(cat)
cat = pd.get_dummies(data=cat, drop_first=True)
cat
cat.isna().sum()
df = pd.concat([num, cat], axis=1)
df
df.dtypes
df.isna().any()
df.shape
x = df.drop(['Transported'], axis=1)
y = df['Transported']
x
y
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=True)
x_train
x_test
y_train
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()