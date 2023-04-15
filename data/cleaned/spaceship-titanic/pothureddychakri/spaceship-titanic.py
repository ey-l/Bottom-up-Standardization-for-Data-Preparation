import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train
test
train.head()
train.describe()
train.info()
train.isna().any()
train.isna().sum()
train.nunique()
train.shape
train['Transported'].value_counts()
sns.countplot(x='Transported', data=train)
train.isna().sum()
train.drop(['Name', 'PassengerId', 'Cabin'], inplace=True, axis=1)
train['HomePlanet'].value_counts()
train.isna().sum()
train.dtypes
train['Destination'].value_counts()
train.replace({False: 0, True: 1}, inplace=True)
train
train.dtypes
cat = train.select_dtypes(['object'])
cat
num = train.select_dtypes(['int64', 'float64'])
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