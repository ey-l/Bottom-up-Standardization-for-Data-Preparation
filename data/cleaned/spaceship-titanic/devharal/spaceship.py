import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.shape
test.shape
test.dtypes
train.info()
train.head()
train.apply(lambda x: sum(x.isnull()))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
var_mod = ['HomePlanet', 'CryoSleep', 'VIP', 'Transported']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
train.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
var_mod = ['HomePlanet', 'CryoSleep', 'VIP']
le = LabelEncoder()
for i in var_mod:
    test[i] = le.fit_transform(test[i])
train = train.fillna(0)
test = test.fillna(0)
train.apply(lambda x: sum(x.isnull()))
test.apply(lambda x: sum(x.isnull()))
data = pd.concat([train, test], ignore_index=True)
data.apply(lambda x: sum(x.isnull()))
data.shape
train.head()
train1 = train.drop(['PassengerId', 'Cabin', 'Destination', 'Name', 'Transported'], axis=1)
test1 = train.Transported
ytest = test.drop(['PassengerId', 'Cabin', 'Destination', 'Name'], axis=1)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()