import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(train.shape)
train.head()
train.nunique().sort_values(ascending=False)
round(train.isnull().sum() * 100 / len(train), 2).sort_values(ascending=False)
y = train['Transported']
train = train.drop(['PassengerId', 'Name', 'Transported'], axis=1)
for i in train:
    train[i].fillna(train[i].mode()[0], inplace=True)
obj = list(train.select_dtypes(['object']).columns)
obj
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in obj:
    train[i] = le.fit_transform(train[[i]])
train
sns.boxplot(data=train)
from sklearn.preprocessing import Normalizer