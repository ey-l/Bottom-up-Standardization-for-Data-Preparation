import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.Transported = le.fit_transform(df['Transported'])
df
df.drop(['PassengerId'], axis=1, inplace=True)
df
df.drop(['Name'], axis=1, inplace=True)
df
df.VIP = le.fit_transform(df['VIP'])
df
df.Destination = le.fit_transform(df['Destination'])
df
df.CryoSleep = le.fit_transform(df['CryoSleep'])
df
df.drop(['Cabin'], axis=1, inplace=True)
df
df.HomePlanet = le.fit_transform(df['HomePlanet'])
df
df.isnull().sum()
df['Age'].fillna(df['Age'].median(skipna=True), inplace=True)
df.isnull().sum()
df = df.dropna()
df
y = df.Transported
x = df.drop(['Transported'], axis=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
s_x = sc.fit_transform(x)
s_x
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(s_x, y, random_state=0, test_size=0.25)
x_train
x_train.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()