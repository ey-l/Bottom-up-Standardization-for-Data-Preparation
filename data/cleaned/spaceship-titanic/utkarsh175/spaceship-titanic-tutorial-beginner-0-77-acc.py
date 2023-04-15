import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_test.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['HomePlanet'] = le.fit_transform(df_train['HomePlanet'])
df_test['HomePlanet'] = le.fit_transform(df_test['HomePlanet'])
df_train['CryoSleep'] = le.fit_transform(df_train['CryoSleep'])
df_test['CryoSleep'] = le.fit_transform(df_test['CryoSleep'])
df_train['Cabin'] = le.fit_transform(df_train['Cabin'])
df_test['Cabin'] = le.fit_transform(df_test['Cabin'])
df_train['Destination'] = le.fit_transform(df_train['Destination'])
df_test['Destination'] = le.fit_transform(df_test['Destination'])
df_train.info()
df_train['RoomService'].fillna(df_train['RoomService'].median(), inplace=True)
df_train.info()
df_test.info()
df_test['RoomService'].fillna(df_test['RoomService'].median(), inplace=True)
df_train['FoodCourt'].fillna(df_train['FoodCourt'].median(), inplace=True)
df_test['FoodCourt'].fillna(df_test['FoodCourt'].median(), inplace=True)
df_train['ShoppingMall'].fillna(df_train['ShoppingMall'].median(), inplace=True)
df_test['ShoppingMall'].fillna(df_test['ShoppingMall'].median(), inplace=True)
df_train['Spa'].fillna(df_train['Spa'].median(), inplace=True)
df_test['Spa'].fillna(df_test['Spa'].median(), inplace=True)
df_train['VRDeck'].fillna(df_train['VRDeck'].median(), inplace=True)
df_test['VRDeck'].fillna(df_test['VRDeck'].median(), inplace=True)
df_train.drop('Name', axis=1, inplace=True)
df_test.drop('Name', axis=1, inplace=True)
df_train['VIP'] = le.fit_transform(df_train['VIP'])
df_test['VIP'] = le.fit_transform(df_test['VIP'])
df_train['Transported'] = le.fit_transform(df_train['Transported'])
df_train.drop('PassengerId', axis=1, inplace=True)
df_test.drop('PassengerId', axis=1, inplace=True)
df_train.info()
df_test.info()
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
from sklearn.ensemble import RandomForestClassifier
X_train = df_train.drop('Transported', axis=1)
y_train = df_train[['Transported']]
X_test = df_test
y_train['Transported']
X_train.info()
rf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=7)