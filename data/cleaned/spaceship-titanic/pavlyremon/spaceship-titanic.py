import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df2 = pd.read_csv('data/input/spaceship-titanic/test.csv')
df.head()
df.info()
df.isna().any()
print(f"Home Planet Unique Values => {df['HomePlanet'].unique()}")
print(f"Destination Unique Values => {df['Destination'].unique()}")

def preprocess_data(df):
    data = df.copy()
    l1 = LabelEncoder()
    data = data.drop(columns=['Name'])
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['RoomService'] = data['RoomService'].fillna(data['RoomService'].mean())
    data['FoodCourt'] = data['FoodCourt'].fillna(data['FoodCourt'].mean())
    data['ShoppingMall'] = data['ShoppingMall'].fillna(data['ShoppingMall'].mean())
    data['Spa'] = data['Spa'].fillna(data['Spa'].mean())
    data['VRDeck'] = data['VRDeck'].fillna(data['VRDeck'].mean())
    data['CryoSleep'] = l1.fit_transform(data['CryoSleep'])
    data['Cabin'] = l1.fit_transform(data['Cabin'])
    data['VIP'] = l1.fit_transform(data['VIP'])
    data['HomePlanet'] = l1.fit_transform(data['HomePlanet'])
    data['Destination'] = l1.fit_transform(data['Destination'])
    data['Expenses'] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    data.loc[data['Expenses'] == 0, 'CryoSleep'] = 1
    data.loc[data.CryoSleep == 1, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0
    df_passenger = data.pop('PassengerId')
    return (data, df_passenger)
l1 = LabelEncoder()
(df_train, _) = preprocess_data(df)
df_train['Transported'] = l1.fit_transform(df_train['Transported'])
df_train.head()
df_train.isna().any()
y = df_train.pop('Transported')
X = df_train
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.1)
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'y_val shape: {y_val.shape}')
model = LogisticRegression(max_iter=10000)