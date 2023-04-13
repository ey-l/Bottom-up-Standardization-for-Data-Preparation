import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
train_path = 'data/input/spaceship-titanic/train.csv'
test_path = 'data/input/spaceship-titanic/test.csv'
train = read_csv(train_path)
test = read_csv(test_path)
print(train.head(4))
train = train.drop(['PassengerId', 'Name'], axis=1, inplace=False)
train.isna().sum()
train.info()
df = pd.DataFrame(train)
df = df.fillna(df.mean())
df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
df = df.drop(['Cabin'], axis=1, inplace=False)
df.isna().sum()
df['CryoSleep'] = df['CryoSleep'].astype(int)
df['VIP'] = df['VIP'].astype(int)
df['Side'] = df['Side'].map({'P': 1, 'S': 2}).astype('int8', errors='ignore')
df['Deck'] = df['Deck'].map({'B': 1, 'F': 2, 'A': 3, 'G': 4, 'E': 5, 'D': 6, 'C': 7, 'T': 8}).astype('int8', errors='ignore')
df['Destination'] = df['Destination'].map({'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}).astype('int8', errors='ignore')
df['HomePlanet'] = df['HomePlanet'].map({'Europa': 1, 'Earth': 2, 'Mars': 3}).astype('int8', errors='ignore')
df['Transported'] = df['Transported'].astype('int8')
df['Num'] = df['Num'].astype('int8', errors='ignore')
df = df.dropna()
df.isna().sum()
df.head(20)
print(df)
x = df[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Num', 'Side']]
y = df['Transported']
df.isna().sum()
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()