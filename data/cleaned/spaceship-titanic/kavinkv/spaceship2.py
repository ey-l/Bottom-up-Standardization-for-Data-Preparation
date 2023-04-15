import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
df.info()
df.drop(df.iloc[:, 12:13], axis=1, inplace=True)
df
df.drop(df.iloc[:, 0:2], axis=1, inplace=True)
df
df.drop(df.iloc[:, 1:3], axis=1, inplace=True)
df
df = df.replace(to_replace=False, value=0)
df = df.replace(to_replace=True, value=1)
df
df.isnull().sum()
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['CryoSleep'].fillna(df['CryoSleep'].mean(), inplace=True)
df['VIP'].fillna(df['VIP'].mean(), inplace=True)
df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)
df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=True)
df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=True)
df['Spa'].fillna(df['Spa'].mean(), inplace=True)
df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)
df
x = df.iloc[:, 0:8]
y = df.Transported
x
y
x.isnull().sum()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.8, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()