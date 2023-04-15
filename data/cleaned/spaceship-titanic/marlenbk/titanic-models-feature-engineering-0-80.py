import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df = train.append(test, ignore_index=True)
df.shape
df.drop('Name', axis='columns', inplace=True)
df['Cabin'].str.split('/', expand=True)
df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)
df
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['side'] = labelencoder.fit_transform(df['side'])
df['HomePlanet'] = labelencoder.fit_transform(df['HomePlanet'])
df['VIP'] = labelencoder.fit_transform(df['VIP'])
df['CryoSleep'] = labelencoder.fit_transform(df['CryoSleep'])
df['Destination'] = labelencoder.fit_transform(df['Destination'])
df['ShoppingMall'].fillna(value=df['ShoppingMall'].mean(), inplace=True)
df['Spa'].fillna(value=df['Spa'].mean(), inplace=True)
df['FoodCourt'].fillna(value=df['ShoppingMall'].mean(), inplace=True)
df['VRDeck'].fillna(value=df['VRDeck'].mean(), inplace=True)
df['RoomService'].fillna(value=df['RoomService'].mean(), inplace=True)
df['Age'].fillna(value=df['Age'].mean(), inplace=True)
df_num = df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']]
df_cat = df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']]
import seaborn as sns
sns.pairplot(df_num, hue='Transported')

import plotly.express as px


def strip_plot(df, x, y):
    ax = sns.stripplot(x=df[x], y=df[y])
    plt.title(str(y), fontsize=18, fontweight='bold', fontfamily='serif', loc='left')
    ax.set(xlabel=None, ylabel=None)
fig = plt.figure(figsize=(15, 15))
plt.subplot(3, 2, 1)
strip_plot(train, 'Transported', 'RoomService')
plt.subplot(3, 2, 2)
strip_plot(train, 'Transported', 'FoodCourt')
plt.subplot(3, 2, 3)
strip_plot(train, 'Transported', 'Spa')
plt.subplot(3, 2, 4)
strip_plot(train, 'Transported', 'ShoppingMall')
plt.subplot(3, 2, 5)
strip_plot(train, 'Transported', 'VRDeck')
df['Premium'] = df['RoomService'] + df['Spa'] + df['VRDeck']
df['Basic'] = df['FoodCourt'] + df['ShoppingMall']
df
df.isnull().sum()
df['Destination'].value_counts()
df['deck'].fillna('F', inplace=True)
df['Destination'].fillna('TRAPPIST-1e', inplace=True)
df['num'].fillna('82', inplace=True)
df.drop('Cabin', axis='columns', inplace=True)
df.drop('num', axis='columns', inplace=True)
df = df.set_index('PassengerId')
df['Adult'] = (df['Age'] > 18).astype(int)
spending = ['RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa']
df['TotalSpend'] = df[spending].sum(axis=1)
df
train_df = df[df.Transported.isna() == False]
test_df = df[df.Transported.isna()]
train_df['Transported'] = labelencoder.fit_transform(train_df['Transported'])
train_df['deck'] = labelencoder.fit_transform(train_df['deck'])
test_df['deck'] = labelencoder.fit_transform(test_df['deck'])
test_df
sns.set_theme(font_scale=3)
plt.figure(figsize=(50, 50))
sns.heatmap(df.corr(), annot=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
X = train_df.drop(labels='Transported', axis=1).values
y = train_df.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
lg = LogisticRegression()