import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
file_path = 'data/input/spaceship-titanic/train.csv'
org_data = pd.read_csv(file_path)
data = pd.read_csv(file_path)
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
data.head()
data.info()
data.drop(columns=['PassengerId', 'Name'], inplace=True)
test_data.drop(columns=['PassengerId', 'Name'], inplace=True)
data.isnull().sum()
data.info()
encoder = OrdinalEncoder()
data['HomePlanet'] = encoder.fit_transform(data['HomePlanet'].to_numpy().reshape(-1, 1))
test_data['HomePlanet'] = encoder.transform(test_data['HomePlanet'].to_numpy().reshape(-1, 1))
HomePlanet_cats = encoder.categories_
encoder = OrdinalEncoder()
data['CryoSleep'] = encoder.fit_transform(data['CryoSleep'].to_numpy().reshape(-1, 1))
test_data['CryoSleep'] = encoder.transform(test_data['CryoSleep'].to_numpy().reshape(-1, 1))
CryoSleep_cats = encoder.categories_
data.drop(columns=['Cabin'], inplace=True)
test_data.drop(columns=['Cabin'], inplace=True)
encoder = OrdinalEncoder()
data['Destination'] = encoder.fit_transform(data['Destination'].to_numpy().reshape(-1, 1))
test_data['Destination'] = encoder.transform(test_data['Destination'].to_numpy().reshape(-1, 1))
Destination_cats = encoder.categories_
encoder = OrdinalEncoder()
data['VIP'] = encoder.fit_transform(data['VIP'].to_numpy().reshape(-1, 1))
test_data['VIP'] = encoder.transform(test_data['VIP'].to_numpy().reshape(-1, 1))
VIP_cats = encoder.categories_
encoder = OrdinalEncoder()
data['Transported'] = encoder.fit_transform(data['Transported'].to_numpy().reshape(-1, 1))
Transported_cats = encoder.categories_
imputer = SimpleImputer(strategy='most_frequent')
data['HomePlanet'] = imputer.fit_transform(data['HomePlanet'].to_numpy().reshape(-1, 1))
test_data['HomePlanet'] = imputer.transform(test_data['HomePlanet'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='most_frequent')
data['CryoSleep'] = imputer.fit_transform(data['CryoSleep'].to_numpy().reshape(-1, 1))
test_data['CryoSleep'] = imputer.transform(test_data['CryoSleep'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='most_frequent')
data['VIP'] = imputer.fit_transform(data['VIP'].to_numpy().reshape(-1, 1))
test_data['VIP'] = imputer.transform(test_data['VIP'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='most_frequent')
data['Destination'] = imputer.fit_transform(data['Destination'].to_numpy().reshape(-1, 1))
test_data['Destination'] = imputer.transform(test_data['Destination'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='mean')
data['Age'] = imputer.fit_transform(data['Age'].to_numpy().reshape(-1, 1))
test_data['Age'] = imputer.transform(test_data['Age'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='mean')
data['RoomService'] = imputer.fit_transform(data['RoomService'].to_numpy().reshape(-1, 1))
test_data['RoomService'] = imputer.transform(test_data['RoomService'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='mean')
data['FoodCourt'] = imputer.fit_transform(data['FoodCourt'].to_numpy().reshape(-1, 1))
test_data['FoodCourt'] = imputer.transform(test_data['FoodCourt'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='mean')
data['ShoppingMall'] = imputer.fit_transform(data['ShoppingMall'].to_numpy().reshape(-1, 1))
test_data['ShoppingMall'] = imputer.transform(test_data['ShoppingMall'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='mean')
data['Spa'] = imputer.fit_transform(data['Spa'].to_numpy().reshape(-1, 1))
test_data['Spa'] = imputer.transform(test_data['Spa'].to_numpy().reshape(-1, 1))
imputer = SimpleImputer(strategy='mean')
data['VRDeck'] = imputer.fit_transform(data['VRDeck'].to_numpy().reshape(-1, 1))
test_data['VRDeck'] = imputer.transform(test_data['VRDeck'].to_numpy().reshape(-1, 1))
data.isnull().sum()
sns.catplot(data=org_data, x='HomePlanet', kind='count', hue='Transported', aspect=1, height=6)

fig = px.pie(names=org_data.CryoSleep.unique()[:-1], values=org_data.CryoSleep.value_counts(), width=500, hole=0.4)
fig.update_layout({'title': {'text': 'Cryo Sleep', 'x': 0.5}})
fig.show()
sns.catplot(data=org_data, x='CryoSleep', kind='count', aspect=1, height=5, hue='Transported')

fig = px.pie(names=org_data.Destination.unique()[:-1], values=data.Destination.value_counts(), width=500, hole=0.4)
fig.update_layout({'title': {'text': 'Destination', 'x': 0.45}})
fig.show()
sns.catplot(data=org_data, x='Destination', kind='count', aspect=1, height=5, hue='Transported')

plt.figure(figsize=(20, 8))
sns.histplot(data=org_data, x='Age', hue='Transported', multiple='dodge')

fig = px.pie(names=org_data.VIP.unique()[:-1], values=data.VIP.value_counts(), width=500, hole=0.4)
fig.update_layout({'title': {'text': 'VIP', 'x': 0.48}})
fig.show()
sns.catplot(data=org_data, x='VIP', kind='count', aspect=1, height=5, hue='Transported')

fig = px.pie(names=org_data.Transported.unique(), values=org_data.Transported.value_counts(), hole=0.4, width=500)
fig.update_layout({'title': {'text': 'Transported', 'x': 0.5}})
fig.show()
Y_full = data.pop('Transported')
X_full = data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
(X_train, X_val, y_train, y_val) = train_test_split(X_scaled, Y_full, test_size=0.2)