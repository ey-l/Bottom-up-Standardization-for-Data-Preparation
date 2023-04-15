import numpy as np
import pandas as pd
import seaborn as sn
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.info()
train_data.head(5)
total = float(train_data.shape[0])
ploting = sn.countplot(x='Transported', data=train_data)
for p in ploting.patches:
    height = p.get_height()
    ploting.text(p.get_x() + p.get_width() / 2.0, height + 5, '{:.2f}'.format(height / total * 100), ha='center')

billed_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_data['total_billed'] = train_data[billed_cols].sum(axis=1)
test_data['total_billed'] = test_data[billed_cols].sum(axis=1)
df_train = train_data.drop(columns=billed_cols, axis=1)
df_test = test_data.drop(columns=billed_cols, axis=1)
df_train.head()
cabin_train = train_data['Cabin'].astype('category')
train_data['cabin_group'] = cabin_train.apply(lambda x: x.split('/')[0])
cabin_test = test_data['Cabin'].astype('category')
test_data['cabin_group'] = cabin_test.apply(lambda x: x.split('/')[0])
train_data = train_data.drop(columns='Cabin', axis=1)
test_data = test_data.drop(columns='Cabin', axis=1)
train_data.head()
train_data['cabin_group'].hist()
home_planet_map = {'Europa': 1, 'Earth': 2, 'Mars': 3}
train_data['HomePlanet'] = train_data['HomePlanet'].map(home_planet_map)
test_data['HomePlanet'] = test_data['HomePlanet'].map(home_planet_map)
cryoSleep_map = {False: 0, True: 1}
train_data['CryoSleep'] = train_data['CryoSleep'].map(cryoSleep_map)
test_data['CryoSleep'] = test_data['CryoSleep'].map(cryoSleep_map)
destination_map = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
train_data['Destination'] = train_data['Destination'].map(destination_map)
test_data['Destination'] = test_data['Destination'].map(destination_map)
vip_map = {False: 0, True: 1}
train_data['VIP'] = train_data['VIP'].map(vip_map)
test_data['VIP'] = test_data['VIP'].map(vip_map)
cabin_group_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': '8'}
train_data['cabin_group'] = train_data['cabin_group'].map(cabin_group_map)
test_data['cabin_group'] = test_data['cabin_group'].map(cabin_group_map)
transported_map = {False: 0, True: 1}
train_data['Transported'] = train_data['Transported'].map(transported_map)
train_data.head()
train_data.columns
train_data['HomePlanet'].fillna(train_data['HomePlanet'].median(), inplace=True)
train_data['CryoSleep'].fillna(train_data['CryoSleep'].median(), inplace=True)
train_data['Destination'].fillna(train_data['Destination'].median(), inplace=True)
train_data['VIP'].fillna(train_data['VIP'].median(), inplace=True)
train_data['cabin_group'].fillna(train_data['cabin_group'].median(), inplace=True)
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['total_billed'].fillna(train_data['total_billed'].median(), inplace=True)
test_data['HomePlanet'].fillna(test_data['HomePlanet'].median(), inplace=True)
test_data['CryoSleep'].fillna(test_data['CryoSleep'].median(), inplace=True)
test_data['Destination'].fillna(test_data['Destination'].median(), inplace=True)
test_data['VIP'].fillna(test_data['VIP'].median(), inplace=True)
test_data['cabin_group'].fillna(test_data['cabin_group'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['total_billed'].fillna(test_data['total_billed'].median(), inplace=True)
train_data.drop_duplicates()
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)
train_data.isnull().sum()
test_data.isnull().sum()
train_data = train_data.drop(columns=['Name', 'PassengerId'], axis=1)
test_ids = test_data['PassengerId']
test_data = test_data.drop(columns=['Name', 'PassengerId'], axis=1)
y = train_data['Transported']
X = train_data.drop(columns='Transported')
(X_train, X_valid, y_train, y_valid) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
from sklearn.svm import SVC
model = SVC(kernel='rbf', degree=9, gamma='scale')
from sklearn.model_selection import cross_val_score, cross_val_predict
accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
np.mean(accuracy)