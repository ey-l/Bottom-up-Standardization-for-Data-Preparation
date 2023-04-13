import numpy as np
import pandas as pd
import seaborn as sn
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input1.head(5)
total = float(_input1.shape[0])
ploting = sn.countplot(x='Transported', data=_input1)
for p in ploting.patches:
    height = p.get_height()
    ploting.text(p.get_x() + p.get_width() / 2.0, height + 5, '{:.2f}'.format(height / total * 100), ha='center')
billed_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1['total_billed'] = _input1[billed_cols].sum(axis=1)
_input0['total_billed'] = _input0[billed_cols].sum(axis=1)
df_train = _input1.drop(columns=billed_cols, axis=1)
df_test = _input0.drop(columns=billed_cols, axis=1)
df_train.head()
cabin_train = _input1['Cabin'].astype('category')
_input1['cabin_group'] = cabin_train.apply(lambda x: x.split('/')[0])
cabin_test = _input0['Cabin'].astype('category')
_input0['cabin_group'] = cabin_test.apply(lambda x: x.split('/')[0])
_input1 = _input1.drop(columns='Cabin', axis=1)
_input0 = _input0.drop(columns='Cabin', axis=1)
_input1.head()
_input1['cabin_group'].hist()
home_planet_map = {'Europa': 1, 'Earth': 2, 'Mars': 3}
_input1['HomePlanet'] = _input1['HomePlanet'].map(home_planet_map)
_input0['HomePlanet'] = _input0['HomePlanet'].map(home_planet_map)
cryoSleep_map = {False: 0, True: 1}
_input1['CryoSleep'] = _input1['CryoSleep'].map(cryoSleep_map)
_input0['CryoSleep'] = _input0['CryoSleep'].map(cryoSleep_map)
destination_map = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
_input1['Destination'] = _input1['Destination'].map(destination_map)
_input0['Destination'] = _input0['Destination'].map(destination_map)
vip_map = {False: 0, True: 1}
_input1['VIP'] = _input1['VIP'].map(vip_map)
_input0['VIP'] = _input0['VIP'].map(vip_map)
cabin_group_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': '8'}
_input1['cabin_group'] = _input1['cabin_group'].map(cabin_group_map)
_input0['cabin_group'] = _input0['cabin_group'].map(cabin_group_map)
transported_map = {False: 0, True: 1}
_input1['Transported'] = _input1['Transported'].map(transported_map)
_input1.head()
_input1.columns
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].median(), inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].median(), inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].median(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].median(), inplace=False)
_input1['cabin_group'] = _input1['cabin_group'].fillna(_input1['cabin_group'].median(), inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median(), inplace=False)
_input1['total_billed'] = _input1['total_billed'].fillna(_input1['total_billed'].median(), inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].median(), inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].median(), inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].median(), inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].median(), inplace=False)
_input0['cabin_group'] = _input0['cabin_group'].fillna(_input0['cabin_group'].median(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median(), inplace=False)
_input0['total_billed'] = _input0['total_billed'].fillna(_input0['total_billed'].median(), inplace=False)
_input1.drop_duplicates()
_input1 = _input1.fillna(_input1.mean(), inplace=False)
_input0 = _input0.fillna(_input0.mean(), inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
_input1 = _input1.drop(columns=['Name', 'PassengerId'], axis=1)
test_ids = _input0['PassengerId']
_input0 = _input0.drop(columns=['Name', 'PassengerId'], axis=1)
y = _input1['Transported']
X = _input1.drop(columns='Transported')
(X_train, X_valid, y_train, y_valid) = sklearn.model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
from sklearn.svm import SVC
model = SVC(kernel='rbf', degree=9, gamma='scale')
from sklearn.model_selection import cross_val_score, cross_val_predict
accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
np.mean(accuracy)