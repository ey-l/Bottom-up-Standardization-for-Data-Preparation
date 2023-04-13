import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input0.info()
_input1['HomePlanet'].value_counts()
_input1['Destination'].value_counts()
_input0['HomePlanet'].value_counts()
_input0['Destination'].value_counts()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e')
_input1['CryoSleep'] = 1 * _input1['CryoSleep']
_input1['CryoSleep'].mean()
_input1['CryoSleep'].value_counts()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(0)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(0)
_input1['Age'].mean()
_input1.pivot_table(index=['HomePlanet', 'FoodCourt', 'ShoppingMall'], values='Age', aggfunc=['mean', 'median'])
_input1['Age'] = _input1['Age'].fillna(_input1.groupby(['HomePlanet', 'FoodCourt', 'ShoppingMall'])['Age'].transform('median'))
_input1['Age'] = _input1['Age'].fillna(29)
_input0['Age'].mean()
_input0.pivot_table(index=['HomePlanet', 'FoodCourt', 'ShoppingMall'], values='Age', aggfunc=['mean', 'median'])
_input0['Age'] = _input0['Age'].fillna(_input0.groupby(['HomePlanet', 'FoodCourt', 'ShoppingMall'])['Age'].transform('median'))
_input0['Age'] = _input0['Age'].fillna(29)
_input1['VIP'] = 1 * _input1['VIP']
_input1['VIP'].mean()
_input1['VIP'] = _input1['VIP'].fillna(0)
_input0['VIP'] = 1 * _input0['VIP']
_input0['VIP'].mean()
_input0['VIP'] = _input0['VIP'].fillna(0)
_input1['RoomService'].mean()
_input1.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='RoomService', aggfunc=['mean', 'median'])
_input1['RoomService'] = _input1['RoomService'].fillna(_input1.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['RoomService'].transform('median'))
_input1['RoomService'] = _input1['RoomService'].fillna(225)
_input0['RoomService'].mean()
_input0.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='RoomService', aggfunc=['mean', 'median'])
_input0['RoomService'] = _input0['RoomService'].fillna(_input0.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['RoomService'].transform('median'))
_input0['RoomService'] = _input0['RoomService'].fillna(219)
_input1['FoodCourt'].mean()
_input1.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='FoodCourt', aggfunc=['mean', 'median'])
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['FoodCourt'].transform('median'))
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(458)
_input0['FoodCourt'].mean()
_input0.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='FoodCourt', aggfunc=['mean', 'median'])
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['FoodCourt'].transform('median'))
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(439)
_input1['ShoppingMall'].mean()
_input1.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='ShoppingMall', aggfunc=['mean', 'median'])
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['ShoppingMall'].transform('median'))
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(173)
_input0['ShoppingMall'].mean()
_input0.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='ShoppingMall', aggfunc=['mean', 'median'])
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['ShoppingMall'].transform('median'))
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(177)
_input1['Spa'].mean()
_input1.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='Spa', aggfunc=['mean', 'median'])
_input1['Spa'] = _input1['Spa'].fillna(_input1.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['Spa'].transform('median'))
_input1['Spa'] = _input1['Spa'].fillna(311)
_input0['Spa'].mean()
_input0.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='Spa', aggfunc=['mean', 'median'])
_input0['Spa'] = _input0['Spa'].fillna(_input0.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['Spa'].transform('median'))
_input0['Spa'] = _input0['Spa'].fillna(303)
_input1['VRDeck'].mean()
_input1.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='VRDeck', aggfunc=['mean', 'median'])
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['VRDeck'].transform('median'))
_input1['VRDeck'] = _input1['VRDeck'].fillna(304)
_input0['VRDeck'].mean()
_input0.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='VRDeck', aggfunc=['mean', 'median'])
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['VRDeck'].transform('median'))
_input0['VRDeck'] = _input0['VRDeck'].fillna(310)
_input1.info()
_input0.info()
for i in _input1['HomePlanet'].unique():
    H = _input1.loc[_input1.HomePlanet == i]['Transported']
    rate = sum(H) / len(H)
    print(f'{rate} from {i}')
cryoT = _input1.loc[_input1.CryoSleep == 1]['Transported']
rate_cryoT = sum(cryoT) / len(cryoT)
rate_cryoT
for i in _input1['Destination'].unique():
    H = _input1.loc[_input1.Destination == i]['Transported']
    rate = sum(H) / len(H)
    print(f'{rate} from {i}')
kids = _input1.loc[_input1.Age <= 12]['Transported']
rate_kids = sum(kids) / len(kids)
rate_kids
teens = _input1[(_input1.Age >= 13) & (_input1.Age <= 20)]
len(teens[teens.Transported == 1]) / len(teens)
adults = _input1[(_input1.Age >= 21) & (_input1.Age <= 59)]
len(adults[adults.Transported == 1]) / len(adults)
olds = _input1[_input1.Age >= 60]['Transported']
rate_olds = sum(olds) / len(olds)
rate_olds
VIPt = _input1.loc[_input1.VIP == 1]['Transported']
rate_vip = sum(VIPt) / len(VIPt)
rate_vip
npay = _input1[(_input1.RoomService == 0) & (_input1.FoodCourt == 0) & (_input1.ShoppingMall == 0) & (_input1.Spa == 0) & (_input1.VRDeck == 0)]
len(npay[npay.Transported == 1]) / len(npay)
_input1['Cabin'].unique()
dummies1 = pd.get_dummies(_input1['HomePlanet'])
_input1 = pd.concat([_input1, dummies1], axis=1)
dummies2 = pd.get_dummies(_input0['HomePlanet'])
_input0 = pd.concat([_input0, dummies2], axis=1)
del _input1['HomePlanet']
del _input0['HomePlanet']
dummies1 = pd.get_dummies(_input1['Destination'])
_input1 = pd.concat([_input1, dummies1], axis=1)
dummies2 = pd.get_dummies(_input0['Destination'])
_input0 = pd.concat([_input0, dummies2], axis=1)
del _input1['Destination']
del _input0['Destination']
_input1['kids'] = np.where(_input1['Age'] <= 12, 1, 0)
_input0['kids'] = np.where(_input0['Age'] <= 12, 1, 0)
_input1.loc[:, ['Cabin_1']] = _input1.Cabin.str.split('/', expand=True).iloc[:, 0]
_input1.loc[:, ['Cabin_2']] = _input1.Cabin.str.split('/', expand=True).iloc[:, 1]
_input1.loc[:, ['Cabin_3']] = _input1.Cabin.str.split('/', expand=True).iloc[:, 2]
_input0.loc[:, ['Cabin_1']] = _input0.Cabin.str.split('/', expand=True).iloc[:, 0]
_input0.loc[:, ['Cabin_2']] = _input0.Cabin.str.split('/', expand=True).iloc[:, 1]
_input0.loc[:, ['Cabin_3']] = _input0.Cabin.str.split('/', expand=True).iloc[:, 2]
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
cabincol = ['Cabin_1', 'Cabin_3']
cabin2 = ['Cabin_2']
_input1[cabincol] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(_input1[cabincol]), columns=cabincol)
_input0[cabincol] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(_input0[cabincol]), columns=cabincol)
_input1[cabin2] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(_input1[cabin2]), columns=cabin2)
_input0[cabin2] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(_input0[cabin2]), columns=cabin2)
dummies1 = pd.get_dummies(_input1['Cabin_1'])
_input1 = pd.concat([_input1, dummies1], axis=1)
dummies2 = pd.get_dummies(_input0['Cabin_1'])
_input0 = pd.concat([_input0, dummies2], axis=1)
del _input1['Cabin_1']
del _input0['Cabin_1']
dummies1 = pd.get_dummies(_input1['Cabin_3'])
_input1 = pd.concat([_input1, dummies1], axis=1)
dummies2 = pd.get_dummies(_input0['Cabin_3'])
_input0 = pd.concat([_input0, dummies2], axis=1)
del _input1['Cabin_3']
del _input0['Cabin_3']
_input1['Total'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['Total'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
x_t = _input0.values
y_df = _input1['Transported']
x_df = _input1.drop('Transported', axis=1)
y = y_df.values
x = x_df.values
y.shape
x.shape
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.33, random_state=42)
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(x_t)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)