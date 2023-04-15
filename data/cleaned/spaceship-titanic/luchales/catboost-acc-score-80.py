import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head()
test.head()
train.info()
test.info()
train['HomePlanet'].value_counts()
train['Destination'].value_counts()
test['HomePlanet'].value_counts()
test['Destination'].value_counts()
train['HomePlanet'] = train['HomePlanet'].fillna('Earth')
train['Destination'] = train['Destination'].fillna('TRAPPIST-1e')
test['HomePlanet'] = test['HomePlanet'].fillna('Earth')
test['Destination'] = test['Destination'].fillna('TRAPPIST-1e')
train['CryoSleep'] = 1 * train['CryoSleep']
train['CryoSleep'].mean()
train['CryoSleep'].value_counts()
train['CryoSleep'] = train['CryoSleep'].fillna(0)
test['CryoSleep'] = test['CryoSleep'].fillna(0)
train['Age'].mean()
train.pivot_table(index=['HomePlanet', 'FoodCourt', 'ShoppingMall'], values='Age', aggfunc=['mean', 'median'])
train['Age'] = train['Age'].fillna(train.groupby(['HomePlanet', 'FoodCourt', 'ShoppingMall'])['Age'].transform('median'))
train['Age'] = train['Age'].fillna(29)
test['Age'].mean()
test.pivot_table(index=['HomePlanet', 'FoodCourt', 'ShoppingMall'], values='Age', aggfunc=['mean', 'median'])
test['Age'] = test['Age'].fillna(test.groupby(['HomePlanet', 'FoodCourt', 'ShoppingMall'])['Age'].transform('median'))
test['Age'] = test['Age'].fillna(29)
train['VIP'] = 1 * train['VIP']
train['VIP'].mean()
train['VIP'] = train['VIP'].fillna(0)
test['VIP'] = 1 * test['VIP']
test['VIP'].mean()
test['VIP'] = test['VIP'].fillna(0)
train['RoomService'].mean()
train.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='RoomService', aggfunc=['mean', 'median'])
train['RoomService'] = train['RoomService'].fillna(train.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['RoomService'].transform('median'))
train['RoomService'] = train['RoomService'].fillna(225)
test['RoomService'].mean()
test.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='RoomService', aggfunc=['mean', 'median'])
test['RoomService'] = test['RoomService'].fillna(test.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['RoomService'].transform('median'))
test['RoomService'] = test['RoomService'].fillna(219)
train['FoodCourt'].mean()
train.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='FoodCourt', aggfunc=['mean', 'median'])
train['FoodCourt'] = train['FoodCourt'].fillna(train.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['FoodCourt'].transform('median'))
train['FoodCourt'] = train['FoodCourt'].fillna(458)
test['FoodCourt'].mean()
test.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='FoodCourt', aggfunc=['mean', 'median'])
test['FoodCourt'] = test['FoodCourt'].fillna(test.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['FoodCourt'].transform('median'))
test['FoodCourt'] = test['FoodCourt'].fillna(439)
train['ShoppingMall'].mean()
train.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='ShoppingMall', aggfunc=['mean', 'median'])
train['ShoppingMall'] = train['ShoppingMall'].fillna(train.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['ShoppingMall'].transform('median'))
train['ShoppingMall'] = train['ShoppingMall'].fillna(173)
test['ShoppingMall'].mean()
test.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='ShoppingMall', aggfunc=['mean', 'median'])
test['ShoppingMall'] = test['ShoppingMall'].fillna(test.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['ShoppingMall'].transform('median'))
test['ShoppingMall'] = test['ShoppingMall'].fillna(177)
train['Spa'].mean()
train.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='Spa', aggfunc=['mean', 'median'])
train['Spa'] = train['Spa'].fillna(train.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['Spa'].transform('median'))
train['Spa'] = train['Spa'].fillna(311)
test['Spa'].mean()
test.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='Spa', aggfunc=['mean', 'median'])
test['Spa'] = test['Spa'].fillna(test.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['Spa'].transform('median'))
test['Spa'] = test['Spa'].fillna(303)
train['VRDeck'].mean()
train.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='VRDeck', aggfunc=['mean', 'median'])
train['VRDeck'] = train['VRDeck'].fillna(train.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['VRDeck'].transform('median'))
train['VRDeck'] = train['VRDeck'].fillna(304)
test['VRDeck'].mean()
test.pivot_table(index=['HomePlanet', 'Age', 'CryoSleep', 'Destination'], values='VRDeck', aggfunc=['mean', 'median'])
test['VRDeck'] = test['VRDeck'].fillna(test.groupby(['HomePlanet', 'Age', 'CryoSleep', 'Destination'])['VRDeck'].transform('median'))
test['VRDeck'] = test['VRDeck'].fillna(310)
train.info()
test.info()
for i in train['HomePlanet'].unique():
    H = train.loc[train.HomePlanet == i]['Transported']
    rate = sum(H) / len(H)
    print(f'{rate} from {i}')
cryoT = train.loc[train.CryoSleep == 1]['Transported']
rate_cryoT = sum(cryoT) / len(cryoT)
rate_cryoT
for i in train['Destination'].unique():
    H = train.loc[train.Destination == i]['Transported']
    rate = sum(H) / len(H)
    print(f'{rate} from {i}')
kids = train.loc[train.Age <= 12]['Transported']
rate_kids = sum(kids) / len(kids)
rate_kids
teens = train[(train.Age >= 13) & (train.Age <= 20)]
len(teens[teens.Transported == 1]) / len(teens)
adults = train[(train.Age >= 21) & (train.Age <= 59)]
len(adults[adults.Transported == 1]) / len(adults)
olds = train[train.Age >= 60]['Transported']
rate_olds = sum(olds) / len(olds)
rate_olds
VIPt = train.loc[train.VIP == 1]['Transported']
rate_vip = sum(VIPt) / len(VIPt)
rate_vip
npay = train[(train.RoomService == 0) & (train.FoodCourt == 0) & (train.ShoppingMall == 0) & (train.Spa == 0) & (train.VRDeck == 0)]
len(npay[npay.Transported == 1]) / len(npay)
train['Cabin'].unique()
dummies1 = pd.get_dummies(train['HomePlanet'])
train = pd.concat([train, dummies1], axis=1)
dummies2 = pd.get_dummies(test['HomePlanet'])
test = pd.concat([test, dummies2], axis=1)
del train['HomePlanet']
del test['HomePlanet']
dummies1 = pd.get_dummies(train['Destination'])
train = pd.concat([train, dummies1], axis=1)
dummies2 = pd.get_dummies(test['Destination'])
test = pd.concat([test, dummies2], axis=1)
del train['Destination']
del test['Destination']
train['kids'] = np.where(train['Age'] <= 12, 1, 0)
test['kids'] = np.where(test['Age'] <= 12, 1, 0)
train.loc[:, ['Cabin_1']] = train.Cabin.str.split('/', expand=True).iloc[:, 0]
train.loc[:, ['Cabin_2']] = train.Cabin.str.split('/', expand=True).iloc[:, 1]
train.loc[:, ['Cabin_3']] = train.Cabin.str.split('/', expand=True).iloc[:, 2]
test.loc[:, ['Cabin_1']] = test.Cabin.str.split('/', expand=True).iloc[:, 0]
test.loc[:, ['Cabin_2']] = test.Cabin.str.split('/', expand=True).iloc[:, 1]
test.loc[:, ['Cabin_3']] = test.Cabin.str.split('/', expand=True).iloc[:, 2]
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
cabincol = ['Cabin_1', 'Cabin_3']
cabin2 = ['Cabin_2']
train[cabincol] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(train[cabincol]), columns=cabincol)
test[cabincol] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(test[cabincol]), columns=cabincol)
train[cabin2] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(train[cabin2]), columns=cabin2)
test[cabin2] = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(test[cabin2]), columns=cabin2)
dummies1 = pd.get_dummies(train['Cabin_1'])
train = pd.concat([train, dummies1], axis=1)
dummies2 = pd.get_dummies(test['Cabin_1'])
test = pd.concat([test, dummies2], axis=1)
del train['Cabin_1']
del test['Cabin_1']
dummies1 = pd.get_dummies(train['Cabin_3'])
train = pd.concat([train, dummies1], axis=1)
dummies2 = pd.get_dummies(test['Cabin_3'])
test = pd.concat([test, dummies2], axis=1)
del train['Cabin_3']
del test['Cabin_3']
train['Total'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']
test['Total'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']
train = train.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
test = test.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
x_t = test.values
y_df = train['Transported']
x_df = train.drop('Transported', axis=1)
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