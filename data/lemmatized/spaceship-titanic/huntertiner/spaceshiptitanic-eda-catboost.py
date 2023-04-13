import pandas as pd
import numpy as np
import sklearn
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
pID = _input1['PassengerId']
print(_input1.isnull().sum())
print(_input0.isnull().sum())
cats = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Name']
_input1['RoomService'] = _input1['RoomService'].fillna(0)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0)
_input1['Spa'] = _input1['Spa'].fillna(0)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0)
_input0['RoomService'] = _input0['RoomService'].fillna(0)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(0)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(0)
_input0['Spa'] = _input0['Spa'].fillna(0)
_input0['VRDeck'] = _input0['VRDeck'].fillna(0)
for i in _input1.columns:
    if _input1[i].isna().sum() > 0:
        if i not in cats:
            _input1[i] = _input1[i].fillna(_input1.groupby('Transported')[i].transform('mean'))
for i in _input0.columns:
    if _input0[i].isna().sum() > 0:
        if i not in cats:
            _input0[i] = _input0[i].fillna(_input0[i].mean())
from matplotlib import pyplot as plt
import seaborn as sns
(fig, ax) = plt.subplots(figsize=(12, 8))
sns.countplot(x='Transported', data=_input1)
plt.title('Count of Transported')
print(_input1.columns)
cats = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
corr_df = _input1[num]
cor = corr_df.corr(method='pearson')
print(cor)
(fig, ax) = plt.subplots(figsize=(16, 12))
plt.title('Correlation Plot')
sns.heatmap(cor, mask=np.zeros_like(cor, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
from scipy.stats import chi2_contingency
csq = chi2_contingency(pd.crosstab(_input1['Transported'], _input1['HomePlanet']))
print('P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(_input1['Transported'], _input1['CryoSleep']))
print('P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(_input1['Transported'], _input1['Cabin']))
print('P-value: ', csq[1])
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.countplot(x='Transported', data=_input1, hue='CryoSleep')
plt.title('Impact of CryoSleep on Transported')
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.countplot(x='Transported', data=_input1, hue='HomePlanet')
plt.title('Impact of Home Planet on Transported')
_input1['Cabin'] = _input1['Cabin'].fillna(method='ffill')
_input0['Cabin'] = _input0['Cabin'].fillna(method='ffill')
_input1['Deck'] = _input1['Cabin'].apply(lambda x: x.split('/')[0])
_input1['Num'] = _input1['Cabin'].apply(lambda x: x.split('/')[1])
_input1['Side'] = _input1['Cabin'].apply(lambda x: x.split('/')[2])
_input0['Deck'] = _input0['Cabin'].apply(lambda x: x.split('/')[0])
_input0['Num'] = _input0['Cabin'].apply(lambda x: x.split('/')[1])
_input0['Side'] = _input0['Cabin'].apply(lambda x: x.split('/')[2])
csq = chi2_contingency(pd.crosstab(_input1['Transported'], _input1['Deck']))
print('Deck  P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(_input1['Transported'], _input1['Num']))
print('Num  P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(_input1['Transported'], _input1['Side']))
print('Side  P-value: ', csq[1])
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1['group'] = _input1['PassengerId'].apply(lambda x: x.split('_')[0])
_input0['group'] = _input0['PassengerId'].apply(lambda x: x.split('_')[0])
_input1['Name'] = _input1['Name'].fillna(method='ffill')
_input0['Name'] = _input0['Name'].fillna(method='ffill')
temp = pd.DataFrame(_input1.groupby(['group'])['Name'])
d = {}
for i in range(len(temp)):
    past_last_names = []
    names = list(temp[1][i])
    rltvs = 1
    for j in range(len(list(temp[1][i]))):
        if names[j].split(' ')[1] in past_last_names:
            rltvs += 1
        past_last_names.append(names[j].split(' ')[1])
    d[f'{temp[0][i]}'] = rltvs
_input1['has_relatives'] = _input1['group'].map(d)
temp = pd.DataFrame(_input0.groupby(['group'])['Name'])
d = {}
for i in range(len(temp)):
    past_last_names = []
    names = list(temp[1][i])
    rltvs = 1
    for j in range(len(list(temp[1][i]))):
        if names[j].split(' ')[1] in past_last_names:
            rltvs += 1
        past_last_names.append(names[j].split(' ')[1])
    d[f'{temp[0][i]}'] = rltvs
_input0['has_relatives'] = _input0['group'].map(d)
print(_input1)
del _input1['Name'], _input1['group']
del _input0['Name'], _input0['group']
_input1['ttl_spnd'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['ttl_spnd'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1['Adult'] = True
_input1.loc[_input1['Age'] < 18, 'Adult'] = False
_input0['Adult'] = True
_input0.loc[_input0['Age'] < 18, 'Adult'] = False
del _input1['Cabin'], _input0['Cabin']
cats.remove('Cabin')
cats.append('Deck')
cats.append('Num')
cats.append('Side')
print(cats)
for i in cats:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate((_input1[i], _input0[i])).astype(str)