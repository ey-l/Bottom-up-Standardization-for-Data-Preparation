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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
pID = train['PassengerId']
print(train.isnull().sum())
print(test.isnull().sum())
cats = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Name']
train['RoomService'] = train['RoomService'].fillna(0)
train['FoodCourt'] = train['FoodCourt'].fillna(0)
train['ShoppingMall'] = train['ShoppingMall'].fillna(0)
train['Spa'] = train['Spa'].fillna(0)
train['VRDeck'] = train['VRDeck'].fillna(0)
test['RoomService'] = test['RoomService'].fillna(0)
test['FoodCourt'] = test['FoodCourt'].fillna(0)
test['ShoppingMall'] = test['ShoppingMall'].fillna(0)
test['Spa'] = test['Spa'].fillna(0)
test['VRDeck'] = test['VRDeck'].fillna(0)
for i in train.columns:
    if train[i].isna().sum() > 0:
        if i not in cats:
            train[i] = train[i].fillna(train.groupby('Transported')[i].transform('mean'))
for i in test.columns:
    if test[i].isna().sum() > 0:
        if i not in cats:
            test[i] = test[i].fillna(test[i].mean())
from matplotlib import pyplot as plt
import seaborn as sns
(fig, ax) = plt.subplots(figsize=(12, 8))
sns.countplot(x='Transported', data=train)
plt.title('Count of Transported')

print(train.columns)
cats = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
corr_df = train[num]
cor = corr_df.corr(method='pearson')
print(cor)
(fig, ax) = plt.subplots(figsize=(16, 12))
plt.title('Correlation Plot')
sns.heatmap(cor, mask=np.zeros_like(cor, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)

from scipy.stats import chi2_contingency
csq = chi2_contingency(pd.crosstab(train['Transported'], train['HomePlanet']))
print('P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(train['Transported'], train['CryoSleep']))
print('P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(train['Transported'], train['Cabin']))
print('P-value: ', csq[1])
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.countplot(x='Transported', data=train, hue='CryoSleep')
plt.title('Impact of CryoSleep on Transported')

(fig, ax) = plt.subplots(figsize=(8, 6))
sns.countplot(x='Transported', data=train, hue='HomePlanet')
plt.title('Impact of Home Planet on Transported')

train['Cabin'] = train['Cabin'].fillna(method='ffill')
test['Cabin'] = test['Cabin'].fillna(method='ffill')
train['Deck'] = train['Cabin'].apply(lambda x: x.split('/')[0])
train['Num'] = train['Cabin'].apply(lambda x: x.split('/')[1])
train['Side'] = train['Cabin'].apply(lambda x: x.split('/')[2])
test['Deck'] = test['Cabin'].apply(lambda x: x.split('/')[0])
test['Num'] = test['Cabin'].apply(lambda x: x.split('/')[1])
test['Side'] = test['Cabin'].apply(lambda x: x.split('/')[2])
csq = chi2_contingency(pd.crosstab(train['Transported'], train['Deck']))
print('Deck  P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(train['Transported'], train['Num']))
print('Num  P-value: ', csq[1])
csq = chi2_contingency(pd.crosstab(train['Transported'], train['Side']))
print('Side  P-value: ', csq[1])
train['CryoSleep'] = train['CryoSleep'].fillna(False)
test['CryoSleep'] = test['CryoSleep'].fillna(False)
train['group'] = train['PassengerId'].apply(lambda x: x.split('_')[0])
test['group'] = test['PassengerId'].apply(lambda x: x.split('_')[0])
train['Name'] = train['Name'].fillna(method='ffill')
test['Name'] = test['Name'].fillna(method='ffill')
temp = pd.DataFrame(train.groupby(['group'])['Name'])
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
train['has_relatives'] = train['group'].map(d)
temp = pd.DataFrame(test.groupby(['group'])['Name'])
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
test['has_relatives'] = test['group'].map(d)
print(train)
del train['Name'], train['group']
del test['Name'], test['group']
train['ttl_spnd'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']
test['ttl_spnd'] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']
train['Adult'] = True
train.loc[train['Age'] < 18, 'Adult'] = False
test['Adult'] = True
test.loc[test['Age'] < 18, 'Adult'] = False
del train['Cabin'], test['Cabin']
cats.remove('Cabin')
cats.append('Deck')
cats.append('Num')
cats.append('Side')
print(cats)
for i in cats:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate((train[i], test[i])).astype(str)