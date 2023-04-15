import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from scipy import stats
import os

seed = 1337

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed)
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
pID = train['PassengerId']
cats = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Name']
train.head()

def show_nan(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True)
    print(missing_value_df)
show_nan(train)
print()
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
'        if i in cats:\n            train[i]=train[i].fillna(method="ffill")'
train['Cabin'] = train['Cabin'].fillna(method='ffill')
test['Cabin'] = test['Cabin'].fillna(method='ffill')
train['deck'] = train['Cabin'].apply(lambda x: x.split('/')[0])
train['num'] = train['Cabin'].apply(lambda x: x.split('/')[1])
train['side'] = train['Cabin'].apply(lambda x: x.split('/')[2])
test['deck'] = test['Cabin'].apply(lambda x: x.split('/')[0])
test['num'] = test['Cabin'].apply(lambda x: x.split('/')[1])
test['side'] = test['Cabin'].apply(lambda x: x.split('/')[2])
"\n#show_nan(train)\nt = pd.concat([train['PassengerId'],train['Cabin']], axis=1, keys=['PassengerId','Cabin'])\nt_t = pd.concat([test['PassengerId'],test['Cabin']], axis=1, keys=['PassengerId','Cabin'])\nt=t.dropna()\nt_t=t_t.dropna()\nt['deck'] = t['Cabin'].apply(lambda x: x.split('/')[0])\nt['num'] = t['Cabin'].apply(lambda x: x.split('/')[1])\nt['side'] = t['Cabin'].apply(lambda x: x.split('/')[2])\n\nt_t['deck'] = t_t['Cabin'].apply(lambda x: x.split('/')[0])\nt_t['num'] = t_t['Cabin'].apply(lambda x: x.split('/')[1])\nt_t['side'] = t_t['Cabin'].apply(lambda x: x.split('/')[2])\n\ndel t['Cabin'],t_t['Cabin']\n\ntrain=train.merge(t,on='PassengerId',how='left')\ntest=test.merge(t_t,on='PassengerId',how='left')\n\ntrain['CryoSleep']=train['CryoSleep'].fillna(False)\ntest['CryoSleep']=test['CryoSleep'].fillna(False)\ntrain['VIP']=train['VIP'].fillna(True)\ntest['VIP']=test['VIP'].fillna(True)\ntrain=train.fillna('missing_val')\ntest=test.fillna('missing_val')\n"
del train['Cabin'], test['Cabin']
cats.remove('Cabin')
cats.append('deck')
cats.append('num')
cats.append('side')
train['CryoSleep'] = train['CryoSleep'].fillna(False)
test['CryoSleep'] = test['CryoSleep'].fillna(False)
for i in test.columns:
    if test[i].isna().sum() > 0:
        if i in cats:
            test[i] = test[i].fillna(test[i].value_counts(ascending=True).index[-1])
"train['last_name']=train['Name'].apply(lambda x: str(x).split(' ')[1])\nd=train['last_name'].value_counts().to_dict()\ntrain['has_relatives']=train['last_name'].map(d)\nprint(max(train['has_relatives']))\ndel train['last_name']\n\ntest['last_name']=test['Name'].apply(lambda x: str(x).split(' ')[1])\nd=test['last_name'].value_counts().to_dict()\ntest['has_relatives']=test['last_name'].map(d)\nprint(max(test['has_relatives']))\ndel test['last_name']"
cats.remove('Name')
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
"\ntrain['RS_part']=train['RoomService']/train['ttl_spnd']\ntrain['FC_part']=train['FoodCourt']/train['ttl_spnd']\ntrain['SM_part']=train['ShoppingMall']/train['ttl_spnd']\ntrain['S_part']=train['Spa']/train['ttl_spnd']\ntrain['VR_part']=train['VRDeck']/train['ttl_spnd']\n\ntest['RS_part']=test['RoomService']/test['ttl_spnd']\ntest['FC_part']=test['FoodCourt']/test['ttl_spnd']\ntest['SM_part']=test['ShoppingMall']/test['ttl_spnd']\ntest['S_part']=test['Spa']/test['ttl_spnd']\ntest['VR_part']=test['VRDeck']/test['ttl_spnd']"
train['Adult'] = True
train.loc[train['Age'] < 18, 'Adult'] = False
test['Adult'] = True
test.loc[test['Age'] < 18, 'Adult'] = False
print(cats)
from sklearn.preprocessing import LabelEncoder
for i in cats:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate((train[i], test[i])).astype(str)