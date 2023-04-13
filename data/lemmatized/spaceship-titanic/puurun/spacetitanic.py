import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
for col in _input1:
    print(f'{col:15}: {_input1[col].unique()}')
_input1.isnull().mean()
_input0.isnull().mean()
combined = (_input1, _input0)
for dataset in combined:
    dataset[['Cabin_1', 'Cabin_2', 'Cabin_3']] = dataset['Cabin'].str.split('/', expand=True)
    dataset[['FirstName', 'LastName']] = dataset['Name'].str.split(' ', expand=True)
    dataset[['GroupId', 'PersonId']] = dataset['PassengerId'].str.split('_', expand=True)
    dataset['TotalSpent'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
_input1.head()
_input0.head()
maybe_not_necessary_var = ['PassengerId', 'Name', 'Cabin', 'PersonId', 'FirstName']
cat_var = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_1', 'Cabin_2', 'Cabin_3', 'GroupId', 'LastName']
cont_var = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent']
cols = cat_var + cont_var
y = _input1['Transported']
X = _input1.drop('Transported', axis=1)
Xt = _input0.copy()
X['CryoSleep'] = X['CryoSleep'].apply(lambda x: str(x) if pd.notnull(x) else x)
Xt['CryoSleep'] = Xt['CryoSleep'].apply(lambda x: str(x) if pd.notnull(x) else x)
X['VIP'] = X['VIP'].apply(lambda x: str(x) if pd.notnull(x) else x)
Xt['VIP'] = Xt['VIP'].apply(lambda x: str(x) if pd.notnull(x) else x)
X.isnull().sum()

def Categorize(x, xt, cat_var, cols):
    x = x.copy()[cols]
    xt = xt.copy()[cols]
    x[cat_var] = x[cat_var]
    xt[cat_var] = xt[cat_var]
    for c in cat_var:
        x.loc[x.isnull()[c], c] = 0
        xt.loc[xt.isnull()[c], c] = 0
    dicts = {col: {key: val for (val, key) in enumerate(x[col].unique(), 1)} for col in cat_var}
    for dictcol in dicts:
        unknown_key = list(set(xt[dictcol].unique()) - set(x[dictcol].unique()))
        for key in unknown_key:
            xt.loc[xt[dictcol] == key, dictcol] = 0
        for (key, val) in dicts[dictcol].items():
            x.loc[x[dictcol] == key, dictcol] = val
            xt.loc[xt[dictcol] == key, dictcol] = val
    return (x, xt)
(Xc, Xtc) = Categorize(X, Xt, cat_var, cols)
Xc

def FillContVar(x, xt, cont_var, group_var, cols):
    x = x.copy()[cols]
    xt = xt.copy()[cols]
    for dataset in (x, xt):
        for cv in cont_var:
            dataset[cv] = dataset.groupby(group_var)[cv].apply(lambda a: a.fillna(a.median()))
    return (x, xt)
(Xcc, Xtcc) = FillContVar(Xc, Xtc, cont_var, ['HomePlanet', 'CryoSleep'], cols)
Xcc.head()
Xtcc.head()
Xcc.isnull().sum()
Xtcc.isnull().sum()
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(1222, min_samples_leaf=4, oob_score=True, n_jobs=-1, random_state=2)