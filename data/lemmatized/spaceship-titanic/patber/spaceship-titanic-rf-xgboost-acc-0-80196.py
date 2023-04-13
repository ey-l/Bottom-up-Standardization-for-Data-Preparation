import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataprep.eda import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
sns.set()
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
combine = [_input1, _input0]
plot(_input1)
pd.crosstab(_input1['HomePlanet'], _input1['CryoSleep'], normalize='index')
pd.crosstab(_input1['Destination'], _input1['CryoSleep'], normalize='index')
plot_correlation(_input1)
for dataset in combine:
    dataset['Spend'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
    dataset['HasSpend'] = 0
    dataset.loc[dataset['Spend'] > 0, 'HasSpend'] = 1
_input1.head()
plot(_input1, 'Spend')
plot(_input1, 'Spend', 'Transported')
for dataset in combine:
    dataset['Spend'] = np.log(dataset['Spend'])
_input1.head()
plt.figure(figsize=(10, 5))
sns.histplot(data=_input1, x='Spend', hue='Transported', bins=30)
for dataset in combine:
    dataset['Spend'] = dataset['Spend'].fillna(0)
    dataset['Spend'] = dataset['Spend'].replace([np.inf, -np.inf], 0)
dataset['Spend'].isna().sum()
CreditFeatures = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for var in CreditFeatures:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=_input1, x=var, hue='Transported', bins=30)
for dataset in combine:
    for col in dataset[CreditFeatures]:
        dataset[col] = np.log(pd.to_numeric(dataset[col], errors='coerce'))
        dataset[col] = dataset[col].replace([-np.inf], 0, inplace=False)
_input1.head()
for dataset in combine:
    dataset['PaxGroup'] = dataset['PassengerId'].str[:4]
_input1.head()
for dataset in combine:
    dataset['PaxGroupSize'] = dataset['PaxGroup'].map(dataset['PaxGroup'].value_counts())
_input1.head()
_input1[['PaxGroupSize', 'Transported']].groupby(['PaxGroupSize'], as_index=False).mean().sort_values(by='Transported', ascending=False)
for dataset in combine:
    dataset.loc[dataset['PaxGroupSize'] == 1, 'PaxGroupSize'] = 0
    dataset.loc[(dataset['PaxGroupSize'] == 2) | (dataset['PaxGroupSize'] == 7), 'PaxGroupSize'] = 1
    dataset.loc[(dataset['PaxGroupSize'] == 3) | (dataset['PaxGroupSize'] == 5) | (dataset['PaxGroupSize'] == 6), 'PaxGroupSize'] = 2
    dataset.loc[dataset['PaxGroupSize'] == 4, 'PaxGroupSize'] = 3
    dataset.loc[dataset['PaxGroupSize'] == 8, 'PaxGroupSize'] = 4
_input1.head()
plot(_input1, 'PaxGroupSize')
plot(_input1, 'PaxGroupSize', 'Transported')
for dataset in combine:
    dataset['GivenName'] = dataset['Name'].str.split(' ', expand=True)[0]
    dataset['FamilyName'] = dataset['Name'].str.split(' ', expand=True)[1]
_input1.head()
for dataset in combine:
    dataset['Deck'] = dataset['Cabin'].str[:1]
_input1.head()
plot(_input1, 'Deck')
plot(_input1, 'Deck', 'Transported')
plot_missing(_input1, 'Deck', 'Transported')
_input1[['Deck', 'Transported']].groupby('Deck').mean().sort_values('Transported', ascending=False)
_input1['Deck'].isna().sum()
for dataset in combine:
    dataset['Deck'] = dataset['Deck'].transform(lambda x: x.fillna(x.mode()[0]))
_input1['Deck'].isna().sum()
for dataset in combine:
    dataset['DeckSide'] = dataset['Cabin'].str[-1:]
_input1.head()
plot(_input1, 'DeckSide')
plot(_input1, 'DeckSide', 'Transported')
plot_missing(_input1, 'DeckSide', 'Transported')
_input1[['DeckSide', 'Transported']].groupby('DeckSide').mean().sort_values('Transported', ascending=False)
_input1['DeckSide'].isna().sum()
for dataset in combine:
    dataset['DeckSide'] = dataset['DeckSide'].transform(lambda x: x.fillna(x.mode()[0]))
_input1['DeckSide'].isna().sum()
for dataset in combine:
    dataset['CabinNumber'] = dataset['Cabin'].str.split('/', expand=True)[1].astype('float').astype('Int16')
_input1.head()
for dataset in combine:
    dataset['CabinNumber'] = dataset['CabinNumber'].transform(lambda x: x.fillna(x.mode()[0]))
plot(_input1, 'CabinNumber')
plot(_input1, 'CabinNumber', 'Transported')
for dataset in combine:
    dataset.loc[dataset['CabinNumber'] <= 300, 'CabinNumber'] = 0
    dataset.loc[(dataset['CabinNumber'] > 300) & (dataset['CabinNumber'] <= 600), 'CabinNumber'] = 1
    dataset.loc[(dataset['CabinNumber'] > 600) & (dataset['CabinNumber'] <= 900), 'CabinNumber'] = 2
    dataset.loc[(dataset['CabinNumber'] > 900) & (dataset['CabinNumber'] <= 1200), 'CabinNumber'] = 3
    dataset.loc[(dataset['CabinNumber'] > 1200) & (dataset['CabinNumber'] <= 1500), 'CabinNumber'] = 4
    dataset.loc[(dataset['CabinNumber'] > 1500) & (dataset['CabinNumber'] <= 1800), 'CabinNumber'] = 5
    dataset.loc[dataset['CabinNumber'] > 1800, 'CabinNumber'] = 6
_input1.head()
plot(_input1, 'CabinNumber', 'Transported')
_input1[['CabinNumber', 'Transported']].groupby('CabinNumber').mean().sort_values('Transported', ascending=False)
plot(_input1, 'Age')
plot(_input1, 'Age', 'Transported')
plot_missing(_input1, 'Age', 'Transported')
for dataset in combine:
    dataset['Age'] = dataset['Age'].replace(0, np.nan, inplace=False)
_input1['Age'].isna().sum()
for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(dataset.groupby(['HomePlanet', 'Destination', 'VIP', 'HasSpend', 'PaxGroupSize', 'Deck', 'DeckSide'])['Age'].transform('mean'))
_input1['Age'].isna().sum()
_input1.loc[_input1['Age'].isna()]
for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(dataset.groupby(['HasSpend', 'PaxGroupSize', 'CryoSleep'])['Age'].transform('mean'))
_input1['Age'].isna().sum()
_input1['AgeBand'] = pd.cut(_input1['Age'], 5)
_input1[['AgeBand', 'Transported']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:
    dataset.loc[dataset['Age'] <= 17, 'Age'] = 0
    dataset.loc[dataset['Age'] > 17, 'Age'] = 1
    dataset['Age'] = dataset['Age'].astype('int')
_input1.head()
plot(_input1, 'Age', 'Transported')
plot(_input1, 'CryoSleep')
plot(_input1, 'CryoSleep', 'Transported')
plot_missing(_input1, 'CryoSleep', 'Transported')
for dataset in combine:
    dataset.loc[dataset['CryoSleep'].isna() & (dataset['HasSpend'] == 0), 'CryoSleep'] = 1
_input1['CryoSleep'].isna().sum()
for dataset in combine:
    dataset.loc[dataset['CryoSleep'].isna() & (dataset['HasSpend'] != 0), 'CryoSleep'] = 0
_input1['CryoSleep'].isna().sum()
_input1['CryoSleep']
for dataset in combine:
    dataset['CryoSleep'] = dataset['CryoSleep'].replace({'True': True, 'False': False}).astype('int')
_input1['CryoSleep']
plot(_input1, 'HomePlanet')
plot(_input1, 'HomePlanet', 'Transported')
plot_missing(_input1, 'HomePlanet', 'Transported')
_input1[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='HomePlanet', ascending=False)
pd.pivot_table(_input1, values='PassengerId', index='Deck', columns='HomePlanet', aggfunc='count')
for dataset in combine:
    dataset.loc[(dataset['Deck'] == 'G') & dataset['HomePlanet'].isna(), 'HomePlanet'] = 'Earth'
    dataset.loc[((dataset['Deck'] == 'A') | (dataset['Deck'] == 'B') | (dataset['Deck'] == 'C') | (dataset['Deck'] == 'T')) & dataset['HomePlanet'].isna(), 'HomePlanet'] = 'Europa'
_input1['HomePlanet'].isna().sum()
tmp = _input1.loc[~pd.isna(_input1['HomePlanet']), ['HomePlanet', 'PaxGroup']]
result = tmp.groupby(['PaxGroup', 'HomePlanet'])['HomePlanet'].count().unstack().isna().sum(axis=1)
result.describe()
labels = {'Earth': 1, 'Europa': 2, 'Mars': 3}
for dataset in combine:
    dataset['HomePlanet'] = dataset['HomePlanet'].map(labels)
    tmp = dataset[['HomePlanet', 'PaxGroup']]
    tmp['HomePlanet'] = tmp.groupby('PaxGroup').transform(lambda x: x.fillna(x.mean()))
    dataset['HomePlanet'] = tmp['HomePlanet']
_input1['HomePlanet'].isna().sum()
for dataset in combine:
    dataset['HomePlanet'] = dataset['HomePlanet'].transform(lambda x: x.fillna(x.mode()[0]))
_input1['HomePlanet'].isna().sum()
plot(_input1, 'Destination')
plot(_input1, 'Destination', 'Transported')
plot_missing(_input1, 'Destination', 'Transported')
_input1['Destination'].isna().sum()
pd.pivot_table(_input1, values='PassengerId', index='Deck', columns='Destination', aggfunc='count')
pd.pivot_table(_input1, values='PassengerId', index='HomePlanet', columns='Destination', aggfunc='count')
for dataset in combine:
    dataset['Destination'] = dataset['Destination'].transform(lambda x: x.fillna(x.mode()[0]))
_input1['Destination'].isna().sum()
plot(_input1, 'VIP')
plot(_input1, 'VIP', 'Transported')
plot_missing(_input1, 'VIP', 'Transported')
for dataset in combine:
    dataset['VIP'] = dataset['VIP'].transform(lambda x: x.fillna(x.mode()[0]))
_input1['VIP'].isna().sum()
for dataset in combine:
    dataset['VIP'] = dataset['VIP'].replace({'True': 1, 'False': 0}).astype('int')
for dataset in combine:
    dataset['FamilySize'] = dataset.groupby('FamilyName')['FamilyName'].transform('count')
_input1.head()
plot(_input1, 'FamilySize')
plot(_input1, 'FamilySize', 'Transported')
_input1[['FamilySize', 'Transported']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='FamilySize', ascending=False)
for dataset in combine:
    dataset['FamilySize'] = dataset['FamilySize'].fillna(dataset['FamilySize'].median()).astype('int')
plot(_input1, 'FamilySize', 'Transported')
_input1['FamilySize'].isna().sum()
_input1.head()
_input1 = _input1.drop(['Cabin', 'Name', col, 'PaxGroup', 'AgeBand', 'GivenName', 'FamilyName', 'VIP', 'HasSpend', 'CabinNumber'], axis=1)
_input0 = _input0.drop(['Cabin', 'Name', col, 'PaxGroup', 'GivenName', 'FamilyName', 'VIP', 'HasSpend', 'CabinNumber'], axis=1)
_input1.head()
_input1 = pd.get_dummies(_input1, columns=['CryoSleep', 'HomePlanet', 'Destination', 'DeckSide', 'Deck'], drop_first=True)
_input0 = pd.get_dummies(_input0, columns=['CryoSleep', 'HomePlanet', 'Destination', 'DeckSide', 'Deck'], drop_first=True)
_input1.head()
X_train = _input1.drop(['Transported', 'PassengerId'], axis=1)
y_train = _input1['Transported']
X_test = _input0.drop(['PassengerId'], axis=1).copy()
(X_train.shape, y_train.shape, X_test.shape)
parameters = {'gamma': [5], 'random_state': [42], 'eval_metric': ['auc'], 'objective': ['binary:logistic'], 'min_child_weight': [1], 'subsample': [1], 'colsample_bytree': [0.7], 'max_depth': [8], 'n_estimators': [1000], 'learning_rate': [0.1]}
xgb = XGBClassifier()
xgb_cv = GridSearchCV(estimator=xgb, cv=5, param_grid=parameters, n_jobs=-1)