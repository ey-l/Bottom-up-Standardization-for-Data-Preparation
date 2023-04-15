
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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
combine = [train_df, test_df]
plot(train_df)
pd.crosstab(train_df['HomePlanet'], train_df['CryoSleep'], normalize='index')
pd.crosstab(train_df['Destination'], train_df['CryoSleep'], normalize='index')
plot_correlation(train_df)
for dataset in combine:
    dataset['Spend'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
    dataset['HasSpend'] = 0
    dataset.loc[dataset['Spend'] > 0, 'HasSpend'] = 1
train_df.head()
plot(train_df, 'Spend')
plot(train_df, 'Spend', 'Transported')
for dataset in combine:
    dataset['Spend'] = np.log(dataset['Spend'])
train_df.head()
plt.figure(figsize=(10, 5))
sns.histplot(data=train_df, x='Spend', hue='Transported', bins=30)
for dataset in combine:
    dataset['Spend'] = dataset['Spend'].fillna(0)
    dataset['Spend'] = dataset['Spend'].replace([np.inf, -np.inf], 0)
dataset['Spend'].isna().sum()
CreditFeatures = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for var in CreditFeatures:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=train_df, x=var, hue='Transported', bins=30)
for dataset in combine:
    for col in dataset[CreditFeatures]:
        dataset[col] = np.log(pd.to_numeric(dataset[col], errors='coerce'))
        dataset[col].replace([-np.inf], 0, inplace=True)
train_df.head()
for dataset in combine:
    dataset['PaxGroup'] = dataset['PassengerId'].str[:4]
train_df.head()
for dataset in combine:
    dataset['PaxGroupSize'] = dataset['PaxGroup'].map(dataset['PaxGroup'].value_counts())
train_df.head()
train_df[['PaxGroupSize', 'Transported']].groupby(['PaxGroupSize'], as_index=False).mean().sort_values(by='Transported', ascending=False)
for dataset in combine:
    dataset.loc[dataset['PaxGroupSize'] == 1, 'PaxGroupSize'] = 0
    dataset.loc[(dataset['PaxGroupSize'] == 2) | (dataset['PaxGroupSize'] == 7), 'PaxGroupSize'] = 1
    dataset.loc[(dataset['PaxGroupSize'] == 3) | (dataset['PaxGroupSize'] == 5) | (dataset['PaxGroupSize'] == 6), 'PaxGroupSize'] = 2
    dataset.loc[dataset['PaxGroupSize'] == 4, 'PaxGroupSize'] = 3
    dataset.loc[dataset['PaxGroupSize'] == 8, 'PaxGroupSize'] = 4
train_df.head()
plot(train_df, 'PaxGroupSize')
plot(train_df, 'PaxGroupSize', 'Transported')
for dataset in combine:
    dataset['GivenName'] = dataset['Name'].str.split(' ', expand=True)[0]
    dataset['FamilyName'] = dataset['Name'].str.split(' ', expand=True)[1]
train_df.head()
for dataset in combine:
    dataset['Deck'] = dataset['Cabin'].str[:1]
train_df.head()
plot(train_df, 'Deck')
plot(train_df, 'Deck', 'Transported')
plot_missing(train_df, 'Deck', 'Transported')
train_df[['Deck', 'Transported']].groupby('Deck').mean().sort_values('Transported', ascending=False)
train_df['Deck'].isna().sum()
for dataset in combine:
    dataset['Deck'] = dataset['Deck'].transform(lambda x: x.fillna(x.mode()[0]))
train_df['Deck'].isna().sum()
for dataset in combine:
    dataset['DeckSide'] = dataset['Cabin'].str[-1:]
train_df.head()
plot(train_df, 'DeckSide')
plot(train_df, 'DeckSide', 'Transported')
plot_missing(train_df, 'DeckSide', 'Transported')
train_df[['DeckSide', 'Transported']].groupby('DeckSide').mean().sort_values('Transported', ascending=False)
train_df['DeckSide'].isna().sum()
for dataset in combine:
    dataset['DeckSide'] = dataset['DeckSide'].transform(lambda x: x.fillna(x.mode()[0]))
train_df['DeckSide'].isna().sum()
for dataset in combine:
    dataset['CabinNumber'] = dataset['Cabin'].str.split('/', expand=True)[1].astype('float').astype('Int16')
train_df.head()
for dataset in combine:
    dataset['CabinNumber'] = dataset['CabinNumber'].transform(lambda x: x.fillna(x.mode()[0]))
plot(train_df, 'CabinNumber')
plot(train_df, 'CabinNumber', 'Transported')
for dataset in combine:
    dataset.loc[dataset['CabinNumber'] <= 300, 'CabinNumber'] = 0
    dataset.loc[(dataset['CabinNumber'] > 300) & (dataset['CabinNumber'] <= 600), 'CabinNumber'] = 1
    dataset.loc[(dataset['CabinNumber'] > 600) & (dataset['CabinNumber'] <= 900), 'CabinNumber'] = 2
    dataset.loc[(dataset['CabinNumber'] > 900) & (dataset['CabinNumber'] <= 1200), 'CabinNumber'] = 3
    dataset.loc[(dataset['CabinNumber'] > 1200) & (dataset['CabinNumber'] <= 1500), 'CabinNumber'] = 4
    dataset.loc[(dataset['CabinNumber'] > 1500) & (dataset['CabinNumber'] <= 1800), 'CabinNumber'] = 5
    dataset.loc[dataset['CabinNumber'] > 1800, 'CabinNumber'] = 6
train_df.head()
plot(train_df, 'CabinNumber', 'Transported')
train_df[['CabinNumber', 'Transported']].groupby('CabinNumber').mean().sort_values('Transported', ascending=False)
plot(train_df, 'Age')
plot(train_df, 'Age', 'Transported')
plot_missing(train_df, 'Age', 'Transported')
for dataset in combine:
    dataset['Age'].replace(0, np.nan, inplace=True)
train_df['Age'].isna().sum()
for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(dataset.groupby(['HomePlanet', 'Destination', 'VIP', 'HasSpend', 'PaxGroupSize', 'Deck', 'DeckSide'])['Age'].transform('mean'))
train_df['Age'].isna().sum()
train_df.loc[train_df['Age'].isna()]
for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(dataset.groupby(['HasSpend', 'PaxGroupSize', 'CryoSleep'])['Age'].transform('mean'))
train_df['Age'].isna().sum()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Transported']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:
    dataset.loc[dataset['Age'] <= 17, 'Age'] = 0
    dataset.loc[dataset['Age'] > 17, 'Age'] = 1
    dataset['Age'] = dataset['Age'].astype('int')
train_df.head()
plot(train_df, 'Age', 'Transported')
plot(train_df, 'CryoSleep')
plot(train_df, 'CryoSleep', 'Transported')
plot_missing(train_df, 'CryoSleep', 'Transported')
for dataset in combine:
    dataset.loc[dataset['CryoSleep'].isna() & (dataset['HasSpend'] == 0), 'CryoSleep'] = 1
train_df['CryoSleep'].isna().sum()
for dataset in combine:
    dataset.loc[dataset['CryoSleep'].isna() & (dataset['HasSpend'] != 0), 'CryoSleep'] = 0
train_df['CryoSleep'].isna().sum()
train_df['CryoSleep']
for dataset in combine:
    dataset['CryoSleep'] = dataset['CryoSleep'].replace({'True': True, 'False': False}).astype('int')
train_df['CryoSleep']
plot(train_df, 'HomePlanet')
plot(train_df, 'HomePlanet', 'Transported')
plot_missing(train_df, 'HomePlanet', 'Transported')
train_df[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='HomePlanet', ascending=False)
pd.pivot_table(train_df, values='PassengerId', index='Deck', columns='HomePlanet', aggfunc='count')
for dataset in combine:
    dataset.loc[(dataset['Deck'] == 'G') & dataset['HomePlanet'].isna(), 'HomePlanet'] = 'Earth'
    dataset.loc[((dataset['Deck'] == 'A') | (dataset['Deck'] == 'B') | (dataset['Deck'] == 'C') | (dataset['Deck'] == 'T')) & dataset['HomePlanet'].isna(), 'HomePlanet'] = 'Europa'
train_df['HomePlanet'].isna().sum()
tmp = train_df.loc[~pd.isna(train_df['HomePlanet']), ['HomePlanet', 'PaxGroup']]
result = tmp.groupby(['PaxGroup', 'HomePlanet'])['HomePlanet'].count().unstack().isna().sum(axis=1)
result.describe()
labels = {'Earth': 1, 'Europa': 2, 'Mars': 3}
for dataset in combine:
    dataset['HomePlanet'] = dataset['HomePlanet'].map(labels)
    tmp = dataset[['HomePlanet', 'PaxGroup']]
    tmp['HomePlanet'] = tmp.groupby('PaxGroup').transform(lambda x: x.fillna(x.mean()))
    dataset['HomePlanet'] = tmp['HomePlanet']
train_df['HomePlanet'].isna().sum()
for dataset in combine:
    dataset['HomePlanet'] = dataset['HomePlanet'].transform(lambda x: x.fillna(x.mode()[0]))
train_df['HomePlanet'].isna().sum()
plot(train_df, 'Destination')
plot(train_df, 'Destination', 'Transported')
plot_missing(train_df, 'Destination', 'Transported')
train_df['Destination'].isna().sum()
pd.pivot_table(train_df, values='PassengerId', index='Deck', columns='Destination', aggfunc='count')
pd.pivot_table(train_df, values='PassengerId', index='HomePlanet', columns='Destination', aggfunc='count')
for dataset in combine:
    dataset['Destination'] = dataset['Destination'].transform(lambda x: x.fillna(x.mode()[0]))
train_df['Destination'].isna().sum()
plot(train_df, 'VIP')
plot(train_df, 'VIP', 'Transported')
plot_missing(train_df, 'VIP', 'Transported')
for dataset in combine:
    dataset['VIP'] = dataset['VIP'].transform(lambda x: x.fillna(x.mode()[0]))
train_df['VIP'].isna().sum()
for dataset in combine:
    dataset['VIP'] = dataset['VIP'].replace({'True': 1, 'False': 0}).astype('int')
for dataset in combine:
    dataset['FamilySize'] = dataset.groupby('FamilyName')['FamilyName'].transform('count')
train_df.head()
plot(train_df, 'FamilySize')
plot(train_df, 'FamilySize', 'Transported')
train_df[['FamilySize', 'Transported']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='FamilySize', ascending=False)
for dataset in combine:
    dataset['FamilySize'] = dataset['FamilySize'].fillna(dataset['FamilySize'].median()).astype('int')
plot(train_df, 'FamilySize', 'Transported')
train_df['FamilySize'].isna().sum()
train_df.head()
train_df = train_df.drop(['Cabin', 'Name', col, 'PaxGroup', 'AgeBand', 'GivenName', 'FamilyName', 'VIP', 'HasSpend', 'CabinNumber'], axis=1)
test_df = test_df.drop(['Cabin', 'Name', col, 'PaxGroup', 'GivenName', 'FamilyName', 'VIP', 'HasSpend', 'CabinNumber'], axis=1)
train_df.head()
train_df = pd.get_dummies(train_df, columns=['CryoSleep', 'HomePlanet', 'Destination', 'DeckSide', 'Deck'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['CryoSleep', 'HomePlanet', 'Destination', 'DeckSide', 'Deck'], drop_first=True)
train_df.head()
X_train = train_df.drop(['Transported', 'PassengerId'], axis=1)
y_train = train_df['Transported']
X_test = test_df.drop(['PassengerId'], axis=1).copy()
(X_train.shape, y_train.shape, X_test.shape)
parameters = {'gamma': [5], 'random_state': [42], 'eval_metric': ['auc'], 'objective': ['binary:logistic'], 'min_child_weight': [1], 'subsample': [1], 'colsample_bytree': [0.7], 'max_depth': [8], 'n_estimators': [1000], 'learning_rate': [0.1]}
xgb = XGBClassifier()
xgb_cv = GridSearchCV(estimator=xgb, cv=5, param_grid=parameters, n_jobs=-1)