import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.options.display.float_format = '{:,.2f}'.format
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
_input1 = pd.concat([_input1.drop(['Transported'], axis=1), _input0], axis=0)
_input1 = _input1.reset_index(drop=True, inplace=False)
combine = [_input1]
_input1.head()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')

def update(TargetUpdate):
    tempdf = _input1[:8693].copy()
    TargetUpdate = pd.concat([tempdf, TargetUpdate['Transported']], axis=1)
    return TargetUpdate
from collections import Counter
for dataset in combine:
    dataset['ID1'] = dataset.PassengerId.str.split('_', n=1, expand=True)[0]
    dataset['ID2'] = dataset.PassengerId.str.split('_', n=1, expand=True)[1]
    dataset['ID1'] = dataset.PassengerId.str.split('_', n=1, expand=True)[0]
    CountID = Counter(dataset['ID1'])
    dataset['ID1_count'] = dataset['ID1'].map(CountID)
    dataset['ID1'] = dataset['ID1'].astype(int)
    dataset['ID2'] = dataset['ID2'].astype(int)
    dataset['Name1'] = dataset.Name.str.split(' ', n=1, expand=True)[0]
    CountName1 = Counter(dataset['Name1'])
    dataset['Name1Count'] = dataset['Name1'].map(CountName1)
    dataset['NameLen1'] = dataset['Name1'].str.count('\\S')
    dataset['Name2'] = dataset.Name.str.split(' ', n=1, expand=True)[1]
    CountName2 = Counter(dataset['Name2'])
    dataset['Name2Count'] = dataset['Name2'].map(CountName2)
    dataset['NameLen2'] = dataset['Name2'].str.count('\\S')
    dataset['NameLen'] = dataset['NameLen1'] + dataset['NameLen2']
    dataset['VIP'] = dataset['VIP'].replace(np.nan, -1).astype(int)
    dataset['VIP'] = dataset['VIP'].replace(-1, np.nan)
    dataset['CryoSleep'] = dataset['CryoSleep'].replace(np.nan, -1).astype(int)
    dataset['CryoSleep'] = dataset['CryoSleep'].replace(-1, np.nan)
    dataset = dataset.drop(['PassengerId', 'Name'], axis=1, inplace=False)
print('#Lost Before:', _input1['Cabin'].isnull().sum())
CabinValue = _input1[_input1['Cabin'].notnull()][['ID1', 'Cabin']]
CabinDict = dict(zip(CabinValue['ID1'], CabinValue['Cabin']))
for dataset in combine:
    index = dataset.loc[dataset['Cabin'].isnull()].index
    dataset.loc[index, 'Cabin'] = dataset.iloc[index, :]['ID1'].map(CabinDict)
print('#Lost After:', _input1['Cabin'].isnull().sum())
print('#Lost Before:', _input1['Cabin'].isnull().sum())
NameValue = _input1[_input1['Cabin'].notnull() & _input1['Name2'].notnull()][['Name2', 'Cabin']]
NameDict = dict(zip(NameValue['Name2'], NameValue['Cabin']))
for dataset in combine:
    index = dataset.loc[dataset['Cabin'].isnull()].index
    dataset.loc[index, 'Cabin'] = dataset.iloc[index, :]['Name2'].map(NameDict)
print('#Lost After:', _input1['Cabin'].isnull().sum())
_input1['Cabin'] = _input1['Cabin'].fillna('E/500/S', inplace=False)
print('#Lost After:', _input1['Cabin'].isnull().sum())
for dataset in combine:
    dataset['Cabin1'] = dataset.Cabin.str.split('/', n=2, expand=True)[0]
    dataset['Cabin2'] = dataset.Cabin.str.split('/', n=2, expand=True)[1]
    dataset['Cabin2'] = dataset['Cabin2'].replace(np.nan, -1).astype(int)
    dataset['Cabin2'] = dataset['Cabin2'].replace(-1, np.nan)
    dataset['Cabin3'] = dataset.Cabin.str.split('/', n=2, expand=True)[2]
    dataset['Cabin3'] = dataset['Cabin3'].map({'S': 1, 'P': 0})
    dataset['RatioIDCabin'] = dataset['ID1'] / dataset['Cabin2']
_input1 = update(_input1)
fig = plt.figure(figsize=(10, 20))
plt.subplot(4, 1, 1)
sns.countplot(data=_input1, x='Cabin1', hue='Transported', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
plt.title('Cabin1')
plt.subplot(4, 1, 2)
sns.histplot(data=_input1, x='Cabin2', hue='Transported', binwidth=20)
plt.vlines(300, ymin=0, ymax=200, color='black')
plt.vlines(600, ymin=0, ymax=200, color='black')
plt.vlines(900, ymin=0, ymax=200, color='black')
plt.vlines(1200, ymin=0, ymax=200, color='black')
plt.title('Cabin number')
plt.xlim([0, 2000])
plt.subplot(4, 1, 3)
sns.countplot(data=_input1, x='Cabin3', hue='Transported')
plt.title('Cabin side')
fig.tight_layout()
plt.subplot(4, 1, 4)
sns.histplot(data=_input1, x='RatioIDCabin', hue='Transported', binwidth=20)
plt.vlines(20, ymin=0, ymax=3500, color='black')
plt.title('Cabin number')
plt.xlim([0, 45])
for dataset in combine:
    dataset['RatioIDCabin'] = dataset['RatioIDCabin'].apply(lambda x: 1 if x > 20 else 0)
    dataset['CabinCategory'] = np.nan
    for i in range(7):
        dataset.loc[(dataset['Cabin2'] >= 300 * i) & (dataset['Cabin2'] < 300 * (i + 1)), 'CabinCategory'] = i + 1
    dataset['CabinCategory'] = dataset['CabinCategory'].apply(lambda x: 1 if x in {1, 3, 4} else 0)
_input1 = update(_input1)
fig = plt.figure(figsize=(10, 3))
plt.subplot(2, 1, 1)
sns.countplot(data=_input1, x='RatioIDCabin', hue='Transported')
fig = plt.figure(figsize=(10, 3))
plt.subplot(2, 1, 2)
sns.countplot(data=_input1, x='CabinCategory', hue='Transported')
fig = plt.figure(figsize=(15, 3))
plt.subplot(1, 1, 1)
sns.countplot(data=_input1, x='Cabin1', hue='HomePlanet', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
print(pd.crosstab(_input1['Cabin1'], _input1['HomePlanet']))
print('#Lost after:', _input1['HomePlanet'].isnull().sum())
for dataset in combine:
    CabEuropa = {'A', 'B', 'C', 'T'}
    indexHomeCab1G = dataset.loc[_input1['Cabin1'].isin(CabEuropa) & _input1['HomePlanet'].isnull()].index
    for i in indexHomeCab1G:
        dataset.loc[dataset['HomePlanet'].index == i, 'HomePlanet'] = 'Europa'
    CabEarth = {'G'}
    indexHomeCab1G = dataset.loc[_input1['Cabin1'].isin(CabEarth) & _input1['HomePlanet'].isnull()].index
    for i in indexHomeCab1G:
        dataset.loc[dataset['HomePlanet'].index == i, 'HomePlanet'] = 'Earth'
print('#Lost before:', _input1['HomePlanet'].isnull().sum())
print('#Lost after:', _input1['HomePlanet'].isnull().sum())
PlanetName = dataset[dataset['HomePlanet'].notnull()]['HomePlanet'].unique()
for dataset in combine:
    for i in PlanetName:
        Planet_ID = dataset.loc[dataset['ID1'].notnull() & (dataset['HomePlanet'] == i)]['ID1'].values
        IndexPlanet = dataset.loc[dataset['ID1'].isin(Planet_ID)].index
        for ii in IndexPlanet:
            dataset.loc[dataset['HomePlanet'].index == ii, 'HomePlanet'] = i
        Planet_ID = dataset.loc[dataset['Cabin'].notnull() & (dataset['HomePlanet'] == i)]['Cabin'].values
        IndexPlanet = dataset.loc[dataset['Cabin'].isin(Planet_ID)].index
        for ii in IndexPlanet:
            dataset.loc[dataset['HomePlanet'].index == ii, 'HomePlanet'] = i
        Planet_ID = dataset.loc[dataset['Name2'].notnull() & (dataset['HomePlanet'] == i)]['Name2'].values
        IndexPlanet = dataset.loc[dataset['Name2'].isin(Planet_ID)].index
        for ii in IndexPlanet:
            dataset.loc[dataset['HomePlanet'].index == ii, 'HomePlanet'] = i
print('#Lost before:', _input1['HomePlanet'].isnull().sum())
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
print('#Lost after:', _input1['HomePlanet'].isnull().sum())
for dataset in combine:
    indexHomeCab1D = dataset.loc[_input1['Cabin1'].isin({'D', 'E'}) & _input1['HomePlanet'].isnull()].index
    for i in indexHomeCab1D:
        dataset.loc[dataset['HomePlanet'].index == i, 'HomePlanet'] = 'Europa'
    indexHomeCab = dataset.loc[_input1['HomePlanet'].isnull()].index
    for i in indexHomeCab:
        dataset.loc[dataset['HomePlanet'].index == i, 'HomePlanet'] = 'Earth'
print('#Lost before:', _input1['HomePlanet'].isnull().sum())
name = _input1[_input1['Name1'].notnull()]['Name1'].values
suffix = ('e', 'a', 'y')
female = [f for f in name if f.endswith(suffix)]
for dataset in combine:
    index = dataset.loc[dataset['Name1'].isin(female)].index
    dataset['NameSex'] = 0
    for i in index:
        dataset.loc[dataset['Name1'].index == i, 'NameSex'] = 1
_input1 = _input1.drop(['Name1', 'Name2'], axis=1, inplace=False)
_input1 = update(_input1)
Names = [f for f in _input1.columns if 'Name' in f]
for (i, Name) in enumerate(Names):
    fig = plt.figure(figsize=(10, 12))
    plt.subplot(len(Names), 1, i + 1)
    sns.countplot(data=_input1, x=Name, hue='Transported')
    plt.title(Name)
for dataset in combine:
    dataset['Name1Count'] = dataset['Name1Count'].apply(lambda x: 1 if x <= 6 else 0)
    dataset['NameLen1'] = dataset['NameLen1'].apply(lambda x: 1 if x == 7 else 0)
    dataset['Name2Count'] = dataset['Name2Count'].apply(lambda x: 1 if x <= 6 else 0)
    dataset['NameLen'] = dataset['NameLen'].apply(lambda x: 1 if x in {8, 9, 13, 14, 15, 16} else 0)
    dataset = dataset.drop(['NameLen2'], axis=1, inplace=False)
_input1 = update(_input1)
Names = [f for f in _input1.columns if 'Name' in f]
for (i, Name) in enumerate(Names):
    fig = plt.figure(figsize=(10, 12))
    plt.subplot(len(Names), 1, i + 1)
    sns.countplot(data=_input1, x=Name, hue='Transported')
    plt.title(Name)
Names = [f for f in _input1.columns if 'Name' in f]
for Name in Names:
    print(f'#Lost before: {Name}:{_input1[Name].isnull().sum()}')
fig = plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
sns.countplot(data=_input1, x='ID1_count', hue='Transported')
plt.title('ID1_count')
for dataset in combine:
    dataset['ID1_count'] = dataset['ID1_count'].apply(lambda x: 1 if x in {1} else 0)
_input1 = update(_input1)
plt.subplot(2, 1, 2)
sns.countplot(data=_input1, x='ID1_count', hue='Transported')
plt.title('ID1_count')
_input1 = update(_input1)
fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(1, 1, 1)
sns.histplot(data=_input1, x='ID1', axes=ax, bins=30, kde=False, hue='Transported')
plt.vlines(3100, ymin=0, ymax=200, color='black')
plt.vlines(7100, ymin=0, ymax=200, color='black')
ax.set_title('ID1')
for dataset in combine:
    dataset['LuckyID'] = 0
    dataset.loc[(dataset['ID1'] > 3000) & (dataset['ID1'] < 7100), 'LuckyID'] = 1
_input1 = update(_input1)
fig = plt.figure(figsize=(20, 5))
plt.subplot(1, 1, 1)
sns.countplot(data=_input1, x='LuckyID', hue='Transported')
plt.title('LuckyID')
MoneyName = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1['Money'] = _input1[MoneyName].sum(axis=1)
_input1['Money1'] = 1
_input1.loc[_input1['Money'] < 1, 'Money1'] = 0
print(_input1.groupby(['HomePlanet', 'Money1', 'ID1_count', 'Cabin1'])['Age'].median().unstack().fillna(0))
print('#Lost before:', _input1['Age'].isnull().sum())
index = _input1.loc[_input1['Age'].isnull(), 'Age'].index
_input1.loc[_input1['Age'].isnull(), 'Age'] = _input1.groupby(['HomePlanet', 'Money1', 'ID1_count', 'Cabin1'])['Age'].transform(lambda x: x.fillna(x.median()))[index]
print('#Lost after:', _input1['Age'].isnull().sum())
_input1 = update(_input1)
plt.figure(figsize=(10, 4))
sns.histplot(data=_input1, x='Age', hue='Transported', binwidth=1, kde=True)
plt.vlines(12, ymin=0, ymax=200, color='black')
plt.vlines(18, ymin=0, ymax=200, color='black')
plt.title('Age distribution')
plt.xlabel('Age (years)')
for dataset in combine:
    dataset['Age_group'] = np.nan
    dataset.loc[dataset['Age'] <= 12, 'Age_group'] = 'Age12'
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] < 18), 'Age_group'] = 'Age17'
    dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] <= 25), 'Age_group'] = 'Age25'
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age_group'] = 'Age30'
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 50), 'Age_group'] = 'Age50'
    dataset.loc[dataset['Age'] > 50, 'Age_group'] = 'Age51+'
_input1 = update(_input1)
plt.figure(figsize=(10, 4))
g = sns.countplot(data=_input1, x='Age_group', hue='Transported', order=['Age12', 'Age17', 'Age25', 'Age30', 'Age50', 'Age51+'])
plt.title('Age group distribution')
print(pd.crosstab(_input1['Cabin1'], _input1['VIP']))
print('#Lost before:', _input1['VIP'].isnull().sum())
for dataset in combine:
    indexVIPCabin = dataset.loc[dataset['Cabin1'].isin({'G', 'T'}) & dataset['VIP'].isnull()].index
    for i in indexVIPCabin:
        dataset.loc[dataset['VIP'].index == i, 'VIP'] = 0
print('#Lost after:', _input1['VIP'].isnull().sum())
print(pd.crosstab(_input1['HomePlanet'], _input1['VIP']))
print('#Lost before:', _input1['VIP'].isnull().sum())
for dataset in combine:
    indexVIPHome = dataset.loc[(dataset['HomePlanet'] == 'Earth') & dataset['VIP'].isnull()].index
    for i in indexVIPHome:
        dataset.loc[dataset['VIP'].index == i, 'VIP'] = 0
print('#Lost after:', _input1['VIP'].isnull().sum())
print(pd.crosstab(_input1['VIP'], _input1['Age_group']))
print('#Lost before:', _input1['VIP'].isnull().sum())
for dataset in combine:
    index = dataset.loc[dataset['VIP'].isnull() & dataset['Age_group'].isin({'Age12', 'Age17'})].index
    for i in index:
        dataset.loc[dataset['VIP'].index == i, 'VIP'] = 0
print('#Lost after:', _input1['VIP'].isnull().sum())
print('#Lost before:', _input1['VIP'].isnull().sum())
_input1['VIP'] = _input1['VIP'].fillna(0, inplace=False)
print('#Lost after:', _input1['VIP'].isnull().sum())
_input1 = update(_input1)
fig = plt.figure(figsize=(20, 5))
plt.subplot(1, 1, 1)
sns.countplot(data=_input1, x='VIP', hue='Transported')
plt.title('VIP')
MoneyName = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for Names in MoneyName:
    print(f'{Names} #Lost after: , {_input1[Names].isnull().sum()}')
    for dataset in combine:
        indexMoneyCryoSleep = dataset.loc[dataset[Names].isnull() & (dataset['CryoSleep'] == 1)].index
        for i in indexMoneyCryoSleep:
            dataset.loc[dataset['CryoSleep'].index == i, Names] = 0
    print(f'{Names} #Lost before: , {_input1[Names].isnull().sum()}')
for dataset in combine:
    for Names in MoneyName:
        print(_input1.groupby(['CryoSleep', 'Destination', 'HomePlanet', 'Age_group'])[Names].mean()[:3])
        print('#Lost before:', dataset[Names].isnull().sum())
        index = dataset.loc[dataset[Names].isnull(), Names].index
        dataset.loc[dataset[Names].isnull(), Names] = dataset.groupby(['CryoSleep', 'Destination', 'HomePlanet', 'Age_group'])[Names].transform(lambda x: x.fillna(x.mean()))
        print('#Lost after:', dataset[Names].isnull().sum())
for dataset in combine:
    for Names in MoneyName:
        print('#Lost before:', dataset[Names].isnull().sum())
        dataset[Names] = dataset[Names].fillna(0, inplace=False)
        print('#Lost after:', dataset[Names].isnull().sum())

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    Q1 = dataframe[col_name].quantile(q1)
    Q3 = dataframe[col_name].quantile(q3)
    IQR = Q3 - Q1
    low_limit = Q1 - 1.5 * IQR
    up_limit = Q3 + 1.5 * IQR
    return (low_limit, up_limit)

def replace_with_thresholds(dataframe, variable):
    (low_limit, up_limit) = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
for col in MoneyName:
    replace_with_thresholds(_input1, col)
_input1['MoneyCount'] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].replace(0, np.nan, inplace=False).count(axis=1, numeric_only=True)
_input1 = _input1.drop(['Money1'], axis=1, inplace=False)
_input1 = update(_input1)
fig = plt.figure(figsize=(20, 5))
plt.subplot(1, 1, 1)
sns.countplot(data=_input1, x='MoneyCount', hue='Transported')
plt.title('MoneyCount')
_input1 = update(_input1)
fig = plt.figure(figsize=(20, 5))
plt.subplot(1, 1, 1)
sns.countplot(data=_input1, x='MoneyCount', hue='CryoSleep')
plt.title('MoneyCount')
MoneyName = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
print('#Lost before:', dataset['CryoSleep'].isnull().sum())
for Names in MoneyName:
    for dataset in combine:
        indexMoneyCryoSleep = dataset.loc[dataset['CryoSleep'].isnull() & (dataset[Names] > 0)].index
        for i in indexMoneyCryoSleep:
            dataset.loc[dataset['CryoSleep'].index == i, 'CryoSleep'] = 0
print('#Lost after:', dataset['CryoSleep'].isnull().sum())
print(_input1.groupby(['MoneyCount', 'Age_group'])['CryoSleep'].mean())
print('#Lost before:', dataset['CryoSleep'].isnull().sum())
for Names in MoneyName:
    for dataset in combine:
        indexMoneyCryoSleep = dataset.loc[~(dataset['Age_group'] == 'Age12') & (dataset['MoneyCount'] == 0) & dataset['CryoSleep'].isnull()].index
        for i in indexMoneyCryoSleep:
            dataset.loc[dataset['CryoSleep'].index == i, 'CryoSleep'] = 1
print('#Lost after:', dataset['CryoSleep'].isnull().sum())
for dataset in combine:
    print('#Lost before:', dataset['CryoSleep'].isnull().sum())
    dataset['CryoSleep'] = dataset['CryoSleep'].fillna(1, inplace=False)
    print('#Lost after:', dataset['CryoSleep'].isnull().sum())
_input1 = update(_input1)
fig = plt.figure(figsize=(20, 5))
plt.subplot(1, 1, 1)
sns.countplot(data=_input1, x='CryoSleep', hue='Transported')
plt.title('CryoSleep')
_input1 = update(_input1)
fig = plt.figure(figsize=(20, 5))
plt.subplot(1, 1, 1)
sns.countplot(data=_input1, x='Destination', hue='Transported')
plt.title('Destination')
for dataset in combine:
    print('#Lost before:', dataset['Destination'].isnull().sum())
    dataset['Destination'] = dataset['Destination'].fillna('TRAPPIST-1e', inplace=False)
    print('#Lost after:', dataset['Destination'].isnull().sum())
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
_input1 = update(_input1)
X = _input1.drop('Transported', axis=1)
y = _input1['Transported'].astype(int)
Columns_of_IntFloat = [f for f in X.columns if X[f].dtype != 'object']
Scaler_list = ['TestModel', MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler(), Normalizer(), QuantileTransformer(), PowerTransformer()]
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.3, random_state=97)
for scaler in Scaler_list:
    print('Test {} : '.format(str(scaler).split('(')[0]), end=' ')
    if scaler == 'TestModel':
        model = LogisticRegression()