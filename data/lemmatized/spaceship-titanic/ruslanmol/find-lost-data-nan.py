import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.options.display.float_format = '{:,.2f}'.format
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1 = pd.concat([_input1.drop(['Transported'], axis=1), _input0], axis=0)
_input1 = _input1.reset_index(drop=True, inplace=False)
combine = [_input1]
_input1.head()
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
from collections import Counter
for dataset in combine:
    dataset['ID1'] = dataset.PassengerId.str.split('_', n=1, expand=True)[0]
    dataset['ID1'] = dataset['ID1'].astype(int)
    dataset['ID2'] = dataset.PassengerId.str.split('_', n=1, expand=True)[1]
    dataset['ID2'] = dataset['ID2'].astype(int)
    dataset['ID1'] = dataset.PassengerId.str.split('_', n=1, expand=True)[0]
    CountID = Counter(dataset['ID1'])
    dataset['ID1_count'] = dataset['ID1'].map(CountID)
    dataset['Name1'] = dataset.Name.str.split(' ', n=1, expand=True)[0]
    dataset['Name2'] = dataset.Name.str.split(' ', n=1, expand=True)[1]
    dataset['Cabin1'] = dataset.Cabin.str.split('/', n=2, expand=True)[0]
    dataset['Cabin2'] = dataset.Cabin.str.split('/', n=2, expand=True)[1]
    dataset['Cabin2'] = dataset['Cabin2'].replace(np.nan, -1).astype(int)
    dataset['Cabin2'] = dataset['Cabin2'].replace(-1, np.nan)
    CountCabin = Counter(dataset['Cabin2'])
    dataset['Cabin2_count'] = dataset['Cabin2'].map(CountCabin)
    dataset['Cabin3'] = dataset.Cabin.str.split('/', n=2, expand=True)[2]
    dataset['VIP'] = dataset['VIP'].replace(np.nan, -1).astype(int)
    dataset['VIP'] = dataset['VIP'].replace(-1, np.nan)
    dataset['CryoSleep'] = dataset['CryoSleep'].replace(np.nan, -1).astype(int)
    dataset['CryoSleep'] = dataset['CryoSleep'].replace(-1, np.nan)
combine = [_input1]
for dataset in combine:
    dataset['AgeGroup'] = np.nan
    for i in range(7):
        dataset.loc[(dataset['Age'] >= 12 * i) & (dataset['Age'] < 12 * (i + 1)), 'AgeGroup'] = i + 1
Money = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for dataset in combine:
    for i in Money:
        dataset.loc[dataset[i] > 0, i] = 1
_input1['Cabin'].isnull().sum()
for dataset in combine:
    CabinValue = dataset[['ID1', 'Cabin']]
    CabinDict = dict(zip(CabinValue['ID1'], CabinValue['Cabin']))
    dataset['Cabin'] = dataset['ID1'].map(CabinDict)
_input1['Cabin'].isnull().sum()
_input1['HomePlanet'].isnull().sum()
PlanetName = dataset[dataset['HomePlanet'].notnull()]['HomePlanet'].unique()
for dataset in combine:
    for i in PlanetName:
        Planet_ID = dataset.loc[dataset['ID1'].notnull() & (dataset['HomePlanet'] == i)]['ID1'].values
        IndexPlanet = dataset.loc[dataset['ID1'].isin(Planet_ID)].index
        for ii in IndexPlanet:
            dataset.loc[dataset['HomePlanet'].index == ii, 'HomePlanet'] = i
        Planet_ID = dataset.loc[dataset['Name2'].notnull() & (dataset['HomePlanet'] == i)]['Name2'].values
        IndexPlanet = dataset.loc[dataset['Name2'].isin(Planet_ID)].index
        for ii in IndexPlanet:
            dataset.loc[dataset['HomePlanet'].index == ii, 'HomePlanet'] = i
        Planet_ID = dataset.loc[dataset['Cabin'].notnull() & (dataset['HomePlanet'] == i)]['Cabin'].values
        IndexPlanet = dataset.loc[dataset['Cabin'].isin(Planet_ID)].index
        for ii in IndexPlanet:
            dataset.loc[dataset['HomePlanet'].index == ii, 'HomePlanet'] = i
print(pd.crosstab(_input1['Cabin1'], _input1['HomePlanet']))
for dataset in combine:
    CabEuropa = {'A', 'B', 'C', 'T'}
    indexHomeCab1G = dataset.loc[_input1['Cabin1'].isin(CabEuropa) & _input1['HomePlanet'].isnull()].index
    for i in indexHomeCab1G:
        dataset.loc[dataset['HomePlanet'].index == i, 'HomePlanet'] = 'Europa'
    CabEarth = {'G'}
    indexHomeCab1G = dataset.loc[_input1['Cabin1'].isin(CabEarth) & _input1['HomePlanet'].isnull()].index
    for i in indexHomeCab1G:
        dataset.loc[dataset['HomePlanet'].index == i, 'HomePlanet'] = 'Earth'
_input1['HomePlanet'].isnull().sum()
_input1['Route'] = np.nan

def PlanetRoute(dataset):
    indexEaT = dataset.loc[(dataset['HomePlanet'] == 'Earth') & (dataset['Destination'] == 'TRAPPIST-1e')].index
    for i in indexEaT:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'EaT'
    indexEaC = dataset.loc[(dataset['HomePlanet'] == 'Earth') & (dataset['Destination'] == '55 Cancri e')].index
    for i in indexEaC:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'EaC'
    indexEaP = dataset.loc[(dataset['HomePlanet'] == 'Earth') & (dataset['Destination'] == 'PSO J318.5-22')].index
    for i in indexEaP:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'EaP'
    indexMT = dataset.loc[(dataset['HomePlanet'] == 'Mars') & (dataset['Destination'] == 'TRAPPIST-1e')].index
    for i in indexMT:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'MT'
    indexMC = dataset.loc[(dataset['HomePlanet'] == 'Mars') & (dataset['Destination'] == '55 Cancri e')].index
    for i in indexMC:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'MC'
    indexMP = dataset.loc[(dataset['HomePlanet'] == 'Mars') & (dataset['Destination'] == 'PSO J318.5-22')].index
    for i in indexMP:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'MP'
    indexEuT = dataset.loc[(dataset['HomePlanet'] == 'Europa') & (dataset['Destination'] == 'TRAPPIST-1e')].index
    for i in indexEuT:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'Eut'
    indexEuC = dataset.loc[(dataset['HomePlanet'] == 'Europa') & (dataset['Destination'] == '55 Cancri e')].index
    for i in indexEuC:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'EuC'
    indexEuP = dataset.loc[(dataset['HomePlanet'] == 'Europa') & (dataset['Destination'] == 'PSO J318.5-22')].index
    for i in indexEuP:
        dataset.loc[dataset['Route'].index == i, 'Route'] = 'EuP'
combine = [_input1]
PlanetRoute(_input1)
_input1['VIP'].isnull().sum()
print(pd.crosstab(_input1['Cabin1'], _input1['VIP']))
for dataset in combine:
    indexVIPCabin = dataset.loc[dataset['Cabin1'].isin({'G', 'T'}) & dataset['VIP'].isnull()].index
    for i in indexVIPCabin:
        dataset.loc[dataset['VIP'].index == i, 'VIP'] = 0
print(pd.crosstab(_input1['HomePlanet'], _input1['VIP']))
for dataset in combine:
    indexVIPHome = dataset.loc[(dataset['HomePlanet'] == 'Earth') & dataset['VIP'].isnull()].index
    for i in indexVIPHome:
        dataset.loc[dataset['VIP'].index == i, 'VIP'] = 0
print(pd.crosstab(_input1['Route'], _input1['VIP']))
for dataset in combine:
    VIPRoute = {'EaC', 'EaP', 'EaT', 'MC'}
    indexVIPRoute = dataset.loc[dataset['Route'].isin(VIPRoute) & dataset['VIP'].isnull()].index
    for i in indexVIPRoute:
        dataset.loc[dataset['VIP'].index == i, 'VIP'] = 0
print(pd.crosstab(_input1['VIP'], _input1['AgeGroup']))
for dataset in combine:
    index = dataset.loc[dataset['VIP'].isnull() & dataset['AgeGroup'].isin({1})].index
    for i in index:
        dataset.loc[dataset['VIP'].index == i, 'VIP'] = 0
_input1['VIP'].isnull().sum()
for dataset in combine:
    for i in Money:
        print(dataset[i].isnull().sum())
MoneyName = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for Names in MoneyName:
    print(pd.crosstab(_input1[Names], _input1['CryoSleep']))
    for dataset in combine:
        indexMoneyCryoSleep = dataset.loc[dataset[Names].isnull() & (dataset['CryoSleep'] == 1)].index
        for i in indexMoneyCryoSleep:
            dataset.loc[dataset['CryoSleep'].index == i, Names] = 0
MoneyName = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for Names in MoneyName:
    print(pd.crosstab(_input1['AgeGroup'], _input1[Names]))
    for dataset in combine:
        index = dataset.loc[dataset[Names].isnull() & (dataset['AgeGroup'] == 1)].index
        for i in index:
            dataset.loc[dataset[Names].index == i, Names] = 0
Money = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for dataset in combine:
    dataset['Money'] = dataset[Money].sum(axis=1)
    for i in Money:
        ID = dataset.loc[dataset[i].isnull()].index
        for ii in ID:
            dataset.loc[dataset[i].index == ii, 'Money'] = np.nan
for dataset in combine:
    for i in Money:
        print(dataset[i].isnull().sum())
MoneyName = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for Names in MoneyName:
    for dataset in combine:
        indexMoneyCryoSleep = dataset.loc[dataset['CryoSleep'].isnull() & (dataset[Names] == 1)].index
        for i in indexMoneyCryoSleep:
            dataset.loc[dataset['CryoSleep'].index == i, 'CryoSleep'] = 0
_input1.columns
'HOW IT WORKS? '
'Функция случайным образом перемешивает признаки, \nдалее делаем из признаков таблицы пересенчения, \nкак было сделано вручную выше,\nнаходим значения близкие к границам правдоподобности 0.999 = 1, 0.001 =0\nвносим эти значения'
'The function randomly shuffles features,\nthen we make the intersection table from the signs,\nas done manually above,\nfind values \u200b\u200bclose to the likelihood limits 0.999 = 1, 0.001 =0\nenter these values'
import random as rnd
from itertools import combinations

def AutoImputerFull(dataset, title, coefmax=0.999, total=100, Bool=False):
    coefmin = 1 - coefmax
    if title == 'VIP':
        coefmax = 0.999
        coefmin = 0.001
    print('Number of empty values before validation ' + title + ': ' + str(sum(dataset[title].isnull())))
    Col_for_crosstab = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'ID2', 'ID1_count', 'Cabin1', 'Cabin3', 'AgeGroup', 'Route', 'Money']
    Col_for_crosstab.remove(title)
    Combination_crosstab = list(combinations(Col_for_crosstab, 3))
    for (Col1, Col2, Col3) in Combination_crosstab:
        tempdf = dataset[[Col1, Col2, Col3, title]].groupby([Col1, Col2, Col3], as_index=True).mean().sort_values(by=title, ascending=False)
        tempdfmax = tempdf[tempdf[title] > coefmax]
        tempdfmin = tempdf[tempdf[title] < coefmin]
        if len(tempdfmax) > 0:
            tempdfmax = tempdfmax.reset_index().reindex()
            for ii in range(len(tempdfmax)):
                index = dataset.loc[(dataset[Col1] == tempdfmax.iloc[ii][Col1]) & (dataset[Col2] == tempdfmax.iloc[ii][Col2]) & (dataset[Col3] == tempdfmax.iloc[ii][Col3]) & dataset[title].isnull()].index
                index1 = dataset.loc[(dataset[Col1] == tempdfmax.iloc[ii][Col1]) & (dataset[Col2] == tempdfmax.iloc[ii][Col2]) & (dataset[Col3] == tempdfmax.iloc[ii][Col3]) & (dataset[title] == 1)].index
                if len(index1) > total:
                    if len(index) > 0:
                        print('Number of empty values:' + str(len(index)) + 'unit' + ' from ' + str(len(index1)) + ', Mean: ' + str(round(tempdfmax.iloc[ii][title], 2)) + '## ' + str(dict(tempdfmax.iloc[ii][:-1])))
                        for iii in index:
                            if rnd.random() <= tempdfmax.iloc[ii][title]:
                                dataset.loc[dataset[title].index == iii, title] = 1
        elif Bool == True:
            if len(tempdfmin) > 0:
                tempdfmin = tempdfmin.reset_index().reindex()
                for ii in range(len(tempdfmin)):
                    index = dataset.loc[(dataset[Col1] == tempdfmin.iloc[ii][Col1]) & (dataset[Col2] == tempdfmin.iloc[ii][Col2]) & (dataset[Col3] == tempdfmin.iloc[ii][Col3]) & dataset[title].isnull()].index
                    index0 = dataset.loc[(dataset[Col1] == tempdfmin.iloc[ii][Col1]) & (dataset[Col2] == tempdfmin.iloc[ii][Col2]) & (dataset[Col3] == tempdfmin.iloc[ii][Col3]) & (dataset[title] == 0)].index
                    if len(index0) > total:
                        if len(index) > 0:
                            print('Number of empty values:' + str(len(index)) + 'unit' + ' from ' + str(len(index0)) + ', Mean: ' + str(round(tempdfmin.iloc[ii][title], 2)) + '## ' + str(dict(tempdfmin.iloc[ii][:-1])))
                            for iii in index:
                                if rnd.random() <= 1 - tempdfmin.iloc[ii][title]:
                                    dataset.loc[dataset[title].index == iii, title] = 0
    print('Number of empty values after validation  ' + title + ': ' + str(sum(dataset[title].isnull())))
_input1.isnull().sum().sort_values(ascending=False)
_input1.isnull().sum().sort_values(ascending=False)