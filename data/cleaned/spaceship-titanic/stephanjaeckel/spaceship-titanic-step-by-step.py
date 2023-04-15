import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
data_trn = pd.read_csv('data/input/spaceship-titanic/train.csv')
data_tst = pd.read_csv('data/input/spaceship-titanic/test.csv')


data_trn.Transported = data_trn.Transported.astype('float')
data_tst['Transported'] = 0.5
dataset = pd.concat([data_trn, data_tst], axis=0)
dataset.reset_index(inplace=True, drop=True)
dataset.sort_values(by=['PassengerId'], inplace=True)
dataset['Incomplete'] = 0
dataset.CryoSleep = dataset.CryoSleep.astype('float')
dataset.VIP = dataset.VIP.astype('float')
tmp = dataset.CryoSleep == 1
dataset.RoomService[tmp] = 0
dataset.FoodCourt[tmp] = 0
dataset.ShoppingMall[tmp] = 0
dataset.Spa[tmp] = 0
dataset.VRDeck[tmp] = 0
tmp = dataset.Age < 13
dataset.RoomService[tmp].fillna(0, inplace=True)
dataset.FoodCourt[tmp].fillna(0, inplace=True)
dataset.ShoppingMall[tmp].fillna(0, inplace=True)
dataset.Spa[tmp].fillna(0, inplace=True)
dataset.VRDeck[tmp].fillna(0, inplace=True)
tmp = dataset.RoomService.isna() | dataset.FoodCourt.isna() | dataset.ShoppingMall.isna() | dataset.Spa.isna() | dataset.VRDeck.isna()
dataset.Incomplete[tmp] = 1
dataset.RoomService.fillna(0, inplace=True)
dataset.FoodCourt.fillna(0, inplace=True)
dataset.ShoppingMall.fillna(0, inplace=True)
dataset.Spa.fillna(0, inplace=True)
dataset.VRDeck.fillna(0, inplace=True)
dataset['Money'] = dataset.RoomService + dataset.FoodCourt + dataset.ShoppingMall + dataset.Spa + dataset.VRDeck
tmp = (dataset.Money == 0) & (dataset.Age >= 13) & pd.isna(dataset.CryoSleep)
dataset.CryoSleep[tmp] = 1
print('Fixed sleeping adults:', tmp.sum())
tmp = (dataset.Money > 0) & pd.isna(dataset.CryoSleep)
dataset.CryoSleep[tmp] = 0
print('Fixed awake adults:   ', tmp.sum())
avg_child_age = dataset.Age[dataset.Age < 13].mean()
avg_adult_age = dataset.Age[dataset.Age >= 13].mean()
dataset.Incomplete[pd.isna(dataset.Age)] = 1
tmp = (dataset.Money == 0) & (dataset.CryoSleep == 0) & pd.isna(dataset.Age)
dataset.Age[tmp] = avg_child_age
print('Fixed child age:      ', tmp.sum())
tmp = ((dataset.VIP == 1) | (dataset.Money > 0)) & pd.isna(dataset.Age)
dataset.Age[tmp] = avg_adult_age
print('Fixed adult age:      ', tmp.sum())
tmp = (dataset.Age < 18) & pd.isna(dataset.VIP)
dataset.VIP[tmp] = 0
print('Fixed adolescent VIP: ', tmp.sum())
tmp = (dataset.VIP == 1) & (dataset.CryoSleep == 0)
avg_VIP_spending = dataset.Money[tmp].mean()
tmp = (dataset.VIP == 0) & (dataset.CryoSleep == 0) & (dataset.Age > 18)
avg_nonVIP_spending = dataset.Money[tmp].mean()
tmp = pd.isna(dataset.VIP) & (dataset.Money > avg_VIP_spending)
dataset.VIP[tmp] = 1
print('Fixed adult VIP:      ', tmp.sum())
tmp = pd.isna(dataset.VIP)
dataset.VIP[tmp] = 0
print('Fixed adult nonVIP:   ', tmp.sum())
del avg_VIP_spending, avg_nonVIP_spending, tmp
dataset['GroupID'] = dataset.PassengerId.str.slice(stop=4).astype('float')
groups = pd.get_dummies(dataset.GroupID).sum()
groups = pd.DataFrame(groups, columns=['noPassengers'])
groups['inGroup'] = (groups.noPassengers > 1).astype('float')
dataset['inGroup'] = dataset.GroupID
dataset.inGroup = dataset.inGroup.replace(groups.inGroup.index, groups.inGroup.values)
groups['HomePlanet'] = np.NaN
groups['Cabin'] = np.NaN
groups['Destination'] = np.NaN
for x in groups.index:
    if groups.noPassengers[x] > 1:
        GroupID = dataset.GroupID == x
        for y in dataset.index[GroupID]:
            if pd.notna(dataset.HomePlanet[y]):
                groups.HomePlanet[x] = dataset.HomePlanet[y]
            if pd.notna(dataset.Cabin[y]):
                groups.Cabin[x] = dataset.Cabin[y]
            if pd.notna(dataset.Destination[y]):
                groups.Destination[x] = dataset.Destination[y]
GroupID = dataset.GroupID.copy()
GroupID.replace(groups.HomePlanet.index, groups.HomePlanet.values, inplace=True)
tmp = pd.isna(dataset.HomePlanet) & (dataset.inGroup == 1)
dataset.HomePlanet[tmp] = GroupID[tmp]
print('Fixed HomePlanet:     ', tmp.sum())
GroupID = dataset.GroupID.copy()
GroupID.replace(groups.Cabin.index, groups.Cabin.values, inplace=True)
tmp = pd.isna(dataset.Cabin) & (dataset.inGroup == 1)
dataset.Cabin[tmp] = GroupID[tmp]
print('Fixed Cabin:          ', tmp.sum())
GroupID = dataset.GroupID.copy()
GroupID.replace(groups.Destination.index, groups.Destination.values, inplace=True)
tmp = pd.isna(dataset.Destination) & (dataset.inGroup == 1)
dataset.Destination[tmp] = GroupID[tmp]
print('Fixed Destination:    ', tmp.sum())
del GroupID, groups, tmp, x, y
tmp = pd.isna(dataset.Cabin)
dataset.Cabin[tmp] = 'X/-1/X'
print('Added dummy cabin:    ', tmp.sum())
dataset['Deck'] = dataset.Cabin.str.slice(stop=1)
dataset['Side'] = dataset.Cabin.str.slice(start=-1)
dataset['cNum'] = dataset.Cabin.str.slice(start=2, stop=-2).astype('float')
fig = plt.figure(figsize=(16, 10))
grp = dataset.groupby(['Deck', 'HomePlanet'])['HomePlanet'].size()
grp = grp.unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 1)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Deck', 'Destination'])['Destination'].size()
grp = grp.unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 2)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Side', 'HomePlanet'])['HomePlanet'].size()
grp = grp.unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 3)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Side', 'Destination'])['Destination'].size()
grp = grp.unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 4)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Destination', 'HomePlanet'])['HomePlanet'].size()
grp = grp.unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 5)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Deck', 'Side'])['Side'].size()
grp = grp.unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 6)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
sub = fig.add_subplot(2, 4, 7)
sns.countplot(data=dataset, x='Deck', palette='RdYlBu')
sub = fig.add_subplot(2, 4, 8)
sns.countplot(data=dataset, x='Side', palette='RdYlBu')
del fig, grp, sub, tmp
tmp = pd.isna(dataset.HomePlanet) & (dataset.Deck == 'G')
dataset.HomePlanet[tmp] = 'Earth'
print('Fixed Planet (Earth): ', tmp.sum())
tmp = pd.isna(dataset.HomePlanet) & (dataset.Destination == 'PSO J318.5-22')
dataset.HomePlanet[tmp] = 'Earth'
print('Fixed Planet (Earth): ', tmp.sum())
tmp = pd.isna(dataset.HomePlanet) & ((dataset.Deck == 'A') | (dataset.Deck == 'B') | (dataset.Deck == 'C') | (dataset.Deck == 'T'))
dataset.HomePlanet[tmp] = 'Europa'
print('Fixed Planet (Europ.):', tmp.sum())
tmp = (dataset.Deck == 'X') & (dataset.HomePlanet == 'Earth')
dataset.Deck[tmp] = 'G'
print('Fixed Deck (Earth):   ', tmp.sum())
del tmp
dataset['DeckSide'] = dataset.Deck + '-' + dataset.Side
tmp = dataset.Side != 'X'
grp = np.sort(dataset.DeckSide[tmp].unique())
fig = plt.figure(figsize=(10, 5))
chart = sns.scatterplot(data=dataset.loc[tmp], x='cNum', y='GroupID', hue='DeckSide', hue_order=grp)
chart.set(title='GroupID vs. Cabin Number')
room_table = np.zeros((int(dataset.cNum.max()) + 2, 16))
index = np.zeros(16, dtype='int')
for x in dataset.index:
    tmp = grp == dataset.DeckSide[x]
    if tmp.sum() == 1:
        room_table[int(dataset.cNum[x]), tmp] = dataset.GroupID[x]
        index[tmp] = int(dataset.cNum[x]) + 1
    else:
        for y in range(len(index)):
            room_table[index[y], y] = dataset.GroupID[x]
for y in range(len(index)):
    room_table[index[y], y] = 0
fixed_cNum = 0
fixed_DeckSide = 0
for x in dataset.index:
    if dataset.Side[x] == 'X':
        tmp = room_table == dataset.GroupID[x]
        val = np.any(tmp, axis=1)
        if sum(val) == 1:
            dataset.cNum[x] = np.argwhere(val != 0)[0][0]
            fixed_cNum += 1
        val = np.any(tmp, axis=0)
        if sum(val) == 1:
            dataset.DeckSide[x] = grp[np.argwhere(val != 0)[0][0]]
            fixed_DeckSide += 1
dataset.Deck = dataset.DeckSide.str.slice(stop=1)
dataset.Side = dataset.DeckSide.str.slice(start=-1)
print('Fixed Cabin Number:   ', fixed_cNum)
print('Fixed DeckSide:       ', fixed_DeckSide)
del room_table, index, x, y, tmp, fixed_cNum, fixed_DeckSide, val
tmp = (dataset.Deck == 'X') & (dataset.HomePlanet == 'Europa')
dataset.Deck[tmp] = 'B'
print('Fixed Deck (Europa):  ', tmp.sum())
tmp = (dataset.Deck == 'X') & (dataset.HomePlanet == 'Mars')
dataset.Deck[tmp] = 'F'
print('Fixed Deck (Mars):    ', tmp.sum())
data = np.random.randint(0, 2, size=(dataset.shape[0], 1))
tmp = dataset.Deck == 'X'
dataset.Deck[tmp & (data[:, 0] == 0)] = 'F'
dataset.Deck[tmp & (data[:, 0] == 1)] = 'G'
print('Fixed Deck (rest):    ', tmp.sum())
data = np.random.randint(0, 2, size=(dataset.shape[0], 1))
tmp = dataset.Side == 'X'
dataset.Incomplete[tmp] = 1
dataset.Side[tmp & (data[:, 0] == 0)] = 'S'
dataset.Side[tmp & (data[:, 0] == 1)] = 'P'
print('Fixed Side:           ', tmp.sum())
from sklearn.linear_model import LinearRegression
dataset['DeckSide'] = dataset.Deck + '-' + dataset.Side
missing = dataset.cNum == -1
for DeckSide in grp:
    index = dataset.DeckSide == DeckSide
    print('Deck:', DeckSide, ', Passengers:', sum(index), ', Missing:', sum(index & missing))
    if sum(index & missing) > 0:
        x = dataset.GroupID.loc[index & ~missing].to_numpy().reshape(-1, 1)
        y = dataset.cNum.loc[index & ~missing].to_numpy().reshape(-1, 1)