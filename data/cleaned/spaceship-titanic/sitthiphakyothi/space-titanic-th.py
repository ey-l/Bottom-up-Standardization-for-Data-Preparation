import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')




train.info()
test.info()
train.Transported = train.Transported.astype('float')
test['Transported'] = 0.5
dataset = pd.concat([train, test], axis=0)
dataset.reset_index(inplace=True, drop=True)
dataset.CryoSleep = dataset.CryoSleep.astype('float')
dataset.VIP = dataset.VIP.astype('float')
tmp = dataset.CryoSleep == 1
dataset.RoomService[tmp] = 0
dataset.FoodCourt[tmp] = 0
dataset.ShoppingMall[tmp] = 0
dataset.Spa[tmp] = 0
dataset.VRDeck[tmp] = 0
dataset.RoomService.fillna(0, inplace=True)
dataset.FoodCourt.fillna(0, inplace=True)
dataset.ShoppingMall.fillna(0, inplace=True)
dataset.Spa.fillna(0, inplace=True)
dataset.VRDeck.fillna(0, inplace=True)
dataset['Money'] = dataset.RoomService + dataset.FoodCourt + dataset.ShoppingMall + dataset.Spa + dataset.VRDeck
tmp = (dataset.Money == 0) & (dataset.Age >= 13) & pd.isna(dataset.CryoSleep)
dataset.CryoSleep[tmp] = 1
tmp = (dataset.Money > 0) & pd.isna(dataset.CryoSleep)
dataset.CryoSleep[tmp] = 0
avg_child_age = dataset.Age[dataset.Age < 13].mean()
avg_adult_age = dataset.Age[dataset.Age >= 13].mean()
tmp = (dataset.Money == 0) & (dataset.CryoSleep == 0) & pd.isna(dataset.Age)
dataset.Age[tmp] = avg_child_age
tmp = ((dataset.VIP == 1) | (dataset.Money > 0)) & pd.isna(dataset.Age)
dataset.Age[tmp] = avg_adult_age
tmp = (dataset.Age < 18) & pd.isna(dataset.VIP)
dataset.VIP[tmp] = 0
avg_VIP_spending = dataset.Money[(dataset.VIP == 1) & (dataset.CryoSleep == 0)].mean()
avg_nonVIP_spending = dataset.Money[(dataset.VIP == 0) & (dataset.CryoSleep == 0) & (dataset.Age > 18)].mean()
tmp = pd.isna(dataset.VIP) & (dataset.Money > 0.5 * (avg_VIP_spending - avg_nonVIP_spending) + avg_nonVIP_spending)
dataset.VIP[tmp] = 1
tmp = pd.isna(dataset.VIP)
dataset.VIP[tmp] = 0
del avg_VIP_spending, avg_nonVIP_spending, tmp
dataset['GroupID'] = dataset.PassengerId.str.slice(stop=4).astype('float')
groups = pd.get_dummies(dataset.GroupID).sum()
groups = pd.DataFrame(groups, columns=['noPassengers'])
groups['inGroup'] = groups.noPassengers > 1
dataset['inGroup'] = dataset.GroupID
dataset.inGroup = dataset.inGroup.replace(groups.inGroup.index, groups.inGroup.values.astype('float'))

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
GroupID = dataset.GroupID.copy()
GroupID.replace(groups.Cabin.index, groups.Cabin.values, inplace=True)
tmp = pd.isna(dataset.Cabin) & (dataset.inGroup == 1)
dataset.Cabin[tmp] = GroupID[tmp]
GroupID = dataset.GroupID.copy()
GroupID.replace(groups.Destination.index, groups.Destination.values, inplace=True)
tmp = pd.isna(dataset.Destination) & (dataset.inGroup == 1)
dataset.Destination[tmp] = GroupID[tmp]
del GroupID, groups, tmp, x, y

tmp = pd.isna(dataset.Cabin)
dataset.Cabin[tmp] = 'X/-1/X'
print('Added dummy cabin:    ', tmp.sum())
dataset['Deck'] = dataset.Cabin.str.slice(stop=1)
dataset['Side'] = dataset.Cabin.str.slice(start=-1)
dataset['cNum'] = dataset.Cabin.str.slice(start=2, stop=-2).astype('float')
fig = plt.figure(figsize=(16, 10))
grp = dataset.groupby(['Deck', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 1)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Deck', 'Destination'])['Destination'].size().unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 2)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Side', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 3)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Side', 'Destination'])['Destination'].size().unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 4)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Destination', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 5)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
grp = dataset.groupby(['Deck', 'Side'])['Side'].size().unstack().fillna(0).astype('int')
sub = fig.add_subplot(2, 4, 6)
sns.heatmap(grp, annot=True, linewidths=0.5, fmt='d', cbar=False)
sub = fig.add_subplot(2, 4, 7)
sns.countplot(data=dataset, x='Deck', palette='RdYlBu')
sub = fig.add_subplot(2, 4, 8)
sns.countplot(data=dataset, x='Side', palette='RdYlBu')
del fig, grp, sub, tmp
tmp = pd.isna(dataset.HomePlanet) & (dataset.Deck == 'G')
dataset.HomePlanet[tmp] = 'Earth'
tmp = pd.isna(dataset.HomePlanet) & (dataset.Destination == 'PSO J318.5-22')
dataset.HomePlanet[tmp] = 'Earth'
tmp = pd.isna(dataset.HomePlanet) & ((dataset.Deck == 'A') | (dataset.Deck == 'B') | (dataset.Deck == 'C') | (dataset.Deck == 'T'))
dataset.HomePlanet[tmp] = 'Earth'
tmp = (dataset.Deck == 'X') & (dataset.HomePlanet == 'Earth')
dataset.Deck[tmp] = 'G'
tmp = (dataset.Deck == 'X') & (dataset.HomePlanet == 'Europa')
dataset.Deck[tmp] = 'B'
tmp = (dataset.Deck == 'X') & (dataset.HomePlanet == 'Mars')
dataset.Deck[tmp] = 'F'
data = np.random.randint(0, 2, size=(dataset.shape[0], 1))
tmp = dataset.Deck == 'X'
dataset.Deck[tmp & (data[:, 0] == 0)] = 'F'
dataset.Deck[tmp & (data[:, 0] == 1)] = 'G'
data = np.random.randint(0, 2, size=(dataset.shape[0], 1))
tmp = dataset.Side == 'X'
dataset.Side[tmp & (data[:, 0] == 0)] = 'S'
dataset.Side[tmp & (data[:, 0] == 1)] = 'P'
del data, tmp
dataset['DeckSide'] = dataset.Deck + '-' + dataset.Side
grp = np.sort(dataset.DeckSide.unique())
sns.scatterplot(data=dataset, x='cNum', y='GroupID', hue='DeckSide', hue_order=grp).set(title='GroupID vs. Cabin Number')
from sklearn.linear_model import LinearRegression
missing = dataset.cNum == -1
for DeckSide in grp:
    index = dataset.DeckSide == DeckSide
    if sum(index & missing) > 0:
        x = dataset.GroupID.loc[index & ~missing].to_numpy().reshape(-1, 1)
        y = dataset.cNum.loc[index & ~missing].to_numpy().reshape(-1, 1)