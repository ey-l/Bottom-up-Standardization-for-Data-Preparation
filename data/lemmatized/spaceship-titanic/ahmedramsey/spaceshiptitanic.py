from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
y_train = _input1.Transported
test_id = _input0.PassengerId
_input1 = _input1.replace({False: 0, True: 1}, inplace=False)
_input0 = _input0.replace({False: 0, True: 1}, inplace=False)
_input1.head()
_input1.info()
_input1.describe()
for feature in _input1.columns.values:
    missing_values = _input1[_input1[feature].isnull()].shape[0]
    total_values = _input1.shape[0]
    print('{} - {:.2f}% Missing Values'.format(feature.ljust(15), missing_values / total_values * 100))
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
categorical_features = [feature for feature in _input1.columns.values if feature not in numerical_features and feature not in ['PassengerId', 'Name', 'Transported', 'Cabin']]
(_, ax) = plt.subplots(4, 2, figsize=(30, 30))
ax = ax.flatten()
for (i, feature) in enumerate(numerical_features):
    ax[i].set_title(feature)
    sns.histplot(data=_input1, x=feature, hue='Transported', kde=True, ax=ax[i])
(_, ax) = plt.subplots(2, 2, figsize=(30, 30))
ax = ax.flatten()
for (i, feature) in enumerate(categorical_features):
    sns.countplot(data=_input1, x=feature, ax=ax[i], hue='Transported')
(_, ax) = plt.subplots(2, 2, figsize=(30, 30))
ax = ax.flatten()
for (i, feature) in enumerate(categorical_features):
    sns.barplot(data=_input1, x=feature, y='Transported', ax=ax[i])
plt.figure(figsize=(30, 30))
sns.heatmap(_input1.corr(), annot=True, cmap='plasma')
for dataset in [_input1, _input0]:
    dataset.HomePlanet = dataset.HomePlanet.fillna(_input1.HomePlanet.mode()[0], inplace=False)
    dataset.CryoSleep = dataset.CryoSleep.fillna(_input1.CryoSleep.mode()[0], inplace=False)
    dataset.Destination = dataset.Destination.fillna(_input1.Destination.mode()[0], inplace=False)
    dataset.Age = dataset.Age.fillna(_input1.Age.median(), inplace=False)
    dataset.VIP = dataset.VIP.fillna(_input1.VIP.mode()[0], inplace=False)
    dataset.RoomService = dataset.RoomService.fillna(_input1.RoomService.median(), inplace=False)
    dataset.FoodCourt = dataset.FoodCourt.fillna(_input1.FoodCourt.median(), inplace=False)
    dataset.ShoppingMall = dataset.ShoppingMall.fillna(_input1.ShoppingMall.median(), inplace=False)
    dataset.Spa = dataset.Spa.fillna(_input1.Spa.median(), inplace=False)
    dataset.VRDeck = dataset.VRDeck.fillna(_input1.VRDeck.median(), inplace=False)
for data in [_input1, _input0]:
    data['ServicesFee'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
no_members_in_team = pd.concat([_input1, _input0], axis=0).PassengerId.apply(lambda _id: _id.split('_')[0]).value_counts()
for data in [_input1, _input0]:
    data['TeamMembers'] = no_members_in_team[data.PassengerId.apply(lambda _id: _id.split('_')[0]).values].values
for data in [_input1, _input0]:
    data['Cabin_Deck'] = data.Cabin.apply(lambda cabin: None if str(cabin) == 'nan' else cabin.split('/')[0])
    data['Cabin_Num'] = data.Cabin.apply(lambda cabin: None if str(cabin) == 'nan' else cabin.split('/')[1]).astype('float64')
    data['Cabin_Side'] = data.Cabin.apply(lambda cabin: None if str(cabin) == 'nan' else cabin.split('/')[2])
(_, ax) = plt.subplots(1, 2, figsize=(60, 30))
for (i, feature) in enumerate(['ServicesFee', 'TeamMembers']):
    sns.histplot(data=_input1, x=feature, hue='Transported', ax=ax[i])
for dataset in [_input1, _input0]:
    dataset.Cabin_Deck = dataset.Cabin_Deck.fillna(_input1.Cabin_Deck.mode()[0], inplace=False)
    dataset.Cabin_Num = dataset.Cabin_Num.fillna(_input1.Cabin_Num.mode()[0], inplace=False)
    dataset.Cabin_Side = dataset.Cabin_Side.fillna(_input1.Cabin_Side.mode()[0], inplace=False)
_input1 = _input1.drop(columns=['Transported', 'Cabin', 'Name', 'PassengerId'], inplace=False)
_input0 = _input0.drop(columns=['Cabin', 'Name', 'PassengerId'], inplace=False)
plt.figure(figsize=(15, 15))