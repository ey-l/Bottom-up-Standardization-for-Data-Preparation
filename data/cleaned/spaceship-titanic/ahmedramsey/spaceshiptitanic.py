from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
y_train = train_data.Transported
test_id = test_data.PassengerId
train_data.replace({False: 0, True: 1}, inplace=True)
test_data.replace({False: 0, True: 1}, inplace=True)
train_data.head()
train_data.info()
train_data.describe()
for feature in train_data.columns.values:
    missing_values = train_data[train_data[feature].isnull()].shape[0]
    total_values = train_data.shape[0]
    print('{} - {:.2f}% Missing Values'.format(feature.ljust(15), missing_values / total_values * 100))
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
categorical_features = [feature for feature in train_data.columns.values if feature not in numerical_features and feature not in ['PassengerId', 'Name', 'Transported', 'Cabin']]
(_, ax) = plt.subplots(4, 2, figsize=(30, 30))
ax = ax.flatten()
for (i, feature) in enumerate(numerical_features):
    ax[i].set_title(feature)
    sns.histplot(data=train_data, x=feature, hue='Transported', kde=True, ax=ax[i])
(_, ax) = plt.subplots(2, 2, figsize=(30, 30))
ax = ax.flatten()
for (i, feature) in enumerate(categorical_features):
    sns.countplot(data=train_data, x=feature, ax=ax[i], hue='Transported')
(_, ax) = plt.subplots(2, 2, figsize=(30, 30))
ax = ax.flatten()
for (i, feature) in enumerate(categorical_features):
    sns.barplot(data=train_data, x=feature, y='Transported', ax=ax[i])
plt.figure(figsize=(30, 30))
sns.heatmap(train_data.corr(), annot=True, cmap='plasma')
for dataset in [train_data, test_data]:
    dataset.HomePlanet.fillna(train_data.HomePlanet.mode()[0], inplace=True)
    dataset.CryoSleep.fillna(train_data.CryoSleep.mode()[0], inplace=True)
    dataset.Destination.fillna(train_data.Destination.mode()[0], inplace=True)
    dataset.Age.fillna(train_data.Age.median(), inplace=True)
    dataset.VIP.fillna(train_data.VIP.mode()[0], inplace=True)
    dataset.RoomService.fillna(train_data.RoomService.median(), inplace=True)
    dataset.FoodCourt.fillna(train_data.FoodCourt.median(), inplace=True)
    dataset.ShoppingMall.fillna(train_data.ShoppingMall.median(), inplace=True)
    dataset.Spa.fillna(train_data.Spa.median(), inplace=True)
    dataset.VRDeck.fillna(train_data.VRDeck.median(), inplace=True)
for data in [train_data, test_data]:
    data['ServicesFee'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
no_members_in_team = pd.concat([train_data, test_data], axis=0).PassengerId.apply(lambda _id: _id.split('_')[0]).value_counts()
for data in [train_data, test_data]:
    data['TeamMembers'] = no_members_in_team[data.PassengerId.apply(lambda _id: _id.split('_')[0]).values].values
for data in [train_data, test_data]:
    data['Cabin_Deck'] = data.Cabin.apply(lambda cabin: None if str(cabin) == 'nan' else cabin.split('/')[0])
    data['Cabin_Num'] = data.Cabin.apply(lambda cabin: None if str(cabin) == 'nan' else cabin.split('/')[1]).astype('float64')
    data['Cabin_Side'] = data.Cabin.apply(lambda cabin: None if str(cabin) == 'nan' else cabin.split('/')[2])
(_, ax) = plt.subplots(1, 2, figsize=(60, 30))
for (i, feature) in enumerate(['ServicesFee', 'TeamMembers']):
    sns.histplot(data=train_data, x=feature, hue='Transported', ax=ax[i])
for dataset in [train_data, test_data]:
    dataset.Cabin_Deck.fillna(train_data.Cabin_Deck.mode()[0], inplace=True)
    dataset.Cabin_Num.fillna(train_data.Cabin_Num.mode()[0], inplace=True)
    dataset.Cabin_Side.fillna(train_data.Cabin_Side.mode()[0], inplace=True)
train_data.drop(columns=['Transported', 'Cabin', 'Name', 'PassengerId'], inplace=True)
test_data.drop(columns=['Cabin', 'Name', 'PassengerId'], inplace=True)
plt.figure(figsize=(15, 15))