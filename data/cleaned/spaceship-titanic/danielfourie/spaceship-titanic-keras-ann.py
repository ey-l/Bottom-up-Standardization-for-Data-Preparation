import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset.head()
dataset.dtypes
dataset = dataset.set_index('PassengerId')
dataset.shape[0]
dataset.isnull().sum()
dataset = dataset.drop(columns=['Name'], axis=1)
import seaborn as sns
sns.histplot(x=dataset['Age'], kde=True)
dataset['Age'].skew()
corr = dataset.corr()[['Transported']]
sns.heatmap(corr, annot=True)
dataset[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = dataset[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
dataset.isnull().sum()
dataset['Age'].fillna(value=int(dataset['Age'].mean()), inplace=True)
dataset.isnull().sum()
dataset = dataset.fillna(value='Missing')
dataset.isnull().sum()
dataset['Deck'] = ''
dataset['Side'] = ''
for rowNum in range(dataset.shape[0]):
    if dataset['Cabin'][rowNum] == 'Missing':
        dataset['Deck'][rowNum] = 'Missing'
        dataset['Side'][rowNum] = 'Missing'
    else:
        dataset['Deck'][rowNum] = dataset['Cabin'][rowNum][0]
        dataset['Side'][rowNum] = dataset['Cabin'][rowNum].split('/')[2]
dataset = dataset.drop(columns=['Cabin'], axis=1)
dataset.head()
dataset['AgeBin'] = pd.cut(dataset['Age'], bins=8, labels=False)
dataset = dataset.drop(columns=['Age'], axis=1)
dataset.head()
new_cols_order = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeBin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
dataset = dataset.reindex(columns=new_cols_order)
dataset.head()
for colName in dataset.columns:
    if colName in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        next
    else:
        dataset[colName] = dataset[colName].astype('category')
dataset.dtypes
for i in dataset.columns:
    print(f'The column "{i}" has {len(dataset[i].value_counts())} unique values.')
dataset['Transported'] = dataset['Transported'].astype(int)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
dataset[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = sc.fit_transform(dataset[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
dataset.head()
X_CategoricalFeatures = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeBin']
dataset = pd.get_dummies(data=dataset, columns=X_CategoricalFeatures, drop_first=True)
dataset.head()
transportedColumn = dataset.pop('Transported')
dataset = pd.concat([dataset, transportedColumn], axis=1)
dataset.head()
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier()