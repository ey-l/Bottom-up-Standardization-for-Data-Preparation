import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.columns
_input0.columns
_input1.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1.describe()
_input1['HomePlanet'].value_counts()
_input1['CryoSleep'].value_counts()
Cabin_split = _input1['Cabin'].str.split('/', expand=True)
Cabin_split
print(len(Cabin_split[0].value_counts()))
print(len(Cabin_split[1].value_counts()))
print(len(Cabin_split[2].value_counts()))
print(Cabin_split[0].value_counts())
print(Cabin_split[2].value_counts())
_input1['cabin_deck'] = Cabin_split[0]
_input1['cabin_side'] = Cabin_split[2]
_input1.head()
_input1['Destination'].value_counts()
sns.histplot(_input1['Age'])
_input1['VIP'].value_counts()
sns.scatterplot(x='RoomService', data=_input1, y='Age')
sns.scatterplot(x='RoomService', data=_input1, y='FoodCourt')
sns.scatterplot(x='ShoppingMall', data=_input1, y='FoodCourt')
sns.scatterplot(x='ShoppingMall', data=_input1, y='Spa')
sns.scatterplot(x='VRDeck', data=_input1, y='Spa')
_input1['Transported'].value_counts()
name_split = _input1['Name'].str.split(' ', expand=True)
name_split
(name_split[1].value_counts() > 2).sum()
name_split[1].value_counts() > 2
_input1 = _input1.drop(columns=['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
_input1.head()
sns.boxplot(data=_input1, x='RoomService')
sns.histplot(data=_input1, x='RoomService', bins=5)
_input1 = pd.get_dummies(_input1, columns=['HomePlanet'], prefix='HomePlanet')
_input1 = pd.get_dummies(_input1, columns=['CryoSleep'], prefix='CryoSleep', drop_first=True)
_input1 = pd.get_dummies(_input1, columns=['Destination'], prefix='Destination')
_input1 = pd.get_dummies(_input1, columns=['VIP'], prefix='VIP', drop_first=True)
_input1 = pd.get_dummies(_input1, columns=['cabin_deck'], prefix='cabin_deck')
_input1 = pd.get_dummies(_input1, columns=['cabin_side'], prefix='cabin_side')
_input1.columns
_input1.isnull().sum()
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(_input1.corr()))
heatmap = sns.heatmap(_input1.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation', fontdict={'fontsize': 14}, pad=14)
X = _input1.loc[:, _input1.columns != 'Transported']
y = _input1['Transported']
KnnImpute = KNNImputer()