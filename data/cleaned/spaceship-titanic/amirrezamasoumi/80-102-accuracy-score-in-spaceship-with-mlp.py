import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
data_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
data_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
data_train.head()
data_train.columns
data_test.columns
data_train.info()
data_train.isnull().sum()
data_test.isnull().sum()
data_train.describe()
data_train['HomePlanet'].value_counts()
data_train['CryoSleep'].value_counts()
Cabin_split = data_train['Cabin'].str.split('/', expand=True)
Cabin_split
print(len(Cabin_split[0].value_counts()))
print(len(Cabin_split[1].value_counts()))
print(len(Cabin_split[2].value_counts()))
print(Cabin_split[0].value_counts())
print(Cabin_split[2].value_counts())
data_train['cabin_deck'] = Cabin_split[0]
data_train['cabin_side'] = Cabin_split[2]
data_train.head()
data_train['Destination'].value_counts()
sns.histplot(data_train['Age'])
data_train['VIP'].value_counts()
sns.scatterplot(x='RoomService', data=data_train, y='Age')
sns.scatterplot(x='RoomService', data=data_train, y='FoodCourt')
sns.scatterplot(x='ShoppingMall', data=data_train, y='FoodCourt')
sns.scatterplot(x='ShoppingMall', data=data_train, y='Spa')
sns.scatterplot(x='VRDeck', data=data_train, y='Spa')
data_train['Transported'].value_counts()
name_split = data_train['Name'].str.split(' ', expand=True)
name_split
(name_split[1].value_counts() > 2).sum()
name_split[1].value_counts() > 2
data_train.drop(columns=['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
data_train.head()
sns.boxplot(data=data_train, x='RoomService')
sns.histplot(data=data_train, x='RoomService', bins=5)
data_train = pd.get_dummies(data_train, columns=['HomePlanet'], prefix='HomePlanet')
data_train = pd.get_dummies(data_train, columns=['CryoSleep'], prefix='CryoSleep', drop_first=True)
data_train = pd.get_dummies(data_train, columns=['Destination'], prefix='Destination')
data_train = pd.get_dummies(data_train, columns=['VIP'], prefix='VIP', drop_first=True)
data_train = pd.get_dummies(data_train, columns=['cabin_deck'], prefix='cabin_deck')
data_train = pd.get_dummies(data_train, columns=['cabin_side'], prefix='cabin_side')
data_train.columns
data_train.isnull().sum()
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(data_train.corr()))
heatmap = sns.heatmap(data_train.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation', fontdict={'fontsize': 14}, pad=14)
X = data_train.loc[:, data_train.columns != 'Transported']
y = data_train['Transported']
KnnImpute = KNNImputer()