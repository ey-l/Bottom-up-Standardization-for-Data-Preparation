import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
_input1.shape
_input1.describe()
_input1.info()
for i in _input1.columns:
    print(i, _input1[i].isna().sum())
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = _input1[categorical_columns].mode().iloc[0]
_input1[categorical_columns] = _input1[categorical_columns].fillna(mode)
numerical_columns = ['Age', 'FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
median = _input1[numerical_columns].median()
_input1[numerical_columns] = _input1[numerical_columns].fillna(median)
for i in _input1.columns:
    print(i, _input1[i].isna().sum())
_input1 = _input1.drop(columns=['Name'])
_input1
_input1.duplicated().sum()
corr = _input1.corr()
(fig, ax) = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, ax=ax)
correlated_features = set()
correlation_matrix = corr
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.6:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
len(correlated_features)
_input1.duplicated().sum()
plt.figure(figsize=(6, 5))
_input1['HomePlanet'].value_counts().plot.bar(rot=0)
_input1['CryoSleep'].value_counts().plot(kind='pie', autopct='%1.1f%%')
print(_input1['CryoSleep'].value_counts())
_input1['Transported'].value_counts().plot(kind='pie', autopct='%1.1f%%')
print(_input1['Transported'].value_counts())
bins = [0, 18, 39, 100]
labels = ['Teen', 'Adult', 'Senior']
_input1['Age_Groups'] = pd.cut(_input1['Age'], bins=bins, labels=labels, right=False)
plt.figure(figsize=(6, 5))
_input1['Age_Groups'].value_counts().plot.bar(rot=0)
string = _input1['Cabin'].str.split('/')
_input1['Deck'] = string.map(lambda string: string[0])
_input1['Number'] = string.map(lambda string: string[1])
_input1['Side'] = string.map(lambda string: string[2])
_input1 = _input1.drop(columns=['Number', 'Cabin'])
_input1
plt.figure(figsize=(6, 5))
_input1['Deck'].value_counts().plot.bar(rot=0)
plt.figure(figsize=(6, 5))
_input1['Side'].value_counts().plot.bar(rot=0)
string2 = _input1['PassengerId'].str.split('_')
_input1['Group'] = string2.map(lambda string: string[0])
_input1['Psngr_Num'] = string2.map(lambda string: string[1])
le = LabelEncoder()
for i in _input1.columns:
    if _input1[i].dtype == 'object':
        label_encoder = LabelEncoder()
        _input1[i] = label_encoder.fit_transform(_input1[i])
_input1['CryoSleep'] = label_encoder.fit_transform(_input1['CryoSleep'])
_input1['VIP'] = label_encoder.fit_transform(_input1['VIP'])
_input1['Transported'] = label_encoder.fit_transform(_input1['Transported'])
_input1['Age_Groups'] = label_encoder.fit_transform(_input1['Age_Groups'])
_input1
_input1.plot(kind='box', subplots=True, layout=(5, 5), figsize=(20, 15))
sns.scatterplot(x=_input1['VRDeck'], y=_input1['Transported'])
plt.title('Distributions of VRDeck delay before removing outliers')
_input1[_input1['VRDeck'] >= 13000].shape
_input1 = _input1[_input1['VRDeck'] < 13000]
sns.scatterplot(x=_input1['VRDeck'], y=_input1['Transported'])
plt.title('Distributions of VRDeck delay after removing outliers')
sns.scatterplot(x=_input1['RoomService'], y=_input1['Transported'])
plt.title('Distributions of RoomService  before removing outliers')
_input1[_input1['RoomService'] >= 8500].shape
_input1 = _input1[_input1['RoomService'] < 8500]
sns.scatterplot(x=_input1['RoomService'], y=_input1['Transported'])
plt.title('Distributions of RoomService  after removing outliers')
sns.scatterplot(x=_input1['FoodCourt'], y=_input1['Transported'])
plt.title('Distributions of FoodCourt before removing outliers')
_input1[_input1['FoodCourt'] >= 15000].shape
_input1 = _input1[_input1['FoodCourt'] < 15000]
sns.scatterplot(x=_input1['FoodCourt'], y=_input1['Transported'])
plt.title('Distributions of FoodCourt after removing outliers')
sns.scatterplot(x=_input1['Spa'], y=_input1['Transported'])
plt.title('Distributions of Spa before removing outliers')
_input1[_input1['Spa'] >= 17000].shape
_input1 = _input1[_input1['Spa'] < 17000]
sns.scatterplot(x=_input1['Spa'], y=_input1['Transported'])
plt.title('Distributions of Spa after removing outliers')
_input1.plot(kind='box', subplots=True, layout=(5, 5), figsize=(20, 15))
x = _input1.drop('Transported', axis=1)
y = _input1['Transported']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape
y_train.shape
x_test.shape
lr = LogisticRegression()