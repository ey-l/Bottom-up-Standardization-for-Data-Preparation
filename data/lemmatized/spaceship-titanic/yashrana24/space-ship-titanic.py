import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.isnull().sum()
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = _input1[categorical_columns].mode().iloc[0]
_input1[categorical_columns] = _input1[categorical_columns].fillna(mode)
numerical_columns = ['Age', 'FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
median = _input1[numerical_columns].median()
_input1[numerical_columns] = _input1[numerical_columns].fillna(median)
_input1.isnull().sum()
_input1['HomePlanet'].unique()
counts = _input1['HomePlanet'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('HomePlanet')
_input1['CryoSleep'].unique()
counts = _input1['CryoSleep'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('HomePlanet')
age_counts = _input1['Age'].value_counts()
plt.bar(age_counts.index, age_counts.values)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
counts = _input1['VIP'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('VIP')
corr_matrix = _input1.corr()
sns.heatmap(corr_matrix, cmap='YlGnBu', annot=False)
plt.title('Correlation Matrix Heatmap')
string = _input1['Cabin'].str.split('/')
_input1['Deck'] = string.map(lambda string: string[0])
_input1['Number'] = string.map(lambda string: string[1])
_input1['Side'] = string.map(lambda string: string[2])
_input1 = _input1.drop(columns=['Number', 'Cabin'])
bins = [0, 18, 39, 100]
labels = ['Teen', 'Adult', 'Senior']
_input1['Age_Groups'] = pd.cut(_input1['Age'], bins=bins, labels=labels, right=False)
_input1.head()
string2 = _input1['PassengerId'].str.split('_')
_input1['Group'] = string2.map(lambda string: string[0])
_input1['Psngr_Num'] = string2.map(lambda string: string[1])
_input1.head()
_input1.dtypes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in _input1.columns:
    if _input1[i].dtype == 'object':
        label_encoder = LabelEncoder()
        _input1[i] = label_encoder.fit_transform(_input1[i])
_input1['CryoSleep'] = label_encoder.fit_transform(_input1['CryoSleep'])
_input1['VIP'] = label_encoder.fit_transform(_input1['VIP'])
_input1['Transported'] = label_encoder.fit_transform(_input1['Transported'])
_input1['Age_Groups'] = label_encoder.fit_transform(_input1['Age_Groups'])
_input1.head()
_input1[_input1['RoomService'] >= 8500].shape
_input1 = _input1[_input1['RoomService'] < 8500]
_input1[_input1['Spa'] >= 17000].shape
_input1 = _input1[_input1['Spa'] < 17000]
_input1[_input1['FoodCourt'] >= 15000].shape
_input1 = _input1[_input1['FoodCourt'] < 15000]
_input1.describe()
y = _input1['Transported']
X = _input1.drop(['Transported', 'Name'], axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()