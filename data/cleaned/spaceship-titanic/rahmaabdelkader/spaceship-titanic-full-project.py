import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train
train.shape
train.describe()
train.info()
for i in train.columns:
    print(i, train[i].isna().sum())
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = train[categorical_columns].mode().iloc[0]
train[categorical_columns] = train[categorical_columns].fillna(mode)
numerical_columns = ['Age', 'FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
median = train[numerical_columns].median()
train[numerical_columns] = train[numerical_columns].fillna(median)
for i in train.columns:
    print(i, train[i].isna().sum())
train = train.drop(columns=['Name'])
train
train.duplicated().sum()
corr = train.corr()
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
train.duplicated().sum()
plt.figure(figsize=(6, 5))
train['HomePlanet'].value_counts().plot.bar(rot=0)
train['CryoSleep'].value_counts().plot(kind='pie', autopct='%1.1f%%')
print(train['CryoSleep'].value_counts())
train['Transported'].value_counts().plot(kind='pie', autopct='%1.1f%%')
print(train['Transported'].value_counts())
bins = [0, 18, 39, 100]
labels = ['Teen', 'Adult', 'Senior']
train['Age_Groups'] = pd.cut(train['Age'], bins=bins, labels=labels, right=False)
plt.figure(figsize=(6, 5))
train['Age_Groups'].value_counts().plot.bar(rot=0)
string = train['Cabin'].str.split('/')
train['Deck'] = string.map(lambda string: string[0])
train['Number'] = string.map(lambda string: string[1])
train['Side'] = string.map(lambda string: string[2])
train = train.drop(columns=['Number', 'Cabin'])
train
plt.figure(figsize=(6, 5))
train['Deck'].value_counts().plot.bar(rot=0)
plt.figure(figsize=(6, 5))
train['Side'].value_counts().plot.bar(rot=0)
string2 = train['PassengerId'].str.split('_')
train['Group'] = string2.map(lambda string: string[0])
train['Psngr_Num'] = string2.map(lambda string: string[1])
le = LabelEncoder()
for i in train.columns:
    if train[i].dtype == 'object':
        label_encoder = LabelEncoder()
        train[i] = label_encoder.fit_transform(train[i])
train['CryoSleep'] = label_encoder.fit_transform(train['CryoSleep'])
train['VIP'] = label_encoder.fit_transform(train['VIP'])
train['Transported'] = label_encoder.fit_transform(train['Transported'])
train['Age_Groups'] = label_encoder.fit_transform(train['Age_Groups'])
train
train.plot(kind='box', subplots=True, layout=(5, 5), figsize=(20, 15))

sns.scatterplot(x=train['VRDeck'], y=train['Transported'])
plt.title('Distributions of VRDeck delay before removing outliers')
train[train['VRDeck'] >= 13000].shape
train = train[train['VRDeck'] < 13000]
sns.scatterplot(x=train['VRDeck'], y=train['Transported'])
plt.title('Distributions of VRDeck delay after removing outliers')
sns.scatterplot(x=train['RoomService'], y=train['Transported'])
plt.title('Distributions of RoomService  before removing outliers')
train[train['RoomService'] >= 8500].shape
train = train[train['RoomService'] < 8500]
sns.scatterplot(x=train['RoomService'], y=train['Transported'])
plt.title('Distributions of RoomService  after removing outliers')
sns.scatterplot(x=train['FoodCourt'], y=train['Transported'])
plt.title('Distributions of FoodCourt before removing outliers')
train[train['FoodCourt'] >= 15000].shape
train = train[train['FoodCourt'] < 15000]
sns.scatterplot(x=train['FoodCourt'], y=train['Transported'])
plt.title('Distributions of FoodCourt after removing outliers')
sns.scatterplot(x=train['Spa'], y=train['Transported'])
plt.title('Distributions of Spa before removing outliers')
train[train['Spa'] >= 17000].shape
train = train[train['Spa'] < 17000]
sns.scatterplot(x=train['Spa'], y=train['Transported'])
plt.title('Distributions of Spa after removing outliers')
train.plot(kind='box', subplots=True, layout=(5, 5), figsize=(20, 15))

x = train.drop('Transported', axis=1)
y = train['Transported']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape
y_train.shape
x_test.shape
lr = LogisticRegression()