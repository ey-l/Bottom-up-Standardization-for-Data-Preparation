import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head()
df_train.isnull().sum()
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
mode = df_train[categorical_columns].mode().iloc[0]
df_train[categorical_columns] = df_train[categorical_columns].fillna(mode)
numerical_columns = ['Age', 'FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']
median = df_train[numerical_columns].median()
df_train[numerical_columns] = df_train[numerical_columns].fillna(median)
df_train.isnull().sum()
df_train['HomePlanet'].unique()
counts = df_train['HomePlanet'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('HomePlanet')

df_train['CryoSleep'].unique()
counts = df_train['CryoSleep'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('HomePlanet')

age_counts = df_train['Age'].value_counts()
plt.bar(age_counts.index, age_counts.values)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')

counts = df_train['VIP'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('VIP')

corr_matrix = df_train.corr()
sns.heatmap(corr_matrix, cmap='YlGnBu', annot=False)
plt.title('Correlation Matrix Heatmap')

string = df_train['Cabin'].str.split('/')
df_train['Deck'] = string.map(lambda string: string[0])
df_train['Number'] = string.map(lambda string: string[1])
df_train['Side'] = string.map(lambda string: string[2])
df_train = df_train.drop(columns=['Number', 'Cabin'])
bins = [0, 18, 39, 100]
labels = ['Teen', 'Adult', 'Senior']
df_train['Age_Groups'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)
df_train.head()
string2 = df_train['PassengerId'].str.split('_')
df_train['Group'] = string2.map(lambda string: string[0])
df_train['Psngr_Num'] = string2.map(lambda string: string[1])
df_train.head()
df_train.dtypes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in df_train.columns:
    if df_train[i].dtype == 'object':
        label_encoder = LabelEncoder()
        df_train[i] = label_encoder.fit_transform(df_train[i])
df_train['CryoSleep'] = label_encoder.fit_transform(df_train['CryoSleep'])
df_train['VIP'] = label_encoder.fit_transform(df_train['VIP'])
df_train['Transported'] = label_encoder.fit_transform(df_train['Transported'])
df_train['Age_Groups'] = label_encoder.fit_transform(df_train['Age_Groups'])
df_train.head()
df_train[df_train['RoomService'] >= 8500].shape
df_train = df_train[df_train['RoomService'] < 8500]
df_train[df_train['Spa'] >= 17000].shape
df_train = df_train[df_train['Spa'] < 17000]
df_train[df_train['FoodCourt'] >= 15000].shape
df_train = df_train[df_train['FoodCourt'] < 15000]
df_train.describe()
y = df_train['Transported']
X = df_train.drop(['Transported', 'Name'], axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()