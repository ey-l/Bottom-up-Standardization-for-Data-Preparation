import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
dataset_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset_train.head()
dataset_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
dataset_test.head()
dataset_train.isnull().sum()
dataset_test.isnull().sum()
dataset_train.shape
dataset_test.shape
df_train = dataset_train.fillna(method='ffill')
df_test = dataset_test.fillna(method='ffill')
df_train.isnull().sum()
df_test.isnull().sum()
df_train.describe()
df_test.describe()
df_train.info()
df_train.info()
label_encoder = preprocessing.LabelEncoder()
df_train['CryoSleep_type'] = label_encoder.fit_transform(df_train['CryoSleep'])
df_train['VIP_type'] = label_encoder.fit_transform(df_train['VIP'])
df_train['Transported_type'] = label_encoder.fit_transform(df_train['Transported'])
df_train['HomePlanet_type'] = label_encoder.fit_transform(df_train['HomePlanet'])
df_train['Destination_type'] = label_encoder.fit_transform(df_train['Destination'])
df_train['Spa_type'] = label_encoder.fit_transform(df_train['Spa'])
df_train['FoodCourt_type'] = label_encoder.fit_transform(df_train['FoodCourt'])
df_train['RoomService_type'] = label_encoder.fit_transform(df_train['RoomService'])
label_encoder = preprocessing.LabelEncoder()
df_test['CryoSleep_type'] = label_encoder.fit_transform(df_test['CryoSleep'])
df_test['VIP_type'] = label_encoder.fit_transform(df_test['VIP'])
df_test['HomePlanet_type'] = label_encoder.fit_transform(df_test['HomePlanet'])
df_test['Destination_type'] = label_encoder.fit_transform(df_test['Destination'])
df_test['Spa_type'] = label_encoder.fit_transform(df_test['Spa'])
df_test['FoodCourt_type'] = label_encoder.fit_transform(df_test['FoodCourt'])
df_test['RoomService_type'] = label_encoder.fit_transform(df_test['RoomService'])
del df_train['CryoSleep']
del df_train['VIP']
del df_train['Transported']
del df_train['Name']
del df_train['HomePlanet']
del df_train['Cabin']
del df_train['Destination']
del df_train['PassengerId']
del df_train['Spa']
del df_train['FoodCourt']
del df_train['RoomService']
del df_test['CryoSleep']
del df_test['VIP']
del df_test['Name']
del df_test['HomePlanet']
del df_test['Cabin']
del df_test['Destination']
del df_test['PassengerId']
del df_test['Spa']
del df_test['FoodCourt']
del df_test['RoomService']
df_train.head()
df_test.head()
df_train.info()
df_test.info()
df_train.columns
x = df_train[['Age', 'ShoppingMall', 'VRDeck', 'CryoSleep_type', 'VIP_type', 'HomePlanet_type', 'Destination_type', 'Spa_type', 'FoodCourt_type', 'RoomService_type']]
y = df_train[['Transported_type']]
x
y
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df_train)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print('Mean value:', x_train_scaled.mean(axis=0))
print('SD value:', x_train_scaled.std(axis=0))
df_scaled = pd.DataFrame(scaled, columns=df_train.columns)
df_scaled.head()
from sklearn import tree
clf = tree.DecisionTreeClassifier()