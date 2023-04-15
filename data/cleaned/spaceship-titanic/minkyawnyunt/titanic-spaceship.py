import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
train_path = 'data/input/spaceship-titanic/train.csv'
test_path = 'data/input/spaceship-titanic/test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
train_data
train_data['Group'] = [x.split('_')[0] for x in list(train_data['PassengerId'])]
train_data['Group'] = train_data['Group'].astype(int)
test_data['Group'] = [x.split('_')[0] for x in list(test_data['PassengerId'])]
test_data['Group'] = test_data['Group'].astype(int)
train_data['GroupSize'] = train_data['Group'].map(lambda x: train_data['Group'].value_counts()[x])
train_data = train_data.drop('Group', axis=1)
test_data['GroupSize'] = test_data['Group'].map(lambda x: test_data['Group'].value_counts()[x])
test_data = test_data.drop('Group', axis=1)
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', n=2, expand=True)
train_data = train_data.drop('Cabin', axis=1)
train_data = train_data[['PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Num', 'Side', 'GroupSize', 'Name', 'Transported']]
test_data[['Deck', 'Num', 'Side']] = test_data['Cabin'].str.split('/', n=2, expand=True)
test_data = test_data.drop('Cabin', axis=1)
test_data = test_data[['PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Num', 'Side', 'GroupSize', 'Name']]
train_data[['HomePlanet']] = train_data[['HomePlanet']].fillna('Earth')
train_data[['CryoSleep', 'VIP']] = train_data[['CryoSleep', 'VIP']].fillna(False)
train_data[['Destination']] = train_data[['Destination']].fillna('TRAPPIST-1e')
train_data[['RoomService']] = train_data[['RoomService']].fillna(train_data['RoomService'].mean())
train_data[['FoodCourt']] = train_data[['FoodCourt']].fillna(train_data['FoodCourt'].mean())
train_data[['ShoppingMall']] = train_data[['ShoppingMall']].fillna(train_data['ShoppingMall'].mean())
train_data[['Spa']] = train_data[['Spa']].fillna(train_data['Spa'].mean())
train_data[['VRDeck']] = train_data[['VRDeck']].fillna(train_data['VRDeck'].mean())
train_data[['Deck']] = train_data[['Deck']].fillna('F')
train_data[['Num']] = train_data[['Num']].fillna('82')
train_data[['Side']] = train_data[['Side']].fillna('S')
test_data[['HomePlanet']] = test_data[['HomePlanet']].fillna('Earth')
test_data[['CryoSleep', 'VIP']] = test_data[['CryoSleep', 'VIP']].fillna(False)
test_data[['Destination']] = test_data[['Destination']].fillna('TRAPPIST-1e')
test_data[['RoomService']] = test_data[['RoomService']].fillna(test_data['RoomService'].mean())
test_data[['FoodCourt']] = test_data[['FoodCourt']].fillna(test_data['FoodCourt'].mean())
test_data[['ShoppingMall']] = test_data[['ShoppingMall']].fillna(test_data['ShoppingMall'].mean())
test_data[['Spa']] = test_data[['Spa']].fillna(test_data['Spa'].mean())
test_data[['VRDeck']] = test_data[['VRDeck']].fillna(test_data['VRDeck'].mean())
test_data[['Deck']] = test_data[['Deck']].fillna('F')
test_data[['Num']] = test_data[['Num']].fillna('82')
test_data[['Side']] = test_data[['Side']].fillna('S')
train_data['Side'].value_counts()
(x, y) = (train_data.iloc[:, 1:-2], train_data.iloc[:, [-1]])
x = x.apply(LabelEncoder().fit_transform)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1)
lr = LogisticRegression(max_iter=10000)