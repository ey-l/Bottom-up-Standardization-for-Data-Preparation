import numpy as np
import pandas as pd
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
complete_data = [train_data, test_data]
print(train_data.head())
for df in complete_data:
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = df['Num'].str.strip()
    df['Num'] = df['Num'].astype('float')
print('Train Dataset Missing Count:')
print(train_data.isna().sum(), '\n')
print('Test Dataset Missing Count:')
print(test_data.isna().sum())
import random
for df in complete_data:
    df['VIP'] = df['VIP'].fillna(False)
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    df['RoomService'] = df['RoomService'].fillna(0)
    df['FoodCourt'] = df['FoodCourt'].fillna(0)
    df['ShoppingMall'] = df['ShoppingMall'].fillna(0)
    df['Spa'] = df['Spa'].fillna(0)
    df['VRDeck'] = df['VRDeck'].fillna(0)
    df['HomePlanet'] = df['HomePlanet'].fillna(random.choice(df['HomePlanet'][df['HomePlanet'].notna()]))
    df['Deck'] = df['Deck'].fillna(random.choice(df['Deck'][df['Deck'].notna()]))
    df['Side'] = df['Side'].fillna(random.choice(df['Side'][df['Side'].notna()]))
    df['Destination'] = df['Destination'].fillna(random.choice(df['Destination'][df['Destination'].notna()]))
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Num'] = df['Num'].fillna(train_data.Num.median())
print('Train Dataset Missing Count:')
print(train_data.isna().sum(), '\n')
print('Test Dataset Missing Count:')
print(test_data.isna().sum())
features = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Deck', 'Side', 'Num', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X = pd.get_dummies(train_data[features])
y = train_data['Transported']
final_x_test = pd.get_dummies(test_data[features])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=44)
model_RFC = RandomForestClassifier(n_estimators=90, max_depth=13, max_features=3, min_samples_split=17, random_state=9)