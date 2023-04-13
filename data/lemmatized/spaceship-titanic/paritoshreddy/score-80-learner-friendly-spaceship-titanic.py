import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
complete_data = [_input1, _input0]
print(_input1.head())
for df in complete_data:
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = df['Num'].str.strip()
    df['Num'] = df['Num'].astype('float')
print('Train Dataset Missing Count:')
print(_input1.isna().sum(), '\n')
print('Test Dataset Missing Count:')
print(_input0.isna().sum())
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
    df['Num'] = df['Num'].fillna(_input1.Num.median())
print('Train Dataset Missing Count:')
print(_input1.isna().sum(), '\n')
print('Test Dataset Missing Count:')
print(_input0.isna().sum())
features = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Deck', 'Side', 'Num', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X = pd.get_dummies(_input1[features])
y = _input1['Transported']
final_x_test = pd.get_dummies(_input0[features])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=44)
model_RFC = RandomForestClassifier(n_estimators=90, max_depth=13, max_features=3, min_samples_split=17, random_state=9)