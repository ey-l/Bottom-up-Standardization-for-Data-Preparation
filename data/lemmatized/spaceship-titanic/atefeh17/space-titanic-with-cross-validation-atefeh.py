import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import missingno
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
data_df = pd.concat([_input1, _input0], axis=0)
data_df['Cabin'] = data_df['Cabin'].fillna('F/0/S', inplace=False)
data_df['Cabin'] = data_df.Cabin.apply(str)

def deck(cabin):
    deck = cabin[0]
    return deck

def side(cabin):
    side = cabin[-1]
    return side
data_df['Deck'] = data_df.Cabin.apply(deck)
data_df['Side'] = data_df.Cabin.apply(side)
data_df['HomePlanet'] = data_df['HomePlanet'].fillna('Earth', inplace=False)
data_df['CryoSleep'] = data_df['CryoSleep'].fillna(False, inplace=False)
data_df['Destination'] = data_df['Destination'].fillna('TRAPPIST-1e', inplace=False)
data_df['Age'] = data_df['Age'].fillna(data_df['Age'].mean(), inplace=False)
data_df['VIP'] = data_df['VIP'].fillna(data_df['VIP'].mean(), inplace=False)
data_df['RoomService'] = data_df['RoomService'].fillna(data_df['RoomService'].mean(), inplace=False)
data_df['FoodCourt'] = data_df['FoodCourt'].fillna(data_df['FoodCourt'].mean(), inplace=False)
data_df['ShoppingMall'] = data_df['ShoppingMall'].fillna(data_df['ShoppingMall'].mean(), inplace=False)
data_df['Spa'] = data_df['Spa'].fillna(data_df['Spa'].mean(), inplace=False)
data_df['VRDeck'] = data_df['VRDeck'].fillna(data_df['VRDeck'].mean(), inplace=False)
data_df['Name'] = data_df['Name'].fillna('Unknown', inplace=False)
a = data_df['Transported']
(labels, uniques) = pd.factorize(a)
labels
Transported = pd.DataFrame(labels)
data_oh = pd.get_dummies(data_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side'])
data_oh.drop('Transported', axis=1)
data_oh['Transported'] = Transported
data_oh = data_oh.drop(['Cabin', 'Name'], axis=1, inplace=False)
from sklearn.preprocessing import StandardScaler
num_cols = ['Age', 'Spa', 'ShoppingMall', 'VRDeck', 'RoomService', 'FoodCourt']
data_sd = data_oh.copy()
scaler = StandardScaler()
data_sd[num_cols] = scaler.fit_transform(data_sd[num_cols])
train = data_sd[0:len(_input1)]
test = data_sd[len(_input1):]
y = train['Transported']
x = train.drop('Transported', axis=1)
x_test = test.drop('Transported', axis=1)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as skle
kfold = KFold(n_splits=5, shuffle=True, random_state=1026)
scores = []
for (tr_idx, val_idx) in kfold.split(x):
    (x_tr, x_val) = (x.iloc[tr_idx], x.iloc[val_idx])
    (y_tr, y_val) = (y.iloc[tr_idx], y.iloc[val_idx])
    model = skle.RandomForestClassifier(100)