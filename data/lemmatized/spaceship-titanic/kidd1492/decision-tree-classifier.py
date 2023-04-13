import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.columns
cabin = _input1['Cabin'].astype(str)
cabin = [item.split('/')[0] for item in cabin]
_input1.insert(1, 'Deck', cabin)
_input1.reindex()
_input1['Deck'] = _input1['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
_input1 = _input1.dropna(inplace=False)
_input1 = _input1[['PassengerId', 'CryoSleep', 'Deck', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']].copy()
_input1.head(1)
_input1 = pd.get_dummies(_input1, columns=['Deck'], drop_first=False)
_input1
(train, test) = train_test_split(_input1, test_size=0.1, random_state=6, shuffle=True)
features = [c for c in _input1.columns if c != 'Transported']
model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=2, random_state=2, verbose=False)