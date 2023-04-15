import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.columns
cabin = df['Cabin'].astype(str)
cabin = [item.split('/')[0] for item in cabin]
df.insert(1, 'Deck', cabin)
df.reindex()
df['Deck'] = df['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
df.dropna(inplace=True)
df = df[['PassengerId', 'CryoSleep', 'Deck', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']].copy()
df.head(1)
df = pd.get_dummies(df, columns=['Deck'], drop_first=False)
df
(train, test) = train_test_split(df, test_size=0.1, random_state=6, shuffle=True)
features = [c for c in df.columns if c != 'Transported']
model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=2, random_state=2, verbose=False)