import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_data.sample(5)
train_data.shape
train_data.isnull().sum()
train_data.describe()
train_data.describe(include=['O'])
X = train_data.drop(['Transported'], axis=1)
y = train_data.Transported
X['GroupId'] = X['PassengerId'].map(lambda x: int(x.split('_')[0]))
X['NumberInGroup'] = X['PassengerId'].map(lambda x: int(x.split('_')[1]))
X = X.drop(['PassengerId'], axis=1)
X.sample(5)
X['CabinDeck'] = X['Cabin'].map(lambda x: x.split('/')[0] if not pd.isna(x) else np.nan)
X['CabinNum'] = X['Cabin'].map(lambda x: int(x.split('/')[1]) if not pd.isna(x) else np.nan)
X['CabinSide'] = X['Cabin'].map(lambda x: x.split('/')[2] if not pd.isna(x) else np.nan)
X = X.drop(['Cabin'], axis=1)
X.sample(5)
name_lengths = X.Name.map(lambda x: len(list(x.split())) if not pd.isna(x) else -1)
name_lengths.value_counts()
X['FirstName'] = X.Name.map(lambda x: x.split()[0] if not pd.isna(x) else np.nan)
X['LastName'] = X.Name.map(lambda x: x.split()[1] if not pd.isna(x) else np.nan)
X = X.drop(['Name'], axis=1)
X.sample(5)
X.FirstName.value_counts()
import random
X = X.fillna({'HomePlanet': 'Earth', 'CryoSleep': False, 'CabinDeck': 'F', 'CabinNum': 82, 'CabinSide': 'S', 'Destination': 'TRAPPIST-1e', 'Age': X.Age.median(), 'VIP': False, 'RoomService': X.RoomService.median(), 'FoodCourt': X.FoodCourt.median(), 'ShoppingMall': X.ShoppingMall.median(), 'Spa': X.Spa.median(), 'VRDeck': X.VRDeck.median()})
X.FirstName.fillna(random.choice(X.FirstName[X.FirstName.notna()]), inplace=True)
X.LastName.fillna(random.choice(X.LastName[X.LastName.notna()]), inplace=True)
X.CryoSleep = X.CryoSleep.astype(int)
X.VIP = X.VIP.astype(int)
y = y.astype(int)
X.isnull().sum()
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=42)
from catboost import CatBoostClassifier
cat_features_index = np.where(X.dtypes != float)[0]
model = CatBoostClassifier(eval_metric='Accuracy', use_best_model=True, random_seed=42, iterations=100)