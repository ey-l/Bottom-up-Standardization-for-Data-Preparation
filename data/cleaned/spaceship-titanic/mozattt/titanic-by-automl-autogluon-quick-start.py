

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
train = TabularDataset('data/input/spaceship-titanic/train.csv')
test = TabularDataset('data/input/spaceship-titanic/test.csv')
(id, label) = ('PassengerId', 'Transported')
eval_metric = 'accuracy'
df = pd.concat([train, test], axis=0)
num_columns = []
for col in train:
    if train[col].dtypes != 'object' and col != label:
        num_columns.append(col)
STRATEGY = 'median'
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy=STRATEGY)
df[imputer_cols] = imputer.fit_transform(df[imputer_cols])
df['HomePlanet'].fillna('Z', inplace=True)
label_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for col in label_cols:
    df[col] = df[col].astype(str)
    df[col] = LabelEncoder().fit_transform(df[col])
ss = StandardScaler()
df[num_columns] = ss.fit_transform(df[num_columns])
train = df[:len(train)]
test = df[len(train):]