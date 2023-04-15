import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_filename = 'data/input/spaceship-titanic/train.csv'
test_filename = 'data/input/spaceship-titanic/test.csv'

import numpy as np, pandas as pd, tqdm
import math, sys, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torchvision
from torch.optim import lr_scheduler, Adam, SGD, RMSprop
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
df_train = pd.read_csv(train_filename)
df_test = pd.read_csv(test_filename)
df_test_original = df_test.copy()
df_train

def print_unique(df):
    for col in df:
        unique = df[col].unique()
        max_len = 6
        if len(unique) <= max_len:
            unique = [f'{u}({(df[col] == u).sum()})' for u in unique]
        if len(unique) > max_len:
            unique = list(unique[:max_len]) + [f'...({len(unique) - max_len})']
        print(col, unique)
print_unique(df_train)
df_train_Y = df_train[['Transported']]
good_columns = ['CryoSleep', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df_train_X_good_cols = df_train[good_columns]
df_test_X_good_cols = df_test[good_columns]
print_unique(df_train_X_good_cols)
df_Y = df_train_Y.copy()
df = df_train_X_good_cols.copy()
df_test = df_test_X_good_cols.copy()
df[['CryoSleep']] = df[['CryoSleep']].fillna(value=df[['CryoSleep']].mode().iloc[0, 0])
df_test[['CryoSleep']] = df_test[['CryoSleep']].fillna(value=df_test[['CryoSleep']].mode().iloc[0, 0])
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[num_cols] = df[num_cols].fillna(value=df[num_cols].median())
df_test[num_cols] = df_test[num_cols].fillna(value=df_test[num_cols].median())
df['CryoSleep'] = df['CryoSleep'].astype(int)
df_test['CryoSleep'] = df_test['CryoSleep'].astype(int)
df_Y['Transported'] = df_Y['Transported'].astype(int)
print('\ndf')
print_unique(df)
print('\ndf_Y')
print_unique(df_Y)
print('\ndf_test')
print_unique(df_test)
df.info()
df_Y.info()
df_train.describe(include='all')
df_train.info(verbose=True)
(X_train, X_test, y_train, y_test) = train_test_split(df.to_numpy(), df_Y.to_numpy(), test_size=0.01)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000)