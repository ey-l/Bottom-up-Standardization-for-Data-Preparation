import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
dty = df.dtypes
dty
missing = df.isnull().sum()
missing
total_cells = np.product(df.shape)
total_missing = df.isnull().sum()
percents_missng = total_missing / total_cells * 100
total_perc_missing = missing.sum() / total_cells * 100
total_perc_missing
is_null_df = pd.DataFrame(df.columns)
is_null_df.insert(1, 'total_missing', missing.values)
is_null_df.insert(2, 'percents_missng', percents_missng.values)
is_null_df.insert(3, 'dtype', dty.values)
is_null_df
df.columns
ints = df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
ints
df[df['Age'].isnull()]
df['Age'].fillna(df.Age.mean(), inplace=True)
df.isnull().sum()
df.loc[df.index[64]]
df['FoodCourt'].fillna(0, inplace=True)
df.isnull().sum()
df.loc[df.index[95]]
df.dropna(subset=['RoomService'], inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()