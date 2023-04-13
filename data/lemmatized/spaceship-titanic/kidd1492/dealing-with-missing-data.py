import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
dty = _input1.dtypes
dty
missing = _input1.isnull().sum()
missing
total_cells = np.product(_input1.shape)
total_missing = _input1.isnull().sum()
percents_missng = total_missing / total_cells * 100
total_perc_missing = missing.sum() / total_cells * 100
total_perc_missing
is_null_df = pd.DataFrame(_input1.columns)
is_null_df.insert(1, 'total_missing', missing.values)
is_null_df.insert(2, 'percents_missng', percents_missng.values)
is_null_df.insert(3, 'dtype', dty.values)
is_null_df
_input1.columns
ints = _input1[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
ints
_input1[_input1['Age'].isnull()]
_input1['Age'] = _input1['Age'].fillna(_input1.Age.mean(), inplace=False)
_input1.isnull().sum()
_input1.loc[_input1.index[64]]
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0, inplace=False)
_input1.isnull().sum()
_input1.loc[_input1.index[95]]
_input1 = _input1.dropna(subset=['RoomService'], inplace=False)
_input1.isnull().sum()
_input1 = _input1.dropna(inplace=False)
_input1.isnull().sum()