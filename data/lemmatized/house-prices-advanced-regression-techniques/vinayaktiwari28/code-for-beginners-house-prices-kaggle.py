import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.head()
_input1['Year_old'] = 2021 - _input1['YrSold']
_input1
_input1 = _input1.drop(['YrSold'], axis=1, inplace=False)
_input1.head()
_input1.isnull().sum()
final_dataset = pd.get_dummies(_input1, drop_first=True)
final_dataset = final_dataset.dropna()
final_dataset['Id'] = pd.to_numeric(final_dataset['Id'])
final_dataset = final_dataset.set_index('Id', inplace=False)
y = final_dataset['SalePrice']
X = final_dataset.drop(['SalePrice'], axis=1)
X
from sklearn.ensemble import ExtraTreesRegressor
ex = ExtraTreesRegressor()