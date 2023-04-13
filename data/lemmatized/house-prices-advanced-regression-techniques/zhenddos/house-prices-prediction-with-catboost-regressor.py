import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
all_data = pd.concat([_input1, _input0], axis='rows')
all_data.shape[0] == _input1.shape[0] + _input0.shape[0]
all_data = all_data.set_index('Id', inplace=False)
all_data.info()
pd.get_dummies(data=all_data)
dummy_data = pd.get_dummies(all_data, columns=all_data.select_dtypes(include='object').columns)
dummy_data.shape
train_count = _input1.shape[0]
_input1.tail()
model = CatBoostRegressor()
_input1 = dummy_data.iloc[:train_count]
_input0 = dummy_data.iloc[train_count:].drop(['SalePrice'], axis='columns')
_input1.tail()
_input0.head()
X_train = _input1.drop(['SalePrice'], axis='columns')
y_train = _input1.SalePrice