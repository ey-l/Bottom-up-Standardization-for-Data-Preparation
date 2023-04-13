import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=False)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=False)
print(_input1.shape)
_input0.shape
from catboost import CatBoostRegressor
_input1 = _input1.fillna(0)
_input0 = _input0.fillna(0)
cols = _input1.select_dtypes(include=['object']).columns
model = CatBoostRegressor()
y = _input1.iloc[:, -1]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(_input1.iloc[:, :-1], y)
print(x_train.shape)
y_train.shape