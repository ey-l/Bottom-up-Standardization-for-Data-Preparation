import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=False)
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=False)
print(df_train.shape)
df_test.shape
from catboost import CatBoostRegressor
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
cols = df_train.select_dtypes(include=['object']).columns
model = CatBoostRegressor()
y = df_train.iloc[:, -1]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(df_train.iloc[:, :-1], y)
print(x_train.shape)
y_train.shape