import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()
df_train.head()
df_train['Year_old'] = 2021 - df_train['YrSold']
df_train
df_train.drop(['YrSold'], axis=1, inplace=True)
df_train.head()
df_train.isnull().sum()
final_dataset = pd.get_dummies(df_train, drop_first=True)
final_dataset = final_dataset.dropna()
final_dataset['Id'] = pd.to_numeric(final_dataset['Id'])
final_dataset.set_index('Id', inplace=True)
y = final_dataset['SalePrice']
X = final_dataset.drop(['SalePrice'], axis=1)
X
from sklearn.ensemble import ExtraTreesRegressor
ex = ExtraTreesRegressor()