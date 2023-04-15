import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
all_data = pd.concat([train_data, test_data], axis='rows')
all_data.shape[0] == train_data.shape[0] + test_data.shape[0]
all_data.set_index('Id', inplace=True)
all_data.info()
pd.get_dummies(data=all_data)
dummy_data = pd.get_dummies(all_data, columns=all_data.select_dtypes(include='object').columns)
dummy_data.shape
train_count = train_data.shape[0]
train_data.tail()
model = CatBoostRegressor()
train_data = dummy_data.iloc[:train_count]
test_data = dummy_data.iloc[train_count:].drop(['SalePrice'], axis='columns')
train_data.tail()
test_data.head()
X_train = train_data.drop(['SalePrice'], axis='columns')
y_train = train_data.SalePrice