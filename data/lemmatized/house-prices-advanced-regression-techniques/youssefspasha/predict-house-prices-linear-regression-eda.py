import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.info()
_input1.describe().round(2)
(_input1.columns.drop('SalePrice') == _input0.columns).any()
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
id_test_list = _input0['Id'].tolist()
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
numerical_cols = []
categorical_cols = []
for col in _input1.columns:
    if _input1[col].dtype in ('int64', 'float64'):
        numerical_cols.append(_input1[col].name)
    else:
        categorical_cols.append(_input1[col].name)
numerical_df_train = _input1[numerical_cols]
categorical_df_train = _input1[categorical_cols]
numerical_df_test = _input0[numerical_cols[0:-1]]
categorical_df_test = _input0[categorical_cols]
numerical_df_train.columns
categorical_df_test.columns
numerical_df_train.hist(figsize=(15, 20), bins=30, color='blue', edgecolor='black')
from sklearn.feature_selection import VarianceThreshold
thresholder = VarianceThreshold(threshold=0.15)