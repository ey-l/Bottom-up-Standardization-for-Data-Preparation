import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.describe()
_input1.isnull().sum()
(_input1.columns.drop('SalePrice') == _input0.columns).any()
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
id_test_list = _input0['Id'].tolist()
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
num_cols = []
cat_cols = []
for col in _input1.columns:
    if _input1[col].dtype in ('int64', 'float64'):
        num_cols.append(_input1[col].name)
    else:
        cat_cols.append(_input1[col].name)
num_trainData = _input1[num_cols]
cat_trainData = _input1[cat_cols]
num_testData = _input0[num_cols[0:-1]]
cat_testData = _input0[cat_cols]
num_trainData.columns
cat_trainData.columns
from sklearn.feature_selection import VarianceThreshold
thresholder = VarianceThreshold(threshold=0.15)