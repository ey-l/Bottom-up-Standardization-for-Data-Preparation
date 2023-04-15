import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
train_data.describe()
train_data.isnull().sum()
(train_data.columns.drop('SalePrice') == test_data.columns).any()
train_data.drop(['Id'], axis=1, inplace=True)
id_test_list = test_data['Id'].tolist()
test_data.drop(['Id'], axis=1, inplace=True)
num_cols = []
cat_cols = []
for col in train_data.columns:
    if train_data[col].dtype in ('int64', 'float64'):
        num_cols.append(train_data[col].name)
    else:
        cat_cols.append(train_data[col].name)
num_trainData = train_data[num_cols]
cat_trainData = train_data[cat_cols]
num_testData = test_data[num_cols[0:-1]]
cat_testData = test_data[cat_cols]
num_trainData.columns
cat_trainData.columns
from sklearn.feature_selection import VarianceThreshold
thresholder = VarianceThreshold(threshold=0.15)