import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)
print(test.shape)
num_cols = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
num_cols.remove('Id')
num_cols.remove('SalePrice')
num_analysis = train[num_cols].copy()
for col in num_cols:
    if num_analysis[col].isnull().sum() > 0:
        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1, 1))
clf = ExtraTreesRegressor(random_state=42)