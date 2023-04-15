import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_train.info()
df_train.describe().round(2)
(df_train.columns.drop('SalePrice') == df_test.columns).any()
df_train.drop(['Id'], axis=1, inplace=True)
id_test_list = df_test['Id'].tolist()
df_test.drop(['Id'], axis=1, inplace=True)
numerical_cols = []
categorical_cols = []
for col in df_train.columns:
    if df_train[col].dtype in ('int64', 'float64'):
        numerical_cols.append(df_train[col].name)
    else:
        categorical_cols.append(df_train[col].name)
numerical_df_train = df_train[numerical_cols]
categorical_df_train = df_train[categorical_cols]
numerical_df_test = df_test[numerical_cols[0:-1]]
categorical_df_test = df_test[categorical_cols]
numerical_df_train.columns
categorical_df_test.columns
numerical_df_train.hist(figsize=(15, 20), bins=30, color='blue', edgecolor='black')
from sklearn.feature_selection import VarianceThreshold
thresholder = VarianceThreshold(threshold=0.15)