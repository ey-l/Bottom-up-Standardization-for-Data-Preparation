import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.info()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(_input1.corr(), center=0)
plt.title('Correlations Between Columns')
y = _input1.SalePrice
X = _input1.drop(columns=['SalePrice'], axis=1)
(y.shape, X.shape, _input0.shape)
corr_matrix = _input1.corr()
corr_matrix['SalePrice'][(corr_matrix['SalePrice'] > 0.4) | (corr_matrix['SalePrice'] < -0.4)]
important_num_cols = list(corr_matrix['SalePrice'][(corr_matrix['SalePrice'] > 0.5) | (corr_matrix['SalePrice'] < -0.5)].index)
important_num_cols.remove('SalePrice')
len(important_num_cols)
important_num_cols
X_num_only = X[important_num_cols]
X_num_only.shape
plt.figure(figsize=(10, 8))
sns.heatmap(X_num_only.corr(), center=0)
plt.title('Correlations Between Columns')
corr_X = X_num_only.corr()
len(corr_X)
for i in range(0, len(corr_X) - 1):
    for j in range(i + 1, len(corr_X)):
        if corr_X.iloc[i, j] < -0.6 or corr_X.iloc[i, j] > 0.6:
            print(corr_X.iloc[i, j], i, j, corr_X.index[i], corr_X.index[j])
num_cols = [i for i in X_num_only.columns if i not in ['1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'GarageArea']]
cat_cols = ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']
X_final = X[num_cols]
X_final.shape
X_final['YearRemodAdd'] = X_final['YearRemodAdd'] - X_final['YearBuilt']
X_final.head()
X_final.isna().sum()
X[cat_cols].isna().sum()
X_categorical_df = pd.get_dummies(X[cat_cols], columns=cat_cols)
X_categorical_df
X_final = X_final.join(X_categorical_df)
X_final
from sklearn import preprocessing