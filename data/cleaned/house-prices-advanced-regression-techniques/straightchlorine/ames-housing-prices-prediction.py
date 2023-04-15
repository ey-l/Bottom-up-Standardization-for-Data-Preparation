import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px

df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
is_outlier = (df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 700000)
(fig, ax) = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_train, x='GrLivArea', y='SalePrice', hue=is_outlier, ax=ax, legend=False)
fig.show()
df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 700000)].index)
(fig, ax) = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df_train, x='GrLivArea', y='SalePrice', ax=ax)
train_copy = df_train.copy().drop('SalePrice', axis=1)
test_copy = df_test.copy()
df_total = pd.concat([train_copy, test_copy])

df_total.info()
df_total['GarageYrBlt'] = df_total['GarageYrBlt'].fillna(0).astype('int64')
df_total['MSSubClass'] = df_total['MSSubClass'].astype('object')
df_total['MoSold'] = df_total['MoSold'].astype('object', copy=False)
df_total['YrSold'] = df_total['YrSold'].astype('object', copy=False)
df_total['MSZoning'] = df_total.groupby(['MSSubClass', 'Neighborhood'])['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
df_total['LotFrontage'] = df_total.groupby('MSZoning')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df_total['MasVnrArea'].where(df_total['MasVnrArea'].notna(), 0, inplace=True)
df_total['MasVnrType'].where(df_total['MasVnrType'].notna(), 'None', inplace=True)
df_total['BsmtQual'].where(df_total['TotalBsmtSF'] != 0, 'NA', inplace=True)
df_total['BsmtCond'].where(df_total['TotalBsmtSF'] != 0, 'NA', inplace=True)
df_total['BsmtExposure'].where(df_total['TotalBsmtSF'] != 0, 'NA', inplace=True)
df_total['BsmtFinType1'].where(df_total['TotalBsmtSF'] != 0, 'NA', inplace=True)
df_total['BsmtFinType2'].where(df_total['TotalBsmtSF'] != 0, 'NA', inplace=True)
df_total['BsmtFullBath'].where(df_total['TotalBsmtSF'] != 0, 0.0, inplace=True)
df_total['BsmtHalfBath'].where(df_total['TotalBsmtSF'] != 0, 0.0, inplace=True)
df_total['BsmtFinSF1'].where(df_total['TotalBsmtSF'] != 0, 0.0, inplace=True)
df_total['BsmtFinSF2'].where(df_total['TotalBsmtSF'] != 0, 0.0, inplace=True)
df_total['BsmtUnfSF'].where(df_total['TotalBsmtSF'] != 0, 0.0, inplace=True)
df_total['GarageCond'].where(df_total['GarageArea'] != 0, 'NA', inplace=True)
df_total['GarageQual'].where(df_total['GarageArea'] != 0, 'NA', inplace=True)
df_total['GarageFinish'].where(df_total['GarageArea'] != 0, 'NA', inplace=True)
df_total['GarageType'].where(df_total['GarageArea'] != 0, 'NA', inplace=True)
df_total['Utilities'] = df_total['Utilities'].fillna(df_total['Utilities'].mode()[0])
df_total['Exterior1st'] = df_total.groupby(['MSSubClass', 'Neighborhood'])['Exterior1st'].transform(lambda x: x.fillna(x.mode()[0]))
df_total['Exterior2nd'] = df_total.groupby(['MSSubClass', 'Neighborhood'])['Exterior2nd'].transform(lambda x: x.fillna(x.mode()[0]))
df_total['Electrical'] = df_total['Electrical'].fillna(df_total['Electrical'].mode()[0])
df_total['KitchenQual'] = df_total['KitchenQual'].fillna(df_total['KitchenQual'].mode()[0])
df_total['Functional'] = df_total['Functional'].fillna(df_total['Functional'].mode()[0])
df_total['FireplaceQu'].where(df_total['Fireplaces'] != 0, 'NA', inplace=True)
df_total['PoolQC'].where(df_total['PoolArea'] != 0, 'NA', inplace=True)
df_total['PoolQC'] = df_total['PoolQC'].fillna(df_total['PoolQC'][df_total['PoolArea'] != 0].mode()[0])
df_total['MiscFeature'].where(df_total['MiscVal'] != 0, 'NA', inplace=True)
df_total['MiscFeature'] = df_total['MiscFeature'].fillna('Othr')
df_total['SaleType'] = df_total['SaleType'].fillna(df_total['SaleType'].mode()[0])
df_total.drop(axis=1, columns=['Alley', 'Fence'], errors='ignore', inplace=True)
df_total.info()
df_total.iloc[2120, [28, 29, 30, 31, 33]] = 'NA'
df_total.iloc[2120, [32, 34, 35, 36, 45, 46]] = 0.0
df_total.loc[2121, ['BsmtFinType1', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0.0
df_total['BsmtQual'] = df_total['BsmtQual'].transform(lambda x: x.fillna(x.mode()[0]))
df_total['BsmtCond'] = df_total['BsmtCond'].transform(lambda x: x.fillna(x.mode()[0]))
df_total['BsmtExposure'] = df_total['BsmtExposure'].transform(lambda x: x.fillna(x.mode()[0]))
df_total['BsmtFinType2'] = df_total['BsmtFinType2'].transform(lambda x: x.fillna('Rec'))
df_total.info()
df_total['GarageFinish'] = df_total['GarageFinish'].fillna(df_total['GarageFinish'][df_total['GarageType'] == 'Detchd'].mode()[0])
df_total['GarageQual'] = df_total['GarageQual'].fillna(df_total['GarageQual'][df_total['GarageType'] == 'Detchd'].mode()[0])
df_total['GarageCond'] = df_total['GarageCond'].fillna(df_total['GarageCond'][df_total['GarageType'] == 'Detchd'].mode()[0])
df_total['GarageCars'] = df_total['GarageCars'].fillna(df_total['GarageCars'][df_total['GarageType'] == 'Detchd'].mode()[0])
df_total['GarageArea'] = df_total['GarageArea'].fillna(df_total['GarageArea'][(df_total['GarageType'] == 'Detchd') & (df_total['GarageCars'] == 2.0)].mean())
df_total.info()
df_total = df_total.replace({'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 'PavedDrive': {'NA': 0, 'P': 1, 'Y': 2}, 'PoolQC': {'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, 'CentralAir': {'N': 0, 'Y': 1}})
df_num_ord = df_total.select_dtypes(include=['int64', 'float64'])
df_cat = df_total.select_dtypes(include=['object']).astype(str)
corr = df_num_ord.corr(method='spearman')
kot = corr[((corr >= 0.7) | (corr <= -0.7)) & (corr != 1.0)]
(fig, ax) = plt.subplots(figsize=(18, 18))
sns.heatmap(kot, ax=ax, center=0, annot=True, fmt='0.2f', cbar=False, cmap=sns.diverging_palette(110, 110, s=100, l=60, center='light', as_cmap=True))
fig.savefig('num_corr.jpeg', dpi=1200, bbox_inches='tight')

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr(method='spearman')
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
high_corr_features = correlation(df_num_ord, 0.7)
df_num_ord.drop(high_corr_features, axis=1, inplace=True, errors='ignore')
df_total.drop(high_corr_features, axis=1, inplace=True, errors='ignore')
sns.distplot(df_train['SalePrice'], fit=norm)