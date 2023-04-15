import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df
df.set_index('Id', inplace=True)
df.info()
len(df)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        print('%s has %d null values' % (col, df[col].isnull().sum()))
more_than_1000 = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
df.drop(more_than_1000, axis=1, inplace=True)
test.drop(more_than_1000, axis=1, inplace=True)
df[df['GarageType'].isna()][['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']].isna().sum()
garage = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
for g in garage:
    df[g] = df[g].fillna('NA')
    test[g] = test[g].fillna('NA')
df['GarageYrBlt'] = df['GarageYrBlt'].astype(str)
test['GarageYrBlt'] = test['GarageYrBlt'].astype(str)
df.corr()['LotFrontage'].sort_values(ascending=False)[1:5]
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
df['Electrical'] = df['Electrical'].fillna('Mix')
test['Electrical'] = test['Electrical'].fillna('Mix')
df[df['BsmtQual'].isnull()][['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isna().sum()
for c in range(len(df.columns)):
    if 'Bsmt' in df.columns[c]:
        print('%s-->%d' % (df.columns[c], c))
df[~df['BsmtQual'].isnull() & df['BsmtExposure'].isnull()]
df.iat[948, 31] = 'No'
df[~df['BsmtQual'].isnull() & df['BsmtFinType2'].isnull()]
df.iat[332, 34] = 'Unf'
basement = ['Qual', 'Cond', 'Exposure', 'FinType1', 'FinType2']
for b in basement:
    df['Bsmt' + b] = df['Bsmt' + b].fillna('NA')
    test['Bsmt' + b] = test['Bsmt' + b].fillna('NA')
test.isna().sum()[test.isna().sum() > 0]
test.set_index('Id', inplace=True)
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode())
test['Utilities'] = test['Utilities'].fillna('AllPub')
test['Exterior1st'] = test['Exterior1st'].fillna('VinylSd')
test['Exterior2nd'] = test['Exterior2nd'].fillna('VinylSd')
test[test['BsmtFinSF1'].isna()][['BsmtUnfSF', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtCond', 'BsmtFullBath']]
test[test['BsmtFullBath'].isna()].iloc[:, 45:47].index
test.at[2189, 'BsmtHalfBath'] = 0
test.at[2189, 'BsmtFullBath'] = 0
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(0)
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)
test['KitchenQual'] = test['KitchenQual'].fillna(df['KitchenQual'].mode().loc[0])
test['Functional'] = test['Functional'].fillna(df['Functional'].mode().loc[0])
test['GarageCars'] = test['GarageCars'].fillna(df['GarageCars'].mode().loc[0])
test['GarageArea'] = test['GarageArea'].fillna(df['GarageArea'].mean())
test['SaleType'] = test['SaleType'].fillna(df['SaleType'].mode().loc[0])
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
cols = ['1stFlrSF', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'LotFrontage']
lotfront_df = df[cols].copy()
null_lotfront_df = lotfront_df[lotfront_df['LotFrontage'].isnull()].copy()
lotfront_df.dropna(inplace=True)
X = lotfront_df.drop('LotFrontage', axis=1)
y = lotfront_df['LotFrontage']
sc = StandardScaler()
X_sc = sc.fit_transform(X)
lr = LinearRegression()