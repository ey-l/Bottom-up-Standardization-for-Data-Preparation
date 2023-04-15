import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
m_train = df_train.shape[0]
m_test = df_test.shape[0]
(m_train, m_test)
df = pd.concat([df_train, df_test])
assert df.shape[0] == m_train + m_test
df.head()
(target_col, target) = ('SalePrice', df['SalePrice'])
cols_to_drop = ['Id', target_col]
df.drop(cols_to_drop, axis=1, inplace=True)
df.select_dtypes(object).columns
df.select_dtypes(np.number).columns
num_to_obj_cols = ['MSSubClass', 'MoSold']
df[num_to_obj_cols] = df[num_to_obj_cols].astype(object)
cols_cat = df.select_dtypes(object).columns.to_list()
cols_num = df.select_dtypes(np.number).columns.to_list()
cols_cat_na = df[cols_cat].isnull().sum()[df[cols_cat].isnull().sum() > 0]
cols_cat_na
mode_filled_cols = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'KitchenQual', 'Functional', 'SaleType']
for col in mode_filled_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
none_filled_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for col in none_filled_cols:
    df[col].fillna('None', inplace=True)
df[cols_cat].isnull().sum().sum()
df.select_dtypes(object).columns
oe = OrdinalEncoder(categories=[['Reg', 'IR1', 'IR2', 'IR3']])
df.loc[:, 'LotShape'] = oe.fit_transform(df[['LotShape']])
oe = OrdinalEncoder(categories=[['Gtl', 'Mod', 'Sev']])
df.loc[:, 'LandSlope'] = oe.fit_transform(df[['LandSlope']])
qual_oe = OrdinalEncoder(categories=[['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])
for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    df.loc[:, col] = qual_oe.fit_transform(df[[col]])
oe = OrdinalEncoder(categories=[['None', 'No', 'Mn', 'Av', 'Gd']])
df.loc[:, 'BsmtExposure'] = oe.fit_transform(df[['BsmtExposure']])
oe = OrdinalEncoder(categories=[['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']])
for col in ['BsmtFinType1', 'BsmtFinType2']:
    df.loc[:, col] = oe.fit_transform(df[[col]])
oe = OrdinalEncoder(categories=[['None', 'Unf', 'RFn', 'Fin']])
df.loc[:, 'GarageFinish'] = oe.fit_transform(df[['GarageFinish']])
oe = OrdinalEncoder(categories=[['N', 'P', 'Y']])
df.loc[:, 'PavedDrive'] = oe.fit_transform(df[['PavedDrive']])
oe = OrdinalEncoder(categories=[['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']])
df.loc[:, 'Fence'] = oe.fit_transform(df[['Fence']])
cols_cat = df.select_dtypes(object).columns.to_list()
cols_num = df.select_dtypes(np.number).columns.to_list()
cols_num_na = df[cols_num].isnull().sum()[df[cols_num].isnull().sum() > 0]
cols_num_na
zero_filled_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
for col in zero_filled_cols:
    df[col].fillna(0, inplace=True)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
cols_num_na = df[cols_num].isnull().sum()[df[cols_num].isnull().sum() > 0]
cols_num_na

def catboost_imputer(df, cols_num_na):
    cols_to_impute = df[cols_num_na]
    df.drop(cols_num_na, axis=1, inplace=True)
    for col in cols_num_na:
        (X, y) = (df.copy(), cols_to_impute[col])
        (train_indexes, test_indexes) = (~y.isnull(), y.isnull())
        (X_train, X_test) = (X.loc[train_indexes, :], X.loc[test_indexes, :])
        (y_train, y_test) = (y.loc[train_indexes], y.loc[test_indexes])
        model = CatBoostRegressor(max_depth=8, random_seed=10, subsample=0.65, n_estimators=1000, cat_features=cols_cat, verbose=0)