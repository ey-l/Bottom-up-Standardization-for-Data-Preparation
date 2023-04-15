import pandas as pd
import numpy as np
from scipy import stats
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
id_train = df_train.index
id_test = df_test.index
df_all = pd.concat([df_train, df_test], sort=True)
df_all.head(5)
cols_with_na = df_all.isnull().sum()
cols_with_na = cols_with_na[cols_with_na > 0]
print(cols_with_na.sort_values(ascending=False))
cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
for col in cols_fillna:
    df_all[col].fillna('None', inplace=True)
df_all.loc[df_all.GarageYrBlt.isnull(), 'GarageYrBlt'] = df_all.loc[df_all.GarageYrBlt.isnull(), 'YearBuilt']
df_all.MasVnrArea.fillna(0, inplace=True)
df_all.BsmtFullBath.fillna(0, inplace=True)
df_all.BsmtHalfBath.fillna(0, inplace=True)
df_all.BsmtFinSF1.fillna(0, inplace=True)
df_all.BsmtFinSF2.fillna(0, inplace=True)
df_all.BsmtUnfSF.fillna(0, inplace=True)
df_all.TotalBsmtSF.fillna(0, inplace=True)
df_all.GarageArea.fillna(0, inplace=True)
df_all.GarageCars.fillna(0, inplace=True)
df_all[cols_fillna].head(5)

def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())
df_frontage = pd.get_dummies(df_all.drop('SalePrice', axis=1))
for col in df_frontage.drop('LotFrontage', axis=1).columns:
    df_frontage[col] = scale_minmax(df_frontage[col])
lf_train = df_frontage.dropna()
lf_train_y = lf_train.LotFrontage
lf_train_X = lf_train.drop('LotFrontage', axis=1)
lr = Ridge()