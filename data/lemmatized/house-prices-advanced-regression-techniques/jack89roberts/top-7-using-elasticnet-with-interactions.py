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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
id_train = _input1.index
id_test = _input0.index
df_all = pd.concat([_input1, _input0], sort=True)
df_all.head(5)
cols_with_na = df_all.isnull().sum()
cols_with_na = cols_with_na[cols_with_na > 0]
print(cols_with_na.sort_values(ascending=False))
cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
for col in cols_fillna:
    df_all[col] = df_all[col].fillna('None', inplace=False)
df_all.loc[df_all.GarageYrBlt.isnull(), 'GarageYrBlt'] = df_all.loc[df_all.GarageYrBlt.isnull(), 'YearBuilt']
df_all.MasVnrArea = df_all.MasVnrArea.fillna(0, inplace=False)
df_all.BsmtFullBath = df_all.BsmtFullBath.fillna(0, inplace=False)
df_all.BsmtHalfBath = df_all.BsmtHalfBath.fillna(0, inplace=False)
df_all.BsmtFinSF1 = df_all.BsmtFinSF1.fillna(0, inplace=False)
df_all.BsmtFinSF2 = df_all.BsmtFinSF2.fillna(0, inplace=False)
df_all.BsmtUnfSF = df_all.BsmtUnfSF.fillna(0, inplace=False)
df_all.TotalBsmtSF = df_all.TotalBsmtSF.fillna(0, inplace=False)
df_all.GarageArea = df_all.GarageArea.fillna(0, inplace=False)
df_all.GarageCars = df_all.GarageCars.fillna(0, inplace=False)
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