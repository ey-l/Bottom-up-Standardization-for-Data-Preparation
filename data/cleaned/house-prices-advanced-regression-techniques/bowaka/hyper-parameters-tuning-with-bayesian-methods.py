import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import STATUS_OK
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index)
df = pd.concat([df_train.drop('SalePrice', axis=1), df_test])
df = df.set_index('Id')
df[['PoolQC']] = df[['PoolQC']].fillna('na')
df[['MiscFeature']] = df[['MiscFeature']].fillna('na')
df[['Alley']] = df[['Alley']].fillna('na')
df[['Fence']] = df[['Fence']].fillna('na')
df[['FireplaceQu']] = df[['FireplaceQu']].fillna('na')
lfront = df[['LotArea', 'LotFrontage']].dropna()