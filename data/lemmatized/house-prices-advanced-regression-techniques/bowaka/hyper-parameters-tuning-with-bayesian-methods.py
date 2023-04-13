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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
df = pd.concat([_input1.drop('SalePrice', axis=1), _input0])
df = df.set_index('Id')
df[['PoolQC']] = df[['PoolQC']].fillna('na')
df[['MiscFeature']] = df[['MiscFeature']].fillna('na')
df[['Alley']] = df[['Alley']].fillna('na')
df[['Fence']] = df[['Fence']].fillna('na')
df[['FireplaceQu']] = df[['FireplaceQu']].fillna('na')
lfront = df[['LotArea', 'LotFrontage']].dropna()