import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from scipy import stats
from scipy.stats import norm, skew
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1 = _input1.drop('Id', axis=1, inplace=False)
test_Id = _input0.Id
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input1.head()
' 1: Find NaN values '
NaN_values = _input1.isna().sum() / 1460
NaN_values = NaN_values.loc[NaN_values > 0]
print('NaN_values')
print(NaN_values)
print('')
try:
    _input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis='columns', inplace=False)
    _input0 = _input0.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis='columns', inplace=False)
except:
    pass
' 2: Heatmap'
corr = _input1.corr()
print('Heatmap')
print(corr.SalePrice.sort_values(ascending=False))
plt.subplots(figsize=(14, 8))
print('Correlations')
print(sns.heatmap(corr))
'3: Visualizations'
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
print('Missing values')
for feature in features:
    print(_input1[feature].isna().sum())
plt.figure(figsize=(14, 6))
sns.scatterplot(data=_input1, x='OverallQual', y='SalePrice')
sns.lmplot(data=_input1, x='GarageArea', y='SalePrice', hue='GarageCars')
plt.figure(figsize=(15, 15))
sns.swarmplot(x=_input1.GarageCars, y=_input1.SalePrice)
plt.figure(figsize=(14, 6))
sns.scatterplot(data=_input1, x='TotalBsmtSF', y='SalePrice', label='Basement Area')
sns.scatterplot(data=_input1, x='1stFlrSF', y='SalePrice', label='First Floor Area')
plt.figure(figsize=(15, 10))
sns.swarmplot(x=_input1.FullBath, y=_input1.SalePrice)
plt.figure(figsize=(15, 10))
sns.swarmplot(x=_input1.TotRmsAbvGrd, y=_input1.SalePrice)
plt.figure(figsize=(15, 10))
sns.scatterplot(x=_input1.YearBuilt, y=_input1.SalePrice)
plt.figure(figsize=(15, 10))
sns.scatterplot(x=_input1.YearRemodAdd, y=_input1.SalePrice)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 3000) & (_input1['SalePrice'] < 240000)].index, inplace=False)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 3000) & (_input1['SalePrice'] > 700000)].index, inplace=False)
_input1 = _input1.drop(_input1[(_input1['OverallQual'] == 10) & (_input1['SalePrice'] < 200000)].index, inplace=False)
_input1 = _input1.drop(_input1[(_input1['TotalBsmtSF'] > 4000) & (_input1['1stFlrSF'] > 4000)].index, inplace=False)
_input1 = _input1.drop(_input1[_input1['TotRmsAbvGrd'] == 14].index, inplace=False)
_input1 = _input1.drop(_input1[(_input1['YearBuilt'] < 1900) & (_input1['SalePrice'] > 400000)].index, inplace=False)
plt.figure()
sns.distplot(_input1['SalePrice'], fit=norm)
plt.figure()
sns.distplot(_input1['SalePrice'], fit=norm)
fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=4, wspace=1)
for feature in features:
    ax = fig.add_subplot(5, 2, features.index(feature) + 1)
    sns.distplot(_input1[feature], fit=norm)
    plt.xlabel(feature)
_input1[features]
test_data = _input0[features]
tr_features = ['SalePrice'] + features
model_data = _input1[tr_features]
model_data
(X_train, X_test, y_train, y_test) = tts(model_data.iloc[:, 1:], model_data.iloc[:, 0:1], test_size=0.2)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor