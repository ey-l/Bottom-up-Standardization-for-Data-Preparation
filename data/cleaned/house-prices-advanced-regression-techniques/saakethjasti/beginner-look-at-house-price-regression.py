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
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data.head()
data.drop('Id', axis=1, inplace=True)
test_Id = data_test.Id
data_test.drop('Id', axis=1, inplace=True)
data.head()
' 1: Find NaN values '
NaN_values = data.isna().sum() / 1460
NaN_values = NaN_values.loc[NaN_values > 0]
print('NaN_values')
print(NaN_values)
print('')
try:
    data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis='columns', inplace=True)
    data_test.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis='columns', inplace=True)
except:
    pass
' 2: Heatmap'
corr = data.corr()
print('Heatmap')
print(corr.SalePrice.sort_values(ascending=False))
plt.subplots(figsize=(14, 8))
print('Correlations')
print(sns.heatmap(corr))
'3: Visualizations'
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
print('Missing values')
for feature in features:
    print(data[feature].isna().sum())
plt.figure(figsize=(14, 6))
sns.scatterplot(data=data, x='OverallQual', y='SalePrice')
sns.lmplot(data=data, x='GarageArea', y='SalePrice', hue='GarageCars')
plt.figure(figsize=(15, 15))
sns.swarmplot(x=data.GarageCars, y=data.SalePrice)
plt.figure(figsize=(14, 6))
sns.scatterplot(data=data, x='TotalBsmtSF', y='SalePrice', label='Basement Area')
sns.scatterplot(data=data, x='1stFlrSF', y='SalePrice', label='First Floor Area')
plt.figure(figsize=(15, 10))
sns.swarmplot(x=data.FullBath, y=data.SalePrice)
plt.figure(figsize=(15, 10))
sns.swarmplot(x=data.TotRmsAbvGrd, y=data.SalePrice)
plt.figure(figsize=(15, 10))
sns.scatterplot(x=data.YearBuilt, y=data.SalePrice)
plt.figure(figsize=(15, 10))
sns.scatterplot(x=data.YearRemodAdd, y=data.SalePrice)
data.drop(data[(data['GrLivArea'] > 3000) & (data['SalePrice'] < 240000)].index, inplace=True)
data.drop(data[(data['GrLivArea'] > 3000) & (data['SalePrice'] > 700000)].index, inplace=True)
data.drop(data[(data['OverallQual'] == 10) & (data['SalePrice'] < 200000)].index, inplace=True)
data.drop(data[(data['TotalBsmtSF'] > 4000) & (data['1stFlrSF'] > 4000)].index, inplace=True)
data.drop(data[data['TotRmsAbvGrd'] == 14].index, inplace=True)
data.drop(data[(data['YearBuilt'] < 1900) & (data['SalePrice'] > 400000)].index, inplace=True)
plt.figure()
sns.distplot(data['SalePrice'], fit=norm)
plt.figure()
sns.distplot(data['SalePrice'], fit=norm)
fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=4, wspace=1)
for feature in features:
    ax = fig.add_subplot(5, 2, features.index(feature) + 1)
    sns.distplot(data[feature], fit=norm)
    plt.xlabel(feature)
data[features]
test_data = data_test[features]
tr_features = ['SalePrice'] + features
model_data = data[tr_features]
model_data
(X_train, X_test, y_train, y_test) = tts(model_data.iloc[:, 1:], model_data.iloc[:, 0:1], test_size=0.2)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor