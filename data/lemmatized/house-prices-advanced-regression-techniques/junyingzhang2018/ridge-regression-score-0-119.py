import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import scipy.stats as stats
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.shape)
print(_input0.shape)
print(set(_input1.columns) - set(_input0.columns))
print(_input1.info())
print('-----------------------' * 3)
print(_input0.info())
_input1.drop(['Id'], axis=1).describe()
print(_input1.select_dtypes(exclude=['object']).shape)
print(_input1.select_dtypes(include=['object']).shape)
print(_input1[:5])
print('/n')
print(_input1.tail(5))
plt.figure(figsize=(13, 10))
sns.heatmap(_input1.corr(), vmax=0.8)
_input1 = _input1.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'TotalBsmtSF'], axis=1, inplace=False)
_input0 = _input0.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'TotalBsmtSF'], axis=1, inplace=False)
_input1.corr()['SalePrice'].sort_values(ascending=False).head(11)
_input1.corr()['SalePrice'].abs().sort_values(ascending=False).head(11)
top_corr_features = _input1.corr()['SalePrice'].abs().sort_values(ascending=False).head(11).index
top_corr_features
plt.figure(figsize=(10, 7))
sns.boxplot(x='OverallQual', y='SalePrice', data=_input1)
plt.scatter(x='GrLivArea', y='SalePrice', data=_input1, color='r', marker='*')
_input1['GrLivArea'].sort_values(ascending=False).head(2)
_input1.index[[523, 1298]]
print(_input1.shape)
_input1 = _input1.drop(_input1.index[[523, 1298]], inplace=False)
print(_input1.shape)
print(top_corr_features)
box_feature = ['SalePrice', 'OverallQual', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces']
scatter_feature = ['SalePrice', 'GrLivArea', '1stFlrSF', 'GarageArea']
sns.pairplot(_input1[scatter_feature])
sns.pairplot(_input1[box_feature], kind='scatter', diag_kind='hist')
_input1.isnull().sum().sort_values(ascending=False)
train_nan_pct = _input1.isnull().sum() / _input1.isnull().count()
train_nan_pct = train_nan_pct[train_nan_pct > 0]
train_nan_pct.sort_values(ascending=False)
_input1 = _input1.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=False)
_input0 = _input0.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=False)
_input1['GarageQual'].value_counts()
train_impute_index = train_nan_pct[train_nan_pct < 0.3].index
train_impute_index
train_impute_mode = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
train_impute_median = ['LotFrontage', 'MasVnrArea']
for feature in train_impute_mode:
    _input1[feature] = _input1[feature].fillna(_input1[feature].mode()[0], inplace=False)
    _input0[feature] = _input0[feature].fillna(_input0[feature].mode()[0], inplace=False)
for feature in train_impute_median:
    _input1[feature] = _input1[feature].fillna(_input1[feature].median(), inplace=False)
    _input0[feature] = _input0[feature].fillna(_input0[feature].median(), inplace=False)
_input1.isnull().sum().sort_values(ascending=False).head(5)
test_only_nan = _input0.isnull().sum().sort_values(ascending=False)
test_only_nan = test_only_nan[test_only_nan > 0]
print(test_only_nan.index)
test_impute_mode = ['MSZoning', 'BsmtFullBath', 'Utilities', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'GarageCars', 'KitchenQual']
test_impute_median = ['BsmtFinSF2', 'GarageArea', 'BsmtFinSF1', 'BsmtUnfSF']
for feature in test_impute_mode:
    _input0[feature] = _input0[feature].fillna(_input0[feature].mode()[0], inplace=False)
for feature in test_impute_median:
    _input0[feature] = _input0[feature].fillna(_input0[feature].median(), inplace=False)
_input0.isnull().sum().sort_values(ascending=False).head(5)
TestId = _input0['Id']
total_features = pd.concat((_input1.drop(['Id', 'SalePrice'], axis=1), _input0.drop(['Id'], axis=1)))
total_features = pd.get_dummies(total_features, drop_first=True)
train_features = total_features[0:_input1.shape[0]]
test_features = total_features[_input1.shape[0]:]
sns.distplot(_input1['SalePrice'])
_input1['Log SalePrice'] = np.log1p(_input1['SalePrice'])
sns.distplot(_input1['Log SalePrice'])
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.kdeplot(_input1['SalePrice'], legend=True)
plt.subplot(1, 2, 2)
sns.kdeplot(_input1['Log SalePrice'], legend=True)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(train_features, _input1[['SalePrice']], test_size=0.3, random_state=100)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
rmse = []
alpha = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
for alph in alpha:
    ridge = Ridge(alpha=alph, copy_X=True, fit_intercept=True)