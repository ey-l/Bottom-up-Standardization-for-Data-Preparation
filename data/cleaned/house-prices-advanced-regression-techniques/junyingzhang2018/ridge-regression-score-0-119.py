import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')
import scipy.stats as stats
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)
print(test.shape)
print(set(train.columns) - set(test.columns))
print(train.info())
print('-----------------------' * 3)
print(test.info())
train.drop(['Id'], axis=1).describe()
print(train.select_dtypes(exclude=['object']).shape)
print(train.select_dtypes(include=['object']).shape)
print(train[:5])
print('/n')
print(train.tail(5))
plt.figure(figsize=(13, 10))
sns.heatmap(train.corr(), vmax=0.8)
train.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'TotalBsmtSF'], axis=1, inplace=True)
test.drop(['GarageYrBlt', 'TotRmsAbvGrd', 'TotalBsmtSF'], axis=1, inplace=True)
train.corr()['SalePrice'].sort_values(ascending=False).head(11)
train.corr()['SalePrice'].abs().sort_values(ascending=False).head(11)
top_corr_features = train.corr()['SalePrice'].abs().sort_values(ascending=False).head(11).index
top_corr_features
plt.figure(figsize=(10, 7))
sns.boxplot(x='OverallQual', y='SalePrice', data=train)
plt.scatter(x='GrLivArea', y='SalePrice', data=train, color='r', marker='*')
train['GrLivArea'].sort_values(ascending=False).head(2)
train.index[[523, 1298]]
print(train.shape)
train.drop(train.index[[523, 1298]], inplace=True)
print(train.shape)
print(top_corr_features)
box_feature = ['SalePrice', 'OverallQual', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces']
scatter_feature = ['SalePrice', 'GrLivArea', '1stFlrSF', 'GarageArea']
sns.pairplot(train[scatter_feature])
sns.pairplot(train[box_feature], kind='scatter', diag_kind='hist')
train.isnull().sum().sort_values(ascending=False)
train_nan_pct = train.isnull().sum() / train.isnull().count()
train_nan_pct = train_nan_pct[train_nan_pct > 0]
train_nan_pct.sort_values(ascending=False)
train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
train['GarageQual'].value_counts()
train_impute_index = train_nan_pct[train_nan_pct < 0.3].index
train_impute_index
train_impute_mode = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
train_impute_median = ['LotFrontage', 'MasVnrArea']
for feature in train_impute_mode:
    train[feature].fillna(train[feature].mode()[0], inplace=True)
    test[feature].fillna(test[feature].mode()[0], inplace=True)
for feature in train_impute_median:
    train[feature].fillna(train[feature].median(), inplace=True)
    test[feature].fillna(test[feature].median(), inplace=True)
train.isnull().sum().sort_values(ascending=False).head(5)
test_only_nan = test.isnull().sum().sort_values(ascending=False)
test_only_nan = test_only_nan[test_only_nan > 0]
print(test_only_nan.index)
test_impute_mode = ['MSZoning', 'BsmtFullBath', 'Utilities', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'GarageCars', 'KitchenQual']
test_impute_median = ['BsmtFinSF2', 'GarageArea', 'BsmtFinSF1', 'BsmtUnfSF']
for feature in test_impute_mode:
    test[feature].fillna(test[feature].mode()[0], inplace=True)
for feature in test_impute_median:
    test[feature].fillna(test[feature].median(), inplace=True)
test.isnull().sum().sort_values(ascending=False).head(5)
TestId = test['Id']
total_features = pd.concat((train.drop(['Id', 'SalePrice'], axis=1), test.drop(['Id'], axis=1)))
total_features = pd.get_dummies(total_features, drop_first=True)
train_features = total_features[0:train.shape[0]]
test_features = total_features[train.shape[0]:]
sns.distplot(train['SalePrice'])
train['Log SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['Log SalePrice'])
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.kdeplot(train['SalePrice'], legend=True)
plt.subplot(1, 2, 2)
sns.kdeplot(train['Log SalePrice'], legend=True)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(train_features, train[['SalePrice']], test_size=0.3, random_state=100)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
rmse = []
alpha = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
for alph in alpha:
    ridge = Ridge(alpha=alph, copy_X=True, fit_intercept=True)