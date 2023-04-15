import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set2')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head(10)
test.head(10)
train.info()
msno.bar(train.iloc[:, :40])
msno.bar(train.iloc[:, 40:])
train.iloc[:, :40].describe()
train.iloc[:, 40:-1].describe()
pd.DataFrame(train['SalePrice'].describe())
plt.figure(figsize=(12, 7))
sns.distplot(train['SalePrice']).set(ylabel=None, xlabel=None)
plt.title('House price distribution histogram', fontsize=18)

train['SalePrice'] = np.log1p(train['SalePrice'])
plt.figure(figsize=(12, 7))
sns.distplot(train['SalePrice'])
plt.title('House price distribution histogram after fix', fontsize=18)

corr_train = train.corr()
colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson correlation matrix between features', y=1, size=15)
sns.heatmap(corr_train, vmax=0.8, square=True, cmap=colormap)

train.head()
highest_corr_features = corr_train.index[abs(corr_train['SalePrice']) > 0.5]
plt.figure(figsize=(14, 12))
plt.title('Pearson correlation matrix between features and "SalePrice"', y=1, size=15)
sns.heatmap(train[highest_corr_features].corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

SalePrice = pd.DataFrame(corr_train['SalePrice'].sort_values(ascending=False))
SalePrice
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[features])

y_train = train['SalePrice']
test_id = test['Id']
data = pd.concat([train, test], axis=0, sort=False)
data = data.drop(['Id', 'SalePrice'], axis=1)
Total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
data.drop(missing_data[missing_data['Total'] > 5].index, axis=1, inplace=True)
print(data.isnull().sum().max())
numeric_missed = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars']
for feature in numeric_missed:
    data[feature].fillna(0, inplace=True)
categorical_missed = ['Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning', 'Electrical', 'KitchenQual']
for feature in categorical_missed:
    data[feature].fillna(data[feature].mode()[0], inplace=True)
data['Functional'].fillna('Typ', inplace=True)
data.isnull().sum().max()
from scipy.stats import skew
numeric = data.dtypes[data.dtypes != 'object'].index
skewed = data[numeric].apply(lambda col: skew(col)).sort_values(ascending=False)
skewed = skewed[abs(skewed) > 0.5]
for feature in skewed.index:
    data[feature] = np.log1p(data[feature])
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data = pd.get_dummies(data)
data
x_train = data[:len(y_train)]
x_test = data[len(y_train):]
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

import xgboost as XGB
xgb_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.464, reg_lambda=0.8571, subsample=0.5213, random_state=7, nthread=-1)