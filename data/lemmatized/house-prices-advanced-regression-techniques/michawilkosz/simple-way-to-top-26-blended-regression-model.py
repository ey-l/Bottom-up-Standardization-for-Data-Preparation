import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
sns.set(style='whitegrid')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.info()
_input1 = _input1.drop('Id', axis=1)
_input0 = _input0.drop('Id', axis=1)
_input1.head()
_input0.head()
_input1['SalePrice'].describe()
sns.distplot(_input1['SalePrice'])
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
train_quantitative = _input1[[d for d in _input1.columns if _input1.dtypes[d] != 'object']].copy()
test_quantitative = _input0[[d for d in _input0.columns if _input0.dtypes[d] != 'object']].copy()
train_qualitative = _input1[[d for d in _input1.columns if _input1.dtypes[d] == 'object']].copy()
test_qualitative = _input0[[d for d in _input0.columns if _input0.dtypes[d] == 'object']].copy()
train_quantitative.describe()
train_qualitative.describe()
train_qualitative.columns
missing_train = train_qualitative.isnull().sum().sort_values(ascending=False)
percentage_train = (train_qualitative.isnull().sum() / train_qualitative.isnull().count()).sort_values(ascending=False)
train_info = pd.concat([missing_train, percentage_train], keys=['Missing', 'Percentage'], axis=1)
train_info.head(25)
fig = plt.figure(figsize=(10, 5))
train_plot = sns.barplot(x=missing_train.index[0:20], y=missing_train[0:20])
train_plot.set_xticklabels(train_plot.get_xticklabels(), rotation=90)
plt.title('Number of missing values in categorical data(train)')
missing_test = test_qualitative.isnull().sum().sort_values(ascending=False)
percentage_test = (test_qualitative.isnull().sum() / test_qualitative.isnull().count()).sort_values(ascending=False)
test_info = pd.concat([missing_test, percentage_test], keys=['Missing', 'Percentage'], axis=1)
test_info.head(25)
fig = plt.figure(figsize=(10, 5))
test_plot = sns.barplot(x=missing_test.index[0:20], y=missing_test[0:20])
test_plot.set_xticklabels(test_plot.get_xticklabels(), rotation=90)
plt.title('Number of missing values in categorical data(test)')
for column in train_qualitative.columns:
    train_qualitative[column] = train_qualitative[column].fillna('None')
for column in test_qualitative.columns:
    test_qualitative[column] = test_qualitative[column].fillna('None')
train_qualitative['Electrical'] = train_qualitative['Electrical'].fillna(method='pad')
test_qualitative['SaleType'] = test_qualitative['SaleType'].fillna(method='pad')
test_qualitative['KitchenQual'] = test_qualitative['KitchenQual'].fillna(method='pad')
test_qualitative['Exterior1st'] = test_qualitative['Exterior1st'].fillna(method='pad')
test_qualitative['Exterior2nd'] = test_qualitative['Exterior2nd'].fillna(method='pad')
test_qualitative['Functional'] = test_qualitative['Functional'].fillna(method='pad')
test_qualitative['Utilities'] = test_qualitative['Utilities'].fillna(method='pad')
test_qualitative['MSZoning'] = test_qualitative['MSZoning'].fillna(method='pad')
train_qualitative.isnull().sum().sum()
test_qualitative.isnull().sum().sum()
train_qualitative.shape
test_qualitative.shape
top = 10
corr = train_quantitative.corr()
top10 = corr.nlargest(top, 'SalePrice')['SalePrice'].index
corr_top10 = train_quantitative[top10].corr()
(f, ax) = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_top10, square=True, ax=ax, annot=True, fmt='.2f', annot_kws={'size': 12})
plt.title('Top correlated quantitative features of dataset')
corr = train_quantitative.corr()['SalePrice'].sort_values(ascending=False)
print(corr)
(fig, ax) = plt.subplots(2, 2, figsize=(15, 15))
sns.scatterplot(data=train_quantitative, x='SalePrice', y='GarageArea', ax=ax[0][0])
sns.scatterplot(data=train_quantitative, x='SalePrice', y='GarageCars', ax=ax[0][1])
sns.scatterplot(data=train_quantitative, x='SalePrice', y='TotRmsAbvGrd', ax=ax[1][0])
sns.scatterplot(data=train_quantitative, x='SalePrice', y='GrLivArea', ax=ax[1][1])
corr = train_quantitative.corr()['SalePrice'].sort_values(ascending=False)
print(corr)
train_quantitative = train_quantitative.drop(['GarageCars', 'TotRmsAbvGrd'], axis=1)
test_quantitative = test_quantitative.drop(['GarageCars', 'TotRmsAbvGrd'], axis=1)
(fig, ax) = plt.subplots(17, 2, figsize=(15, 60))
for i in range(len(train_quantitative.columns) - 1):
    r = i // 2
    c = i % 2
    sns.scatterplot(data=train_quantitative, x=train_quantitative.columns[i], y='SalePrice', hue='SalePrice', palette='rocket', ax=ax[r][c])
plt.tight_layout()
missing_train_num = train_quantitative.isnull().sum().sort_values(ascending=False)
percentage_train_num = (train_quantitative.isnull().sum() / train_quantitative.isnull().count()).sort_values(ascending=False)
train_info = pd.concat([missing_train_num, percentage_train_num], keys=['Missing', 'Percentage'], axis=1)
train_info.head(10)
fig = plt.figure(figsize=(10, 5))
test_plot = sns.barplot(x=missing_train_num.index[0:5], y=missing_train_num[0:5])
test_plot.set_xticklabels(test_plot.get_xticklabels(), rotation=90)
plt.title('Number of missing values in numerical data(test)')
missing_test_num = test_quantitative.isnull().sum().sort_values(ascending=False)
percentage_test_num = (test_quantitative.isnull().sum() / test_quantitative.isnull().count()).sort_values(ascending=False)
train_info = pd.concat([missing_test_num, percentage_test_num], keys=['Missing', 'Percentage'], axis=1)
train_info.head(10)
fig = plt.figure(figsize=(10, 5))
test_plot = sns.barplot(x=missing_test_num.index[0:5], y=missing_test_num[0:5])
test_plot.set_xticklabels(test_plot.get_xticklabels(), rotation=90)
plt.title('Number of missing values in numerical data(test)')
train_quantitative['LotFrontage'] = train_quantitative.groupby(train_qualitative['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test_quantitative['LotFrontage'] = test_quantitative.groupby(test_qualitative['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train_quantitative['GarageYrBlt'] = train_quantitative['GarageYrBlt'].fillna(train_quantitative['GarageYrBlt'].median())
test_quantitative['GarageYrBlt'] = test_quantitative['GarageYrBlt'].fillna(test_quantitative['GarageYrBlt'].median())
for column in train_quantitative.columns:
    train_quantitative[column] = train_quantitative[column].fillna(0)
for column in test_quantitative.columns:
    test_quantitative[column] = test_quantitative[column].fillna(0)
train_quantitative.isnull().sum().sum()
test_quantitative.isnull().sum().sum()
train_quantitative['TotalSF'] = train_quantitative['TotalBsmtSF'] + train_quantitative['1stFlrSF'] + train_quantitative['2ndFlrSF']
train_quantitative = train_quantitative.drop(columns={'1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'})
train_quantitative['YrBltAndRemod'] = train_quantitative['YearBuilt'] + train_quantitative['YearRemodAdd']
train_quantitative = train_quantitative.drop(columns={'YearBuilt', 'YearRemodAdd'})
train_quantitative['Bsmt'] = train_quantitative['BsmtFinSF1'] + train_quantitative['BsmtFinSF2']
train_quantitative = train_quantitative.drop(columns={'BsmtFinSF1', 'BsmtFinSF2'})
train_quantitative['TotalBathroom'] = train_quantitative['FullBath'] + 0.5 * train_quantitative['HalfBath'] + train_quantitative['BsmtFullBath'] + 0.5 * train_quantitative['BsmtHalfBath']
train_quantitative = train_quantitative.drop(columns={'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'})
test_quantitative['TotalSF'] = test_quantitative['TotalBsmtSF'] + test_quantitative['1stFlrSF'] + test_quantitative['2ndFlrSF']
test_quantitative = test_quantitative.drop(columns={'1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'})
test_quantitative['YrBltAndRemod'] = test_quantitative['YearBuilt'] + test_quantitative['YearRemodAdd']
test_quantitative = test_quantitative.drop(columns={'YearBuilt', 'YearRemodAdd'})
test_quantitative['Bsmt'] = test_quantitative['BsmtFinSF1'] + test_quantitative['BsmtFinSF2']
test_quantitative = test_quantitative.drop(columns={'BsmtFinSF1', 'BsmtFinSF2'})
test_quantitative['TotalBathroom'] = test_quantitative['FullBath'] + 0.5 * test_quantitative['HalfBath'] + test_quantitative['BsmtFullBath'] + 0.5 * test_quantitative['BsmtHalfBath']
test_quantitative = test_quantitative.drop(columns={'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'})
(fig, ax) = plt.subplots(14, 2, figsize=(15, 60))
for i in range(len(train_quantitative.columns)):
    r = i // 2
    c = i % 2
    sns.scatterplot(data=train_quantitative, x=train_quantitative.columns[i], y='SalePrice', hue='SalePrice', palette='viridis', ax=ax[r][c])
plt.tight_layout()
numerical_to_categorical = ['TotalBathroom', 'Fireplaces', 'MSSubClass', 'OverallCond', 'BedroomAbvGr', 'LowQualFinSF', 'KitchenAbvGr', 'MoSold', 'YrSold', 'PoolArea', 'MiscVal', 'LotArea', '3SsnPorch', 'ScreenPorch']
numerical_categorical_train = train_quantitative[numerical_to_categorical]
train_quantitative = train_quantitative.drop(columns=numerical_to_categorical, inplace=False)
train_quantitative
numerical_categorical_test = test_quantitative[numerical_to_categorical]
test_quantitative = test_quantitative.drop(columns=numerical_to_categorical, inplace=False)
test_quantitative
corr = train_quantitative.corr()['SalePrice'].sort_values(ascending=False)
print(corr)
train_qualitative = pd.concat([train_qualitative, numerical_categorical_train], axis=1)
test_qualitative = pd.concat([test_qualitative, numerical_categorical_test], axis=1)
qualitative = pd.concat((train_qualitative, test_qualitative), sort=False).reset_index(drop=True)
qualitative = pd.get_dummies(qualitative)
train_qualitative_final = qualitative[:train_qualitative.shape[0]]
test_qualitative_final = qualitative[train_qualitative.shape[0]:]
train_qualitative_final.shape
test_qualitative_final.shape
y_pred = np.log1p(_input1['SalePrice'])
y_train = np.log1p(train_quantitative['SalePrice'])
train_quantitative = train_quantitative.drop('SalePrice', axis=1, inplace=False)
sns.distplot(y_train)
print('Train quantitative skewness')
skewed_features_train = []
for column in train_quantitative:
    skew = abs(train_quantitative[column].skew())
    print('{:15}'.format(column), 'Skewness: {:05.2f}'.format(skew))
    if skew > 0.5:
        skewed_features_train.append(column)
skewed_features_train
lam = 0.15
for feat in skewed_features_train:
    train_quantitative[feat] = boxcox1p(train_quantitative[feat], lam)
print('Test quantitative skewness')
skewed_features_test = []
for column in test_quantitative:
    skew = abs(test_quantitative[column].skew())
    print('{:15}'.format(column), 'Skewness: {:05.2f}'.format(skew))
    if skew > 0.75:
        skewed_features_test.append(column)
skewed_features_test
lam = 0.15
for feat in skewed_features_test:
    test_quantitative[feat] = boxcox1p(test_quantitative[feat], lam)
scaling = StandardScaler()
train_quantitative_final = pd.DataFrame(scaling.fit_transform(train_quantitative), columns=train_quantitative.columns)
test_quantitative_final = pd.DataFrame(scaling.fit_transform(test_quantitative), columns=test_quantitative.columns)
train_final = train_quantitative_final.merge(train_qualitative_final, left_index=True, right_index=True).reset_index(drop=True)
train_final.head()
test_qualitative_final = test_qualitative_final.reset_index(drop=True)
test_final = test_quantitative_final.merge(test_qualitative_final, left_index=True, right_index=True).reset_index(drop=True)
test_final.head()
train_final.shape
test_final.shape
(X_train, X_test, Y_train, Y_test) = train_test_split(train_final, y_train, test_size=0.3, random_state=0)

def rmse(actual, predicted):
    return str(np.sqrt(mean_squared_error(actual, predicted)))
lin_reg = LinearRegression()