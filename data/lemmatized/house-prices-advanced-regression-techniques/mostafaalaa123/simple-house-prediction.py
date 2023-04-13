import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.columns.values
print(_input1.head())
print('-' * 20)
print(_input1.info())
full = pd.DataFrame()
medium = pd.DataFrame()
remove_me = pd.DataFrame()
features = _input1.columns.values
number_of_houses = 1460
for feature in features:
    if _input1[feature].count() == number_of_houses:
        full[feature] = _input1[feature]
    elif _input1[feature].count() > number_of_houses * 0.5:
        medium[feature] = _input1[feature]
    else:
        remove_me[feature] = _input1[feature]
print('Number of numerical features: ', end='')
print(len(_input1.select_dtypes(include=['number']).columns.values))
_input1.describe(exclude=['O'])
print('Number of categorical features: ', end='')
print(len(_input1.select_dtypes(include=['O']).columns.values))
_input1.describe(include=['O'])
plt.hist(_input1['SalePrice'])
plt.title('Sale Prices')
_input1 = _input1.drop(['Id'], axis=1)
_input1 = _input1.drop(remove_me.columns.values, axis=1)
numerical_data = _input1.select_dtypes(include=['number'])
categorical_data = _input1.select_dtypes(include=['object'])
feature_zero_ratio = {feature: numerical_data.loc[numerical_data[feature] == 0, feature].count() / 1460 for feature in numerical_data.columns.values}
feature_zero_ratio
for feature in numerical_data:
    if feature_zero_ratio[feature] > 0.3:
        numerical_data = numerical_data.drop([feature], axis=1)
        _input1 = _input1.drop([feature], axis=1)
        if feature in medium:
            medium = medium.drop([feature], axis=1)
_input1.shape
print(numerical_data.columns.values)
print(len(numerical_data.columns.values))
corrmat = numerical_data.corr()
(fig, ax) = plt.subplots(figsize=(12, 12))
sns.set(font_scale=1.25)
sns.heatmap(corrmat, vmax=0.8, annot=True, square=True, annot_kws={'size': 8}, fmt='.2f')
n = 10
most_largest_features = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index
zoomed_corrmat = np.corrcoef(numerical_data[most_largest_features].values.T)
(fig, ax) = plt.subplots(figsize=(6, 6))
sns.set(font_scale=1)
sns.heatmap(zoomed_corrmat, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=most_largest_features.values, xticklabels=most_largest_features.values)
print(most_largest_features.values)
sns.set()
most_largest_features = corrmat.nlargest(7, 'SalePrice')['SalePrice'].index
sns.pairplot(numerical_data[most_largest_features.values], size=1.5)
print(most_largest_features)
numerical_data = numerical_data.drop(['1stFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageYrBlt'], axis=1)
_input1 = _input1.drop(['1stFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageYrBlt'], axis=1)
numerical_data.columns.values
corr_with_price = numerical_data.corr()
corr_with_price = corr_with_price.sort_values(by='SalePrice', ascending=False)
corr_with_price['SalePrice']
numerical_data.columns.values
numerical_have_missing = pd.DataFrame()
categorical_have_missing = pd.DataFrame()
for feature in numerical_data.columns.values:
    if feature in medium:
        numerical_have_missing[feature] = numerical_data[feature]
for feature in categorical_data.columns.values:
    if feature in medium:
        categorical_have_missing[feature] = categorical_data[feature]
print(numerical_have_missing.columns.values)
print('-' * 30)
print(numerical_have_missing.info())
sns.histplot(numerical_have_missing['LotFrontage'])
plt.title('LotFrontage')
old_LotFrontage = list(numerical_have_missing['LotFrontage'].values)
missing_indices = list(numerical_have_missing.loc[numerical_have_missing['LotFrontage'].isnull(), 'LotFrontage'].index)
random_values = [random.randint(60, 80) for _ in range(1460 - numerical_have_missing['LotFrontage'].count())]
random_values_idx = 0
for missing_idx in missing_indices:
    old_LotFrontage[missing_idx] = random_values[random_values_idx]
    random_values_idx += 1
numerical_have_missing['LotFrontage'] = pd.Series(old_LotFrontage)
_input1['LotFrontage'] = pd.Series(old_LotFrontage)
print(numerical_have_missing['LotFrontage'].count())
print(_input1['LotFrontage'].count())
print(len(categorical_have_missing.columns.values))
print('-' * 30)
print(categorical_have_missing.columns.values)
print('-' * 30)
print(categorical_have_missing.count())
_input1 = _input1.drop(['FireplaceQu'], axis=1)
categorical_have_missing = categorical_have_missing.drop(['FireplaceQu'], axis=1)
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
for feature in categorical_have_missing:
    categorical_have_missing[feature] = imputer.fit_transform(categorical_have_missing[feature].values.reshape((-1, 1)))
    _input1[feature] = imputer.fit_transform(_input1[feature].values.reshape((-1, 1)))
print(len(categorical_have_missing.columns.values))
print('-' * 30)
print(categorical_have_missing.columns.values)
print('-' * 30)
print(categorical_have_missing.count())
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'])
_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 200000)].index
_input1['Id'] = pd.Series(_input1.index)
_input1 = _input1.drop(_input1[(_input1['Id'] == 1298) | (_input1['Id'] == 523)].index)
_input0 = _input0[_input1.drop(['SalePrice'], axis=1).columns.values]
_input1 = _input1.drop(['Id'], axis=1)
plt.scatter(_input1['GrLivArea'], _input1['SalePrice'])
print(_input1.shape)
_input1 = pd.get_dummies(_input1)
sns.distplot(_input1['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['SalePrice'], plot=plt)
_input1['SalePrice'] = np.log(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['SalePrice'], plot=plt)
sns.distplot(_input1['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['GrLivArea'], plot=plt)
_input1['GrLiveArea'] = np.log(_input1['GrLivArea'])
sns.distplot(_input1['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(_input1['GrLivArea'], plot=plt)
_input1.shape
target = _input1['SalePrice']
_input1 = _input1.drop(['SalePrice'], axis=1)
(X, y) = (_input1, target)
lin_reg = LinearRegression()