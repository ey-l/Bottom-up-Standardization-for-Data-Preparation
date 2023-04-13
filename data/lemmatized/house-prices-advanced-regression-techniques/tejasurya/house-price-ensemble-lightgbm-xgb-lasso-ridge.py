import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style='darkgrid', font_scale=1.6)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('train shape  {}'.format(_input1.shape))
print('test shape  {}'.format(_input0.shape))
test_copy = _input0.copy()
plt.figure()
_input1['SalePrice'].hist(bins=20)
plt.title('SalePrice' + ' before transformation')
from scipy import stats
_input1['SalePrice'] = np.log(_input1['SalePrice'])
_input1['z_score_target'] = np.abs(stats.zscore(_input1['SalePrice']))
_input1 = _input1.loc[_input1['z_score_target'] < 3].reset_index(drop=True)
del _input1['z_score_target']
plt.figure()
_input1['SalePrice'].hist(bins=20)
plt.title('SalePrice' + ' after transformation')
categorical_features = ['Alley', 'MSSubClass', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'GarageFinish', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
print(len(categorical_features))
nominal_features = ['BedroomAbvGr', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'ExterCond', 'ExterQual', 'Fireplaces', 'FireplaceQu', 'Functional', 'FullBath', 'GarageCars', 'GarageCond', 'GarageQual', 'HalfBath', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LandSlope', 'LotShape', 'PavedDrive', 'PoolQC', 'Street', 'Utilities', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd']
ordinal_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']
print('Nominal features length:\t', len(nominal_features), '\nOrdinal Features length:\t', len(ordinal_features))
numerical_features = nominal_features + ordinal_features
all_features = nominal_features + ordinal_features + categorical_features
_input1 = _input1[all_features + ['SalePrice']].copy()
_input0 = _input0[all_features].copy()
_input1.info()
_input1.describe().T
print(f'Null values: {_input1.isnull().sum()}')
nulls = pd.DataFrame(_input1.isna().sum().sort_values(ascending=False), columns=['null count'])
nulls
print(f'{_input1.duplicated().sum()} duplicates')
_input1.isna().sum()
_input0.isna().sum()
nulls = _input1.shape[0] - _input1.dropna(axis=0).shape[0]
nulls
nums = ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'LotFrontage', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF']
for feature in _input1.columns:
    if feature in nums:
        _input1[feature] = _input1[feature].fillna(0, inplace=False)
    elif feature in ['Alley', 'MasVnrType']:
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('None')
    elif feature in ['BsmtQual', 'MiscFeature', 'PoolQC', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('No')
    elif feature in ['CentralAir', 'PavedDrive']:
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('N')
    elif feature in ['Condition1', 'Condition2']:
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('Norm')
    elif feature in ['ExterCond', 'ExterQual', 'HeatingQC', 'KitchenQual']:
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('TA')
    elif feature in ['LotShape']:
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('Reg')
    elif feature == 'SaleCondition':
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('Normal')
    elif feature == 'Utilities':
        _input1.loc[:, feature] = _input1.loc[:, feature].fillna('AllPub')
for feature in _input0.columns:
    if feature in nums:
        _input0[feature] = _input0[feature].fillna(0, inplace=False)
    elif feature in ['Alley', 'MasVnrType']:
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('None')
    elif feature in ['BsmtQual', 'MiscFeature', 'PoolQC', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('No')
    elif feature in ['CentralAir', 'PavedDrive']:
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('N')
    elif feature in ['Condition1', 'Condition2']:
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('Norm')
    elif feature in ['ExterCond', 'ExterQual', 'HeatingQC', 'KitchenQual']:
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('TA')
    elif feature in ['LotShape']:
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('Reg')
    elif feature == 'SaleCondition':
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('Normal')
    elif feature == 'Utilities':
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('AllPub')
    elif feature == 'SaleType':
        _input0.loc[:, feature] = _input0.loc[:, feature].fillna('WD')
cor = _input1.corr()
_input1.corr().SalePrice.sort_values(ascending=False)
(fig, axs) = plt.subplots(1, 1, figsize=(75, 55))
sns.heatmap(cor, annot=True, fmt='.2f', cmap='coolwarm')
plt.figure(figsize=(10, 5))
sns.lineplot(data=_input1, x='OverallQual', y='SalePrice')
plt.figure(figsize=(25, 25))
for (i, feature) in enumerate(ordinal_features):
    plt.subplot(10, 3, i + 1)
    sns.scatterplot(data=_input1, x=feature, y='SalePrice', color='blue')
plt.tight_layout()
sns.scatterplot(data=_input1, x='GrLivArea', y='SalePrice')
_input1 = _input1[_input1['GrLivArea'] < 4000].reset_index(drop=True)
for feature in ordinal_features:
    if feature in ['YearBuilt', 'YearRemodAdd', 'YrSold']:
        continue
    if (_input1[feature] <= 0).sum() > 0:
        if _input1.loc[_input1[feature] > 0, feature].skew() < 0.5:
            continue
        _input1.loc[_input1[feature] > 0, feature] = np.log(_input1.loc[_input1[feature] > 0, feature])
        _input0.loc[_input0[feature] > 0, feature] = np.log(_input0.loc[_input0[feature] > 0, feature])
    else:
        if _input1[feature].skew() < 0.5:
            continue
        _input1[feature] = np.log(_input1[feature])
        _input0[feature] = np.log(_input0[feature])
from sklearn.preprocessing import RobustScaler
rob = RobustScaler()
_input1[ordinal_features] = rob.fit_transform(_input1[ordinal_features])
_input0[ordinal_features] = rob.fit_transform(_input0[ordinal_features])
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor as lgb
data_train = _input1.copy()
data_test = _input0.copy()
ohe = pd.get_dummies(data_train, columns=categorical_features)
ohe_test = pd.get_dummies(data_test, columns=categorical_features)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
ohe[ordinal_features] = oe.fit_transform(ohe[ordinal_features])
ohe[nominal_features] = oe.fit_transform(ohe[nominal_features])
ohe_test[ordinal_features] = oe.fit_transform(ohe_test[ordinal_features])
ohe_test[nominal_features] = oe.fit_transform(ohe_test[nominal_features])
total_df = pd.concat([ohe, ohe_test], ignore_index=True)
total_df = total_df.drop(columns=['MSSubClass_150'], inplace=False)
train_df = total_df[:len(ohe)]
test_df = total_df[len(ohe):]
train_df['SalePrice'] = _input1['SalePrice']
(train_df.shape, test_df.shape)
train_df.SalePrice.head()
sns.histplot(train_df['SalePrice'], kde=True)
x = train_df.drop(columns=['SalePrice'], axis=1)
y = train_df['SalePrice']
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1, random_state=42)
(x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
test_f = test_df.drop(columns=['SalePrice'])
test_f = test_f.fillna(0)
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

def regression_metrics(y_test, y_pred):
    print('explained_variance: ', round(explained_variance_score(y_test, y_pred), 4))
    print('r2: ', round(r2_score(y_test, y_pred), 4))
    print('MAE:\t', round(mean_absolute_error(y_test, y_pred), 4))
    print('MSE:\t', round(mean_squared_error(y_test, y_pred), 4))
    print('RMSE:\t', round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

def model_evaluate(model, param_grid, x_train, y_train, x_test, y_test, model_name, k_folds=5, scoring='neg_mean_squared_error', fit_parameters={}):
    model_cv = GridSearchCV(model, param_grid, cv=k_folds, verbose=False, scoring=scoring, refit=True)