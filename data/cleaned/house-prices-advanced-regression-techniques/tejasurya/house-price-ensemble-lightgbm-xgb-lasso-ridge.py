import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style='darkgrid', font_scale=1.6)
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('train shape  {}'.format(train.shape))
print('test shape  {}'.format(test.shape))


test_copy = test.copy()
plt.figure()
train['SalePrice'].hist(bins=20)
plt.title('SalePrice' + ' before transformation')

from scipy import stats
train['SalePrice'] = np.log(train['SalePrice'])
train['z_score_target'] = np.abs(stats.zscore(train['SalePrice']))
train = train.loc[train['z_score_target'] < 3].reset_index(drop=True)
del train['z_score_target']
plt.figure()
train['SalePrice'].hist(bins=20)
plt.title('SalePrice' + ' after transformation')

categorical_features = ['Alley', 'MSSubClass', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'GarageFinish', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
print(len(categorical_features))
nominal_features = ['BedroomAbvGr', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'ExterCond', 'ExterQual', 'Fireplaces', 'FireplaceQu', 'Functional', 'FullBath', 'GarageCars', 'GarageCond', 'GarageQual', 'HalfBath', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LandSlope', 'LotShape', 'PavedDrive', 'PoolQC', 'Street', 'Utilities', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd']
ordinal_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']
print('Nominal features length:\t', len(nominal_features), '\nOrdinal Features length:\t', len(ordinal_features))
numerical_features = nominal_features + ordinal_features
all_features = nominal_features + ordinal_features + categorical_features
train = train[all_features + ['SalePrice']].copy()
test = test[all_features].copy()

train.info()
train.describe().T
print(f'Null values: {train.isnull().sum()}')
nulls = pd.DataFrame(train.isna().sum().sort_values(ascending=False), columns=['null count'])
nulls
print(f'{train.duplicated().sum()} duplicates')
train.isna().sum()
test.isna().sum()
nulls = train.shape[0] - train.dropna(axis=0).shape[0]
nulls
nums = ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'LotFrontage', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF']
for feature in train.columns:
    if feature in nums:
        train[feature].fillna(0, inplace=True)
    elif feature in ['Alley', 'MasVnrType']:
        train.loc[:, feature] = train.loc[:, feature].fillna('None')
    elif feature in ['BsmtQual', 'MiscFeature', 'PoolQC', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        train.loc[:, feature] = train.loc[:, feature].fillna('No')
    elif feature in ['CentralAir', 'PavedDrive']:
        train.loc[:, feature] = train.loc[:, feature].fillna('N')
    elif feature in ['Condition1', 'Condition2']:
        train.loc[:, feature] = train.loc[:, feature].fillna('Norm')
    elif feature in ['ExterCond', 'ExterQual', 'HeatingQC', 'KitchenQual']:
        train.loc[:, feature] = train.loc[:, feature].fillna('TA')
    elif feature in ['LotShape']:
        train.loc[:, feature] = train.loc[:, feature].fillna('Reg')
    elif feature == 'SaleCondition':
        train.loc[:, feature] = train.loc[:, feature].fillna('Normal')
    elif feature == 'Utilities':
        train.loc[:, feature] = train.loc[:, feature].fillna('AllPub')
for feature in test.columns:
    if feature in nums:
        test[feature].fillna(0, inplace=True)
    elif feature in ['Alley', 'MasVnrType']:
        test.loc[:, feature] = test.loc[:, feature].fillna('None')
    elif feature in ['BsmtQual', 'MiscFeature', 'PoolQC', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        test.loc[:, feature] = test.loc[:, feature].fillna('No')
    elif feature in ['CentralAir', 'PavedDrive']:
        test.loc[:, feature] = test.loc[:, feature].fillna('N')
    elif feature in ['Condition1', 'Condition2']:
        test.loc[:, feature] = test.loc[:, feature].fillna('Norm')
    elif feature in ['ExterCond', 'ExterQual', 'HeatingQC', 'KitchenQual']:
        test.loc[:, feature] = test.loc[:, feature].fillna('TA')
    elif feature in ['LotShape']:
        test.loc[:, feature] = test.loc[:, feature].fillna('Reg')
    elif feature == 'SaleCondition':
        test.loc[:, feature] = test.loc[:, feature].fillna('Normal')
    elif feature == 'Utilities':
        test.loc[:, feature] = test.loc[:, feature].fillna('AllPub')
    elif feature == 'SaleType':
        test.loc[:, feature] = test.loc[:, feature].fillna('WD')
cor = train.corr()
train.corr().SalePrice.sort_values(ascending=False)
(fig, axs) = plt.subplots(1, 1, figsize=(75, 55))
sns.heatmap(cor, annot=True, fmt='.2f', cmap='coolwarm')
plt.figure(figsize=(10, 5))
sns.lineplot(data=train, x='OverallQual', y='SalePrice')

plt.figure(figsize=(25, 25))
for (i, feature) in enumerate(ordinal_features):
    plt.subplot(10, 3, i + 1)
    sns.scatterplot(data=train, x=feature, y='SalePrice', color='blue')
plt.tight_layout()

sns.scatterplot(data=train, x='GrLivArea', y='SalePrice')
train = train[train['GrLivArea'] < 4000].reset_index(drop=True)
for feature in ordinal_features:
    if feature in ['YearBuilt', 'YearRemodAdd', 'YrSold']:
        continue
    if (train[feature] <= 0).sum() > 0:
        if train.loc[train[feature] > 0, feature].skew() < 0.5:
            continue
        train.loc[train[feature] > 0, feature] = np.log(train.loc[train[feature] > 0, feature])
        test.loc[test[feature] > 0, feature] = np.log(test.loc[test[feature] > 0, feature])
    else:
        if train[feature].skew() < 0.5:
            continue
        train[feature] = np.log(train[feature])
        test[feature] = np.log(test[feature])
from sklearn.preprocessing import RobustScaler
rob = RobustScaler()
train[ordinal_features] = rob.fit_transform(train[ordinal_features])
test[ordinal_features] = rob.fit_transform(test[ordinal_features])
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor as lgb
data_train = train.copy()
data_test = test.copy()
ohe = pd.get_dummies(data_train, columns=categorical_features)
ohe_test = pd.get_dummies(data_test, columns=categorical_features)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
ohe[ordinal_features] = oe.fit_transform(ohe[ordinal_features])
ohe[nominal_features] = oe.fit_transform(ohe[nominal_features])
ohe_test[ordinal_features] = oe.fit_transform(ohe_test[ordinal_features])
ohe_test[nominal_features] = oe.fit_transform(ohe_test[nominal_features])
total_df = pd.concat([ohe, ohe_test], ignore_index=True)

total_df.drop(columns=['MSSubClass_150'], inplace=True)
train_df = total_df[:len(ohe)]

test_df = total_df[len(ohe):]

train_df['SalePrice'] = train['SalePrice']
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