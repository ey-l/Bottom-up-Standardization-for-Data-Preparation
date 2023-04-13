import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import itertools
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = [_input1, _input0]
_input1.shape
_input0.shape
print(_input1.columns.values)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
_input1.isnull().sum()
_input0.isnull().sum()
_input1.dtypes.value_counts()
_input0.dtypes.value_counts()
_input1.describe()
_input1.describe(include=['O'])
cat_futures = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
for each in cat_futures:
    print(tabulate(_input1[[each, 'SalePrice']].groupby([each], as_index=False).mean().sort_values(by='SalePrice', ascending=False), headers='keys', tablefmt='fancy_grid', showindex=False))
cat = _input1.select_dtypes('object')
num = _input1.select_dtypes('number')
num = num.drop(['Id', 'SalePrice'], axis=1)
color_cycle = itertools.cycle(['orange', 'pink', 'blue', 'brown', 'red', 'grey', 'yellow', 'green'])
for i in num:
    plt.scatter(_input1[i], _input1['SalePrice'], color=next(color_cycle))
    plt.title(i)
    plt.xlabel(i)
    plt.ylabel('SalePrice')
for i in cat:
    plt.figure(figsize=(20, 4))
    plt.subplots_adjust(hspace=0.25)
    plt.subplot(1, 2, 1)
    plt.title('How price is different')
    sns.barplot(x=i, y='SalePrice', data=_input1)
    plt.subplot(1, 2, 2)
    plt.title('Home much value counts')
    sns.stripplot(x=i, y='SalePrice', data=_input1)
for dataset in combine:
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(_input1['LotFrontage'].median())
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(_input1['LotFrontage'].median())
    dataset['Alley'] = dataset['Alley'].fillna('None')
    dataset['BsmtQual'] = dataset['BsmtQual'].fillna('NoBsmt')
    dataset['BsmtCond'] = dataset['BsmtCond'].fillna('NoBsmt')
    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna('NoBsmt')
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna('NoBsmt')
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna('NoBsmt')
    dataset['Electrical'] = dataset['Electrical'].fillna(_input1.Electrical.dropna().mode()[0])
    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('Gd')
    dataset['GarageType'] = dataset['GarageType'].fillna('NoGarage')
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
    dataset['GarageFinish'] = dataset['GarageFinish'].fillna('NoGarage')
    dataset['GarageQual'] = dataset['GarageQual'].fillna('NoGarage')
    dataset['GarageCond'] = dataset['GarageCond'].fillna('NoGarage')
    dataset['PoolQC'] = dataset['PoolQC'].fillna('Normal')
    dataset['Fence'] = dataset['Fence'].fillna('NoFence')
    dataset['MiscFeature'] = dataset['MiscFeature'].fillna('None')
    dataset['MSZoning'] = dataset['MSZoning'].fillna('RL')
    dataset['Utilities'] = dataset['Utilities'].fillna('NoSeWa')
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(_input1.Exterior1st.dropna().mode()[0])
    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna('Other')
    dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(0)
    dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(0)
    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(0)
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(0)
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0)
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna('Gd')
    dataset['Functional'] = dataset['Functional'].fillna('Typ')
    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
    dataset['GarageArea'] = dataset['GarageArea'].fillna(0)
    dataset['SaleType'] = dataset['SaleType'].fillna('Oth')
combine = [_input1, _input0]
_input1.isnull().sum().value_counts()
_input0.isnull().sum().value_counts()
encoder = LabelEncoder()
for i in cat:
    for dataset in combine:
        dataset[i] = encoder.fit_transform(dataset[i])
combine = [_input1, _input0]
_input1.head()
_input0.head()
train_df_copy = _input1.copy()
test_df_copy = _input0.copy()
_input1 = _input1.drop(['Id', 'SalePrice'], axis=1)
_input0 = _input0.drop('Id', axis=1)
combine = [_input1, _input0]
scaler = MinMaxScaler()
columns = _input1.columns.values
for dataset in combine:
    dataset_scaled = scaler.fit_transform(dataset[columns])
    dataset[columns] = dataset_scaled
combine = [_input1, _input0]
_input1.head()
X_train = _input1
y_train = train_df_copy['SalePrice']
X_test = _input0
(X_train.shape, y_train.shape, X_test.shape)
rndfreg = RandomForestRegressor()