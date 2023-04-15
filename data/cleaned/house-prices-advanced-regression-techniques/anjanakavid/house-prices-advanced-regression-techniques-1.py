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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = [train_df, test_df]
train_df.shape
test_df.shape
print(train_df.columns.values)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
train_df.isnull().sum()
test_df.isnull().sum()
train_df.dtypes.value_counts()
test_df.dtypes.value_counts()
train_df.describe()
train_df.describe(include=['O'])
cat_futures = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
for each in cat_futures:
    print(tabulate(train_df[[each, 'SalePrice']].groupby([each], as_index=False).mean().sort_values(by='SalePrice', ascending=False), headers='keys', tablefmt='fancy_grid', showindex=False))
cat = train_df.select_dtypes('object')
num = train_df.select_dtypes('number')
num = num.drop(['Id', 'SalePrice'], axis=1)
color_cycle = itertools.cycle(['orange', 'pink', 'blue', 'brown', 'red', 'grey', 'yellow', 'green'])
for i in num:
    plt.scatter(train_df[i], train_df['SalePrice'], color=next(color_cycle))
    plt.title(i)
    plt.xlabel(i)
    plt.ylabel('SalePrice')

for i in cat:
    plt.figure(figsize=(20, 4))
    plt.subplots_adjust(hspace=0.25)
    plt.subplot(1, 2, 1)
    plt.title('How price is different')
    sns.barplot(x=i, y='SalePrice', data=train_df)
    plt.subplot(1, 2, 2)
    plt.title('Home much value counts')
    sns.stripplot(x=i, y='SalePrice', data=train_df)
for dataset in combine:
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(train_df['LotFrontage'].median())
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(train_df['LotFrontage'].median())
    dataset['Alley'] = dataset['Alley'].fillna('None')
    dataset['BsmtQual'] = dataset['BsmtQual'].fillna('NoBsmt')
    dataset['BsmtCond'] = dataset['BsmtCond'].fillna('NoBsmt')
    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna('NoBsmt')
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna('NoBsmt')
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna('NoBsmt')
    dataset['Electrical'] = dataset['Electrical'].fillna(train_df.Electrical.dropna().mode()[0])
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
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(train_df.Exterior1st.dropna().mode()[0])
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
combine = [train_df, test_df]
train_df.isnull().sum().value_counts()
test_df.isnull().sum().value_counts()
encoder = LabelEncoder()
for i in cat:
    for dataset in combine:
        dataset[i] = encoder.fit_transform(dataset[i])
combine = [train_df, test_df]
train_df.head()
test_df.head()
train_df_copy = train_df.copy()
test_df_copy = test_df.copy()
train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
test_df = test_df.drop('Id', axis=1)
combine = [train_df, test_df]
scaler = MinMaxScaler()
columns = train_df.columns.values
for dataset in combine:
    dataset_scaled = scaler.fit_transform(dataset[columns])
    dataset[columns] = dataset_scaled
combine = [train_df, test_df]
train_df.head()
X_train = train_df
y_train = train_df_copy['SalePrice']
X_test = test_df
(X_train.shape, y_train.shape, X_test.shape)
rndfreg = RandomForestRegressor()