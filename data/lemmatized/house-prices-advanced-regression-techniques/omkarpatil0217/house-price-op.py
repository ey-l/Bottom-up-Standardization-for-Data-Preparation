import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import math
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
print(_input1.shape)
print(_input0.shape)
comp_data = pd.concat((_input1, _input0))
comp_data_1 = comp_data
comp_data.shape
_input1.info()
plt.figure(figsize=(16, 9))
sns.heatmap(_input1.isnull())
missing = _input1.isnull().sum() / len(_input1)
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1 = _input1.drop(['Alley'], axis=1, inplace=False)
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(_input1['FireplaceQu'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1 = _input1.drop(['GarageYrBlt'], axis=1, inplace=False)
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1 = _input1.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input1.shape
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input1.isnull().sum()
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0])
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
plt.figure(figsize=(16, 9))
sns.heatmap(_input1.isnull())
_input1 = _input1.dropna(inplace=False)
sns.heatmap(_input1.isnull())
cat_val = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
len(cat_val)

def onehot_encod(multcols):
    train_1 = train_2
    i = 0
    for cols in multcols:
        print(cols)
        train_3 = pd.get_dummies(train_2[cols], drop_first=True)
        train_2 = train_2.drop([cols], axis=1, inplace=False)
        if i == 0:
            train_1 = train_3.copy()
        else:
            train_1 = pd.concat((train_1, train_3))
        i += 1
    train_1 = pd.concat((train_2, train_1))
    return train_1
copy_train = _input1.copy()
_input0.isnull().sum()
plt.figure(figsize=(16, 9))
sns.heatmap(_input0.isnull())
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean())
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0 = _input0.drop(['Alley'], axis=1, inplace=False)
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna(_input0['FireplaceQu'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0 = _input0.drop(['GarageYrBlt'], axis=1, inplace=False)
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0 = _input0.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
plt.figure(figsize=(16, 9))
sns.heatmap(_input0.isnull())
_input0.loc[:, _input0.isnull().any()].head()
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean())
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(_input0['BsmtFinSF2'].mean())
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].mean())
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].mean())
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mode()[0])
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].mode()[0])
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].mean())
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].mean())
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
_input0.shape
train_2 = pd.concat((_input1, _input0))
train_2['SalePrice']
train_2.shape
train_2 = onehot_encod(cat_val)
train_2.shape
train_2 = train_2.drop_duplicates(inplace=False)
train_2.shape
_input1 = train_2.iloc[:1422, :]
_input0 = train_2.iloc[1422:2881, :]
print(_input1.shape)
_input0.shape
_input0 = _input0.drop(['SalePrice'], axis=1, inplace=False)
X_train = _input1.drop(['SalePrice'], axis=1)
y_train = _input1['SalePrice']
_input0.tail()
import xgboost as xgb
regressor = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, early_stopping_rounds=10)