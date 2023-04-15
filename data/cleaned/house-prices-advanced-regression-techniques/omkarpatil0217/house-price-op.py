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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
print(train.shape)
print(test.shape)
comp_data = pd.concat((train, test))
comp_data_1 = comp_data
comp_data.shape
train.info()
plt.figure(figsize=(16, 9))
sns.heatmap(train.isnull())
missing = train.isnull().sum() / len(train)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train.drop(['Alley'], axis=1, inplace=True)
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])
train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])
train.drop(['GarageYrBlt'], axis=1, inplace=True)
train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
train.shape
train.drop(['Id'], axis=1, inplace=True)
train.isnull().sum()
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
plt.figure(figsize=(16, 9))
sns.heatmap(train.isnull())
train.dropna(inplace=True)
sns.heatmap(train.isnull())
cat_val = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
len(cat_val)

def onehot_encod(multcols):
    train_1 = train_2
    i = 0
    for cols in multcols:
        print(cols)
        train_3 = pd.get_dummies(train_2[cols], drop_first=True)
        train_2.drop([cols], axis=1, inplace=True)
        if i == 0:
            train_1 = train_3.copy()
        else:
            train_1 = pd.concat((train_1, train_3))
        i += 1
    train_1 = pd.concat((train_2, train_1))
    return train_1
copy_train = train.copy()
test.isnull().sum()
plt.figure(figsize=(16, 9))
sns.heatmap(test.isnull())
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test.drop(['Alley'], axis=1, inplace=True)
test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test['FireplaceQu'] = test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])
test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])
test.drop(['GarageYrBlt'], axis=1, inplace=True)
test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])
test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])
test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])
test.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
test.drop(['Id'], axis=1, inplace=True)
plt.figure(figsize=(16, 9))
sns.heatmap(test.isnull())
test.loc[:, test.isnull().any()].head()
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
test.shape
train_2 = pd.concat((train, test))
train_2['SalePrice']
train_2.shape
train_2 = onehot_encod(cat_val)
train_2.shape
train_2.drop_duplicates(inplace=True)
train_2.shape
train = train_2.iloc[:1422, :]
test = train_2.iloc[1422:2881, :]
print(train.shape)
test.shape
test.drop(['SalePrice'], axis=1, inplace=True)
X_train = train.drop(['SalePrice'], axis=1)
y_train = train['SalePrice']
test.tail()
import xgboost as xgb
regressor = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, early_stopping_rounds=10)