import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.isnull().sum()
_input1.head()
_input1.shape
train_eda = _input1.drop(['Id', 'MSZoning', 'Utilities', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCond', 'PoolQC', 'Fence'], axis=1)
train_eda.shape
train_eda.head()
train_eda.isnull().sum()
train_eda = train_eda.drop('MiscFeature', axis=1)
train_eda.isnull().sum()
train_eda = train_eda.dropna()
train_eda.isnull().sum()
train_eda.shape
train_eda.info()
train_eda.loc[train_eda.Heating == 'GasA', 'Heating'] = 1
train_eda.loc[train_eda.Heating != 'GasA', 'Heating'] = 0
train_eda['Heating'] = train_eda['Heating'].astype(str).astype(float)
train_eda.loc[train_eda.CentralAir == 'Y', 'CentralAir'] = 1
train_eda.loc[train_eda.CentralAir == 'N', 'CentralAir'] = 0
train_eda['CentralAir'] = train_eda['CentralAir'].astype(str).astype(float)
train_eda.loc[train_eda.GarageQual == 'TA', 'GarageQual'] = 1
train_eda.loc[train_eda.GarageQual != 'TA', 'GarageQual'] = 0
train_eda['GarageQual'] = train_eda['GarageQual'].astype(str).astype(float)
train_eda.loc[train_eda.PavedDrive == 'Y', 'PavedDrive'] = 1
train_eda.loc[train_eda.PavedDrive != 'Y', 'PavedDrive'] = 0
train_eda['PavedDrive'] = train_eda['PavedDrive'].astype(str).astype(float)
train_eda.loc[train_eda.SaleType == 'WD', 'SaleType'] = 1
train_eda.loc[train_eda.SaleType != 'WD', 'SaleType'] = 0
train_eda['SaleType'] = train_eda['SaleType'].astype(str).astype(float)
train_eda.loc[train_eda.SaleCondition == 'Normal', 'SaleCondition'] = 1
train_eda.loc[train_eda.SaleCondition != 'Normal', 'SaleCondition'] = 0
train_eda['SaleCondition'] = train_eda['SaleCondition'].astype(str).astype(float)
train_eda.info()
x_test = _input0.drop(['Id', 'MSZoning', 'Utilities', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCond', 'PoolQC', 'Fence'], axis=1)
x_test.info()
x_test = x_test.drop('MiscFeature', axis=1)
x_test.loc[x_test.Heating == 'GasA', 'Heating'] = 1
x_test.loc[x_test.Heating != 'GasA', 'Heating'] = 0
x_test['Heating'] = x_test['Heating'].astype(str).astype(float)
x_test.loc[x_test.CentralAir == 'Y', 'CentralAir'] = 1
x_test.loc[x_test.CentralAir == 'N', 'CentralAir'] = 0
x_test['CentralAir'] = x_test['CentralAir'].astype(str).astype(float)
x_test.loc[x_test.GarageQual == 'TA', 'GarageQual'] = 1
x_test.loc[x_test.GarageQual != 'TA', 'GarageQual'] = 0
x_test['GarageQual'] = x_test['GarageQual'].astype(str).astype(float)
x_test.loc[x_test.PavedDrive == 'Y', 'PavedDrive'] = 1
x_test.loc[x_test.PavedDrive != 'Y', 'PavedDrive'] = 0
x_test['PavedDrive'] = x_test['PavedDrive'].astype(str).astype(float)
x_test.loc[x_test.SaleType == 'WD', 'SaleType'] = 1
x_test.loc[x_test.SaleType != 'WD', 'SaleType'] = 0
x_test['SaleType'] = x_test['SaleType'].astype(str).astype(float)
x_test.loc[x_test.SaleCondition == 'Normal', 'SaleCondition'] = 1
x_test.loc[x_test.SaleCondition != 'Normal', 'SaleCondition'] = 0
x_test['SaleCondition'] = x_test['SaleCondition'].astype(str).astype(float)
x_test.info()
x_test.isnull().sum()
x_test['LotFrontage'] = x_test['LotFrontage'].fillna(x_test['LotFrontage'].mean())
x_test['MasVnrArea'] = x_test['MasVnrArea'].fillna(x_test['MasVnrArea'].mean())
x_test['BsmtFinSF1'] = x_test['BsmtFinSF1'].fillna(x_test['BsmtFinSF1'].mean())
x_test['BsmtFinSF2'] = x_test['BsmtFinSF2'].fillna(x_test['BsmtFinSF2'].mean())
x_test['BsmtUnfSF'] = x_test['BsmtUnfSF'].fillna(x_test['BsmtUnfSF'].mean())
x_test['TotalBsmtSF'] = x_test['TotalBsmtSF'].fillna(x_test['TotalBsmtSF'].mean())
x_test['BsmtFullBath'] = x_test['BsmtFullBath'].fillna(x_test['BsmtFullBath'].mean())
x_test['BsmtHalfBath'] = x_test['BsmtHalfBath'].fillna(x_test['BsmtHalfBath'].mean())
x_test['GarageYrBlt'] = x_test['GarageYrBlt'].fillna(x_test['GarageYrBlt'].mean())
x_test['GarageCars'] = x_test['GarageCars'].fillna(x_test['GarageCars'].mean())
x_test['GarageArea'] = x_test['GarageArea'].fillna(x_test['GarageArea'].mean())
x_test.isnull().sum()
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
import math
lasso = Lasso()
x = train_eda.drop(['SalePrice'], axis=1)
y = train_eda['SalePrice']