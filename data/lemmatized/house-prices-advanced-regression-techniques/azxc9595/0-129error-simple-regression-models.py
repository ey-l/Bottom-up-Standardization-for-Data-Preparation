import pandas as pd
import numpy as np
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
YTrain = _input1['SalePrice']
_input1
NANColumns = []
i = -1
for a in _input0.isnull().sum():
    i += 1
    if a != 0:
        print(_input0.columns[i], a)
        NANColumns.append(_input0.columns[i])
_input1['LotFrontage'] = _input1['LotFrontage'].replace(np.nan, np.mean(_input1['LotFrontage']))
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].replace(np.nan, np.mean(_input1['GarageYrBlt']))
_input1 = _input1.drop(columns=['Alley', 'PoolQC', 'MasVnrType', 'Fence', 'MiscFeature', 'Id'])
_input1['MasVnrArea'] = _input1['MasVnrArea'].replace(np.nan, np.mean(_input1['MasVnrArea']))
_input1['BsmtQual'] = _input1['BsmtQual'].replace(np.nan, 'X')
_input1['BsmtCond'] = _input1['BsmtCond'].replace(np.nan, 'X')
_input1['BsmtExposure'] = _input1['BsmtExposure'].replace(np.nan, 'X')
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].replace(np.nan, 'X')
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].replace(np.nan, 'X')
_input1['Electrical'] = _input1['Electrical'].replace(np.nan, 'X')
_input1['FireplaceQu'] = _input1['FireplaceQu'].replace(np.nan, 'X')
_input1['GarageType'] = _input1['GarageType'].replace(np.nan, 'X')
_input1['GarageFinish'] = _input1['GarageFinish'].replace(np.nan, 'X')
_input1['GarageQual'] = _input1['GarageQual'].replace(np.nan, 'X')
_input1['GarageCond'] = _input1['GarageCond'].replace(np.nan, 'X')
_input0['LotFrontage'] = _input0['LotFrontage'].replace(np.nan, np.mean(_input0['LotFrontage']))
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].replace(np.nan, np.mean(_input0['GarageYrBlt']))
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].replace(np.nan, np.mean(_input0['BsmtFinSF1']))
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].replace(np.nan, np.mean(_input0['BsmtFinSF2']))
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].replace(np.nan, np.mean(_input0['BsmtUnfSF']))
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].replace(np.nan, np.mean(_input0['TotalBsmtSF']))
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].replace(np.nan, np.mean(_input0['BsmtHalfBath']))
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].replace(np.nan, np.mean(_input0['BsmtFullBath']))
_input0['GarageArea'] = _input0['GarageArea'].replace(np.nan, np.mean(_input0['GarageArea']))
_input0['GarageCars'] = _input0['GarageCars'].replace(np.nan, np.mean(_input0['GarageCars']))
_input0 = _input0.drop(columns=['Alley', 'PoolQC', 'MasVnrType', 'Fence', 'MiscFeature', 'Id'])
_input0['MasVnrArea'] = _input0['MasVnrArea'].replace(np.nan, np.mean(_input0['MasVnrArea']))
_input0['BsmtQual'] = _input0['BsmtQual'].replace(np.nan, 'X')
_input0['BsmtCond'] = _input0['BsmtCond'].replace(np.nan, 'X')
_input0['BsmtExposure'] = _input0['BsmtExposure'].replace(np.nan, 'X')
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].replace(np.nan, 'X')
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].replace(np.nan, 'X')
_input0['Electrical'] = _input0['Electrical'].replace(np.nan, 'X')
_input0['FireplaceQu'] = _input0['FireplaceQu'].replace(np.nan, 'X')
_input0['GarageType'] = _input0['GarageType'].replace(np.nan, 'X')
_input0['GarageFinish'] = _input0['GarageFinish'].replace(np.nan, 'X')
_input0['GarageQual'] = _input0['GarageQual'].replace(np.nan, 'X')
_input0['GarageCond'] = _input0['GarageCond'].replace(np.nan, 'X')
_input0['SaleType'] = _input0['SaleType'].replace(np.nan, 'o')
_input0['Functional'] = _input0['Functional'].replace(np.nan, 'Typ')
_input0['KitchenQual'] = _input0['KitchenQual'].replace(np.nan, 'Gd')
_input0['MSZoning'] = _input0['MSZoning'].replace(np.nan, 'X')
_input0['Utilities'] = _input0['Utilities'].replace(np.nan, 'X')
_input0['Exterior1st'] = _input0['Exterior1st'].replace(np.nan, 'X')
_input0['Exterior2nd'] = _input0['Exterior2nd'].replace(np.nan, 'X')
_input0['GarageCond'] = _input0['GarageCond'].replace(np.nan, 'X')
Y_Btrain = YTrain
Btrain = _input1
Y_train = YTrain[0:1200]
Y_cross = YTrain[1200:1460]
_input1 = _input1.drop('SalePrice', axis=1)
Btrain = Btrain.drop('SalePrice', axis=1)
Cross_data = _input1[1200:1460]
_input1 = _input1[0:1200]
CATEGORICAL_COLUMNS = []
NUMERIC_COLUMNS = []
i = -1
for a in _input1.dtypes:
    i += 1
    if a == float or a == int:
        NUMERIC_COLUMNS.append(_input1.columns[i])
    elif a == object:
        CATEGORICAL_COLUMNS.append(_input1.columns[i])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Btrain[CATEGORICAL_COLUMNS] = Btrain[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
_input1[CATEGORICAL_COLUMNS] = _input1[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
Cross_data[CATEGORICAL_COLUMNS] = Cross_data[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
_input0[CATEGORICAL_COLUMNS] = _input0[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
mlp = MLPRegressor(random_state=1, hidden_layer_sizes=(400, 1), max_iter=400)