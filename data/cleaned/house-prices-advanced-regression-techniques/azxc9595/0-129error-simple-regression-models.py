import pandas as pd
import numpy as np
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
YTrain = train_data['SalePrice']
train_data
NANColumns = []
i = -1
for a in test_data.isnull().sum():
    i += 1
    if a != 0:
        print(test_data.columns[i], a)
        NANColumns.append(test_data.columns[i])
train_data['LotFrontage'] = train_data['LotFrontage'].replace(np.nan, np.mean(train_data['LotFrontage']))
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].replace(np.nan, np.mean(train_data['GarageYrBlt']))
train_data = train_data.drop(columns=['Alley', 'PoolQC', 'MasVnrType', 'Fence', 'MiscFeature', 'Id'])
train_data['MasVnrArea'] = train_data['MasVnrArea'].replace(np.nan, np.mean(train_data['MasVnrArea']))
train_data['BsmtQual'] = train_data['BsmtQual'].replace(np.nan, 'X')
train_data['BsmtCond'] = train_data['BsmtCond'].replace(np.nan, 'X')
train_data['BsmtExposure'] = train_data['BsmtExposure'].replace(np.nan, 'X')
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].replace(np.nan, 'X')
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].replace(np.nan, 'X')
train_data['Electrical'] = train_data['Electrical'].replace(np.nan, 'X')
train_data['FireplaceQu'] = train_data['FireplaceQu'].replace(np.nan, 'X')
train_data['GarageType'] = train_data['GarageType'].replace(np.nan, 'X')
train_data['GarageFinish'] = train_data['GarageFinish'].replace(np.nan, 'X')
train_data['GarageQual'] = train_data['GarageQual'].replace(np.nan, 'X')
train_data['GarageCond'] = train_data['GarageCond'].replace(np.nan, 'X')
test_data['LotFrontage'] = test_data['LotFrontage'].replace(np.nan, np.mean(test_data['LotFrontage']))
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].replace(np.nan, np.mean(test_data['GarageYrBlt']))
test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].replace(np.nan, np.mean(test_data['BsmtFinSF1']))
test_data['BsmtFinSF2'] = test_data['BsmtFinSF2'].replace(np.nan, np.mean(test_data['BsmtFinSF2']))
test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].replace(np.nan, np.mean(test_data['BsmtUnfSF']))
test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].replace(np.nan, np.mean(test_data['TotalBsmtSF']))
test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].replace(np.nan, np.mean(test_data['BsmtHalfBath']))
test_data['BsmtFullBath'] = test_data['BsmtFullBath'].replace(np.nan, np.mean(test_data['BsmtFullBath']))
test_data['GarageArea'] = test_data['GarageArea'].replace(np.nan, np.mean(test_data['GarageArea']))
test_data['GarageCars'] = test_data['GarageCars'].replace(np.nan, np.mean(test_data['GarageCars']))
test_data = test_data.drop(columns=['Alley', 'PoolQC', 'MasVnrType', 'Fence', 'MiscFeature', 'Id'])
test_data['MasVnrArea'] = test_data['MasVnrArea'].replace(np.nan, np.mean(test_data['MasVnrArea']))
test_data['BsmtQual'] = test_data['BsmtQual'].replace(np.nan, 'X')
test_data['BsmtCond'] = test_data['BsmtCond'].replace(np.nan, 'X')
test_data['BsmtExposure'] = test_data['BsmtExposure'].replace(np.nan, 'X')
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].replace(np.nan, 'X')
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].replace(np.nan, 'X')
test_data['Electrical'] = test_data['Electrical'].replace(np.nan, 'X')
test_data['FireplaceQu'] = test_data['FireplaceQu'].replace(np.nan, 'X')
test_data['GarageType'] = test_data['GarageType'].replace(np.nan, 'X')
test_data['GarageFinish'] = test_data['GarageFinish'].replace(np.nan, 'X')
test_data['GarageQual'] = test_data['GarageQual'].replace(np.nan, 'X')
test_data['GarageCond'] = test_data['GarageCond'].replace(np.nan, 'X')
test_data['SaleType'] = test_data['SaleType'].replace(np.nan, 'o')
test_data['Functional'] = test_data['Functional'].replace(np.nan, 'Typ')
test_data['KitchenQual'] = test_data['KitchenQual'].replace(np.nan, 'Gd')
test_data['MSZoning'] = test_data['MSZoning'].replace(np.nan, 'X')
test_data['Utilities'] = test_data['Utilities'].replace(np.nan, 'X')
test_data['Exterior1st'] = test_data['Exterior1st'].replace(np.nan, 'X')
test_data['Exterior2nd'] = test_data['Exterior2nd'].replace(np.nan, 'X')
test_data['GarageCond'] = test_data['GarageCond'].replace(np.nan, 'X')
Y_Btrain = YTrain
Btrain = train_data
Y_train = YTrain[0:1200]
Y_cross = YTrain[1200:1460]
train_data = train_data.drop('SalePrice', axis=1)
Btrain = Btrain.drop('SalePrice', axis=1)
Cross_data = train_data[1200:1460]
train_data = train_data[0:1200]
CATEGORICAL_COLUMNS = []
NUMERIC_COLUMNS = []
i = -1
for a in train_data.dtypes:
    i += 1
    if a == float or a == int:
        NUMERIC_COLUMNS.append(train_data.columns[i])
    elif a == object:
        CATEGORICAL_COLUMNS.append(train_data.columns[i])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Btrain[CATEGORICAL_COLUMNS] = Btrain[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
train_data[CATEGORICAL_COLUMNS] = train_data[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
Cross_data[CATEGORICAL_COLUMNS] = Cross_data[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
test_data[CATEGORICAL_COLUMNS] = test_data[CATEGORICAL_COLUMNS].apply(lambda col: le.fit_transform(col))
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
mlp = MLPRegressor(random_state=1, hidden_layer_sizes=(400, 1), max_iter=400)