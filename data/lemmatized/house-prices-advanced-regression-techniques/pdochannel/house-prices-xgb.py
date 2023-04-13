import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.describe().columns
import pandas as pd
from sklearn import preprocessing
numeric_csv = _input1[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]
nan_columns = np.any(pd.isna(numeric_csv), axis=0)
nan_columns = list(nan_columns[nan_columns == True].index)
nan_columns
numeric_csv['LotFrontage'] = numeric_csv['LotFrontage'].fillna(0)
numeric_csv['MasVnrArea'] = numeric_csv['MasVnrArea'].fillna(0)
numeric_csv['GarageYrBlt'] = numeric_csv['GarageYrBlt'].fillna(0)
x = numeric_csv.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
numeric_csv = pd.DataFrame(x_scaled, columns=['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'])
sale_price_scaler = preprocessing.MinMaxScaler()
targets = _input1['SalePrice'].values.reshape(-1, 1)
target_scaled = sale_price_scaler.fit_transform(targets)
categorical_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope']
categorical_data = _input1[categorical_cols]
categorical_data.head()
nan_columns = np.any(pd.isna(categorical_data), axis=0)
nan_columns = list(nan_columns[nan_columns == True].index)
nan_columns
for col in nan_columns:
    categorical_data[col] = categorical_data[col].fillna('N/A')
mapping_table = dict()
for col in categorical_cols:
    curr_mapping_table = dict()
    unique_values = pd.unique(categorical_data[col])
    for (inx, v) in enumerate(unique_values):
        curr_mapping_table[v] = inx + 1
        categorical_data[col] = categorical_data[col].replace(v, inx + 1)
    mapping_table[col] = curr_mapping_table
train_csv = pd.concat([numeric_csv, categorical_data], axis=1)
x_data = train_csv.values
from xgboost import XGBRFRegressor
reg = XGBRFRegressor()