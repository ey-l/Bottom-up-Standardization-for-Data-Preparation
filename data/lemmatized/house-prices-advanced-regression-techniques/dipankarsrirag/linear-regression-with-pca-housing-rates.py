import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
le = LabelEncoder()
pca = PCA(n_components=0.1, svd_solver='full')
sim = SimpleImputer(strategy='most_frequent')

def PCA(data, name):
    pc = pca.fit_transform(_input1)
    col_n = pc.shape[1]
    columns = []
    for i in range(1, col_n + 1):
        columns.append(name + str(i))
    return pd.DataFrame(pc, columns=columns)

def preProcess(data, train=True):
    for i in _input1.columns:
        _input1[i] = sim.fit_transform(np.array(_input1[i]).reshape(_input1[i].shape[0], -1))
    for i in _input1.columns:
        if type(_input1[i].iloc[2]) == str:
            _input1[i] = le.fit_transform(_input1[i])
    drop = ['Id', 'MiscVal']
    if train:
        drop += ['SalePrice']
        target = _input1.SalePrice + _input1.MiscVal
    _input1 = _input1.drop(drop, axis=1, inplace=False)
    placement = ['MSZoning', 'Street', 'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle']
    data_plac = _input1[placement]
    lot = ['LotFrontage', 'LotArea', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope']
    data_lot = _input1[lot]
    exterior = ['RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond']
    data_ext = _input1[exterior]
    basement = ['Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    data_base = _input1[basement]
    floor = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea']
    data_floor = _input1[floor]
    bhk = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd']
    data_bhk = _input1[bhk]
    func = ['Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'Functional']
    data_func = _input1[func]
    bells = ['Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature']
    data_bells = _input1[bells]
    sale = ['YearBuilt', 'YearRemodAdd', 'YrSold', 'SaleType', 'SaleCondition']
    data_sale = _input1[sale]
    data_plac_red = PCA(data_plac, 'plac')
    data_lot_red = PCA(data_lot, 'lot')
    data_ext_red = PCA(data_ext, 'ext')
    data_base_red = PCA(data_base, 'base')
    data_floor_red = PCA(data_floor, 'floor')
    data_bhk_red = PCA(data_bhk, 'bhk')
    data_func_red = PCA(data_func, 'func')
    data_bells_red = PCA(data_bells, 'bells')
    data_sale_red = PCA(data_sale, 'sale')
    data_red = PCA(pd.concat([data_plac_red, data_lot_red, data_ext_red, data_base_red, data_floor_red, data_bhk_red, data_func_red, data_bells_red, data_sale_red], axis=1), 'red')
    X = np.array(data_red.red1)
    X = X.reshape(X.shape[0], -1)
    mms = MinMaxScaler()
    X = mms.fit_transform(X)
    if train:
        y = np.array(target)
        y = y.reshape(y.shape[0], -1)
        return (data_red, target)
    return X
(X, y) = preProcess(_input1)
(train_x, test_x, train_y, test_y) = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
estimator = LinearRegression(n_jobs=-1)