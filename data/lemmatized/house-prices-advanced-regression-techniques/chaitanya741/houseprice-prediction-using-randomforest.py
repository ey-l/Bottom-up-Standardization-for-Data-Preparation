import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1.describe()
_input1.columns
feature = ['MSSubClass', 'MSZoning', 'LotArea', 'LotShape', 'Street', 'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'Neighborhood', 'BldgType', 'HouseStyle', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'YearBuilt', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterCond', 'BsmtFullBath', 'Foundation', 'BsmtCond', 'TotalBsmtSF', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtFinSF1', 'RoofMatl', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'BsmtFinSF2', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageType', 'GarageFinish', 'GarageArea', 'GarageCond', 'SaleType', 'OpenPorchSF', 'PoolArea', 'MoSold', 'YrSold', 'SalePrice']
data = _input1[feature]
data.isna().sum()
data = data.dropna(axis=0)
data
data.describe()
X = data.loc[:, 'MSSubClass':'YrSold']
y = data['SalePrice']
type(X.loc[0, 'MSSubClass'])
label_encoder = preprocessing.LabelEncoder()
for c in X.columns:
    if type(X.loc[0, c]) == str:
        X[c] = label_encoder.fit_transform(X[c])
    elif type(X.loc[0, c]) == int:
        X[c] = X[c]
(X_train, X_val, y_train, y_val) = train_test_split(X, y, random_state=1)
(X_train, x_val, y_train, y_val) = train_test_split(X, y, random_state=1)
X
model = RandomForestRegressor(random_state=1)