import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
import pandas as pd
pd.set_option('display.max_rows', None)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1['CentralAir'] = _input1['CentralAir'].replace('Y', 1)
_input1['CentralAir'] = _input1['CentralAir'].replace('N', 2)
_input1['KitchenQual'] = _input1['KitchenQual'].replace('Gd', 1)
_input1['KitchenQual'] = _input1['KitchenQual'].replace('TA', 2)
_input1['KitchenQual'] = _input1['KitchenQual'].replace('Ex', 3)
_input1['KitchenQual'] = _input1['KitchenQual'].replace('Fa', 4)
_input1['GarageType'] = _input1['GarageType'].replace('Attchd', 1)
_input1['GarageType'] = _input1['GarageType'].replace('Detchd', 2)
_input1['GarageType'] = _input1['GarageType'].replace('BuiltIn', 3)
_input1['GarageType'] = _input1['GarageType'].replace('CarPort', 4)
_input1['GarageType'] = _input1['GarageType'].replace('NA', 5)
_input1['GarageType'] = _input1['GarageType'].replace('Basment', 6)
_input1['GarageType'] = _input1['GarageType'].replace('2Types', 7)
_input1['BsmtCond'] = _input1['BsmtCond'].replace('TA', 1)
_input1['BsmtCond'] = _input1['BsmtCond'].replace('Gd', 2)
_input1['BsmtCond'] = _input1['BsmtCond'].replace('NA', 3)
_input1['BsmtCond'] = _input1['BsmtCond'].replace('Fa', 4)
_input1['BsmtCond'] = _input1['BsmtCond'].replace('Po', 5)
_input1['Electrical'] = _input1['Electrical'].replace('SBrkr', 1)
_input1['Electrical'] = _input1['Electrical'].replace('FuseA', 2)
_input1['Electrical'] = _input1['Electrical'].replace('FuseF', 3)
_input1['Electrical'] = _input1['Electrical'].replace('FuseP', 4)
_input1['Electrical'] = _input1['Electrical'].replace('SBrkr', 5)
_input1['Electrical'] = _input1['Electrical'].replace('Mix', 6)
_input1['Electrical'] = _input1['Electrical'].replace('NA', 7)
X = _input1[['GrLivArea', 'GarageArea', 'TotalBsmtSF', 'OverallQual', 'OverallCond', 'LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'CentralAir']]
y = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)