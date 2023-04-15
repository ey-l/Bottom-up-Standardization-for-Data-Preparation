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
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data['CentralAir'] = data['CentralAir'].replace('Y', 1)
data['CentralAir'] = data['CentralAir'].replace('N', 2)
data['KitchenQual'] = data['KitchenQual'].replace('Gd', 1)
data['KitchenQual'] = data['KitchenQual'].replace('TA', 2)
data['KitchenQual'] = data['KitchenQual'].replace('Ex', 3)
data['KitchenQual'] = data['KitchenQual'].replace('Fa', 4)
data['GarageType'] = data['GarageType'].replace('Attchd', 1)
data['GarageType'] = data['GarageType'].replace('Detchd', 2)
data['GarageType'] = data['GarageType'].replace('BuiltIn', 3)
data['GarageType'] = data['GarageType'].replace('CarPort', 4)
data['GarageType'] = data['GarageType'].replace('NA', 5)
data['GarageType'] = data['GarageType'].replace('Basment', 6)
data['GarageType'] = data['GarageType'].replace('2Types', 7)
data['BsmtCond'] = data['BsmtCond'].replace('TA', 1)
data['BsmtCond'] = data['BsmtCond'].replace('Gd', 2)
data['BsmtCond'] = data['BsmtCond'].replace('NA', 3)
data['BsmtCond'] = data['BsmtCond'].replace('Fa', 4)
data['BsmtCond'] = data['BsmtCond'].replace('Po', 5)
data['Electrical'] = data['Electrical'].replace('SBrkr', 1)
data['Electrical'] = data['Electrical'].replace('FuseA', 2)
data['Electrical'] = data['Electrical'].replace('FuseF', 3)
data['Electrical'] = data['Electrical'].replace('FuseP', 4)
data['Electrical'] = data['Electrical'].replace('SBrkr', 5)
data['Electrical'] = data['Electrical'].replace('Mix', 6)
data['Electrical'] = data['Electrical'].replace('NA', 7)
X = data[['GrLivArea', 'GarageArea', 'TotalBsmtSF', 'OverallQual', 'OverallCond', 'LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'CentralAir']]
y = data['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)