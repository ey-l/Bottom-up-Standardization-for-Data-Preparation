import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as ply
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1['SalePrice'].describe()
corr_matrix = _input1.corr(method='spearman')
corr_matrix['SalePrice'].sort_values(ascending=False)
housing_text_price = _input1[['MSZoning', 'Utilities', 'LandContour', 'LotConfig', 'Condition1', 'Condition2', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'Neighborhood', 'SalePrice']]
housing_text_price_test = _input0[['MSZoning', 'Utilities', 'LandContour', 'LotConfig', 'Condition1', 'Condition2', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'Neighborhood']]
housing_text_price.head()
imputer = SimpleImputer(strategy='median')
housing_num = _input1[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'TotRmsAbvGrd', 'Fireplaces', 'LotFrontage', 'LotArea', 'MasVnrArea', 'GarageYrBlt']]