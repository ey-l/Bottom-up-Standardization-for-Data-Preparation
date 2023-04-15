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
housing = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
housing_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
housing.head()
housing_test.head()
housing['SalePrice'].describe()
corr_matrix = housing.corr(method='spearman')
corr_matrix['SalePrice'].sort_values(ascending=False)
housing_text_price = housing[['MSZoning', 'Utilities', 'LandContour', 'LotConfig', 'Condition1', 'Condition2', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'Neighborhood', 'SalePrice']]
housing_text_price_test = housing_test[['MSZoning', 'Utilities', 'LandContour', 'LotConfig', 'Condition1', 'Condition2', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'Neighborhood']]
housing_text_price.head()
imputer = SimpleImputer(strategy='median')
housing_num = housing[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'TotRmsAbvGrd', 'Fireplaces', 'LotFrontage', 'LotArea', 'MasVnrArea', 'GarageYrBlt']]