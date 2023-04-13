import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
y = _input1.SalePrice
columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', 'GarageArea', 'PoolArea', 'YrSold']
_input1[columns].isnull().sum()
from sklearn.impute import SimpleImputer
inputer = SimpleImputer(missing_values=np.nan, strategy='constant')