import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df
df['GarageArea']
y = df.SalePrice
columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', 'GarageArea', 'PoolArea', 'YrSold']
df[columns].isnull().sum()
from sklearn.impute import SimpleImputer
inputer = SimpleImputer(missing_values=np.nan, strategy='mean')