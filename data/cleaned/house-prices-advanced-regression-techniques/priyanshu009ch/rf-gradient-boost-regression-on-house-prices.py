import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
dataset.head()
for col in dataset.columns:
    if dataset[col].count() < 1460:
        print(col)
m = dataset['LotFrontage'].mean()
dataset['LotFrontage'].fillna(value=m, inplace=True)
dataset['Alley'].fillna(value='None', inplace=True)
dataset['MasVnrType'].fillna(value='None', inplace=True)
m = dataset['MasVnrArea'].mean()
dataset['MasVnrArea'].fillna(value=m, inplace=True)
dataset['BsmtQual'].fillna(value='None', inplace=True)
dataset['BsmtCond'].fillna(value='None', inplace=True)
dataset['BsmtExposure'].fillna(value='None', inplace=True)
dataset['BsmtFinType1'].fillna(value='None', inplace=True)
dataset['BsmtFinType2'].fillna(value='None', inplace=True)
dataset['Electrical'].fillna(value='None', inplace=True)
dataset['FireplaceQu'].fillna(value='None', inplace=True)
dataset['GarageType'].fillna(value='None', inplace=True)
m = dataset['GarageYrBlt'].mean()
dataset['GarageYrBlt'].fillna(value=m, inplace=True)
dataset['GarageFinish'].fillna(value='None', inplace=True)
dataset['GarageQual'].fillna(value='None', inplace=True)
dataset['GarageCond'].fillna(value='None', inplace=True)
dataset['PoolQC'].fillna(value='None', inplace=True)
dataset['Fence'].fillna(value='None', inplace=True)
dataset['MiscFeature'].fillna(value='None', inplace=True)
for col in dataset.columns:
    if dataset[col].count() < 1460:
        print(col)
for col in dataset.columns:
    if type(dataset[col][0]) == str:
        dataset[col] = dataset[col].astype('category')
        dataset[col] = dataset[col].cat.codes
dataset.head()
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print(len(x), len(x_train), len(x_test))
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)