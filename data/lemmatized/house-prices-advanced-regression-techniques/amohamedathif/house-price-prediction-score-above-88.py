import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
plt.pie(_input1.shape, labels=['rows', 'columns'], colors=['silver', 'orange'], pctdistance=0.7, autopct='%.1f%%', textprops={'weight': 'bold'}, wedgeprops={'edgecolor': 'black', 'linewidth': 2})
plt.title('Shape of dataframe')
_input1.describe()
_input1.dtypes
_input1.size
for columns in _input1:
    print(columns)
features = ['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
for i in features:
    print(_input1[i].isna().sum)
x = _input1[features]
y = _input1['SalePrice']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model = RandomForestRegressor(n_estimators=50)