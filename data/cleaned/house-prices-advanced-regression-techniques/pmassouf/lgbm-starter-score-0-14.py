import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df
df.set_index('Id')
df.dtypes
df[df.select_dtypes('object').columns]
df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(lambda col: col.astype('category'))
df.dtypes
df.dtypes
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
from lightgbm import LGBMRegressor
from lightgbm import cv
from lightgbm import Dataset
from sklearn.metrics import mean_squared_error

def log_rmse(preds, train_data):
    y_true = train_data.get_label()
    y_hat = np.round(preds)
    return ('l_rmse', mean_squared_error(np.log(y_true), np.log(y_hat), squared=False), True)
lgbm = LGBMRegressor(objective='regression', random_state=0)