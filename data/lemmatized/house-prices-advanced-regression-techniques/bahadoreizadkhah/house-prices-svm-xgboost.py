import xgboost as xgb
from sklearn.svm import SVR
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head(10)
_input1.hist(bins=50, figsize=(16, 16))
(fig, ax) = plt.subplots(figsize=(10, 6))
ax.grid()
ax.scatter(_input1['GrLivArea'], _input1['SalePrice'], c='#3f72af', zorder=3, alpha=0.9)
ax.axvline(4500, c='#112d4e', ls='--', zorder=2)
ax.set_xlabel('Ground living area (sq. ft)', labelpad=10)
ax.set_ylabel('Sale price ($)', labelpad=10)
_input1 = _input1[_input1.GrLivArea < 4500]
total = _input0.isna().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total'])
_input1 = _input1.drop(missing_data[missing_data.Total > 0].index, axis=1)
_input0 = _input0.dropna(axis=1)
_input0 = _input0.drop(['Electrical'], axis=1)
dataset = pd.concat([_input1, _input0])
dataset = pd.get_dummies(dataset)
dataset.head(10)
dataset = dataset.dropna()
X = dataset.iloc[:, 0:1].values
print(X)
y = dataset.SalePrice
y = y.values.reshape(len(y), 1)
y