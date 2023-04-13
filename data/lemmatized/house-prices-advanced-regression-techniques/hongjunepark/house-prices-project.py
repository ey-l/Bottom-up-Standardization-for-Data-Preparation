import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1.info()
_input1.describe().T
from sklearn.model_selection import train_test_split
y_1 = _input1['SalePrice']
train_1 = _input1.copy()
train_1 = train_1.drop(['SalePrice'], axis=1, inplace=False)
(X_train_1, X_valid_1, y_train_1, y_valid_1) = train_test_split(train_1, y_1, train_size=0.8, test_size=0.2, random_state=0)
print('\nX_train_1 shape:', X_train_1.shape)
print('\nX_valid_1 shape:', X_valid_1.shape)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def rmse_score(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)