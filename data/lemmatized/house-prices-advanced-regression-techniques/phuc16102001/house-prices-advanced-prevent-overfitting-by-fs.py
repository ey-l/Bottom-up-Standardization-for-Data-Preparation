import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.shape
_input1.info()
_input1.describe().T
print('Missing value by column')
print('-' * 20)
print(_input1.isna().sum())
print('-' * 20)
print('Total:', _input1.isna().sum().sum())
_input1 = _input1.dropna(axis=1)
print('Missing value by column')
print('-' * 20)
print(_input1.isna().sum())
print('-' * 20)
print('Total:', _input1.isna().sum().sum())
X = _input1.drop(['Id', 'SalePrice'], axis=1)
y = _input1['SalePrice']
col_types = X.dtypes
numeric_col = col_types[col_types != 'object'].index
scaler = StandardScaler()
X[numeric_col] = scaler.fit_transform(X[numeric_col])
X.head()
X = pd.get_dummies(X)
X.head()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
model1 = LinearRegression()