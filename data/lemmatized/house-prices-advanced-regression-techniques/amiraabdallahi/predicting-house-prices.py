import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor, Lasso, ElasticNet
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1 = _input1.drop(['Id'], axis=1)
_input0 = _input0.drop(['Id'], axis=1)
print('Number of rows:', _input1.shape[0])
print('Number of columns:', _input1.shape[1])
_input1.info()
_input1 = _input1.select_dtypes(include=['int64', 'float64'])
_input0 = _input0.select_dtypes(include=['int64', 'float64'])
_input1.head()
_input1.isnull().sum()
_input0.isnull().sum()
_input0 = _input0.fillna(_input0.median(), inplace=False)
_input0.isnull().sum()
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].median())
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].median())
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].median())
sum(_input1.duplicated())
x = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
skewed_features = [col for col in _input1.columns if _input1[col].skew() > 0.5]
print(len(skewed_features))
_input1[skewed_features] = _input1[skewed_features].apply(lambda x: np.log1p(x))
skewed_features.remove('SalePrice')
_input0[skewed_features] = _input0[skewed_features].apply(lambda x: np.log1p(x))
x = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
_input0 = sc.transform(_input0)
lr = LinearRegression()
cv_score = cross_validate(lr, x, y, cv=10, scoring=['neg_root_mean_squared_error', 'neg_mean_squared_error'])
cv_score
cv_score['test_neg_root_mean_squared_error'].mean()
sns.distplot(y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()