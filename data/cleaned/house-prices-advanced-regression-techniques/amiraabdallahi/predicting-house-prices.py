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
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_data.head()
train_data = train_data.drop(['Id'], axis=1)
test_data = test_data.drop(['Id'], axis=1)
print('Number of rows:', train_data.shape[0])
print('Number of columns:', train_data.shape[1])
train_data.info()
train_data = train_data.select_dtypes(include=['int64', 'float64'])
test_data = test_data.select_dtypes(include=['int64', 'float64'])
train_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
test_data.fillna(test_data.median(), inplace=True)
test_data.isnull().sum()
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].median())
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].median())
sum(train_data.duplicated())
x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
skewed_features = [col for col in train_data.columns if train_data[col].skew() > 0.5]
print(len(skewed_features))
train_data[skewed_features] = train_data[skewed_features].apply(lambda x: np.log1p(x))
skewed_features.remove('SalePrice')
test_data[skewed_features] = test_data[skewed_features].apply(lambda x: np.log1p(x))
x = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
test_data = sc.transform(test_data)
lr = LinearRegression()
cv_score = cross_validate(lr, x, y, cv=10, scoring=['neg_root_mean_squared_error', 'neg_mean_squared_error'])
cv_score
cv_score['test_neg_root_mean_squared_error'].mean()
sns.distplot(y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()