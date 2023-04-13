import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.feature_selection import chi2
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import scipy.stats
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.info()
_input1.describe(exclude=['int', 'float'])
_input1.describe(exclude='object')
missings = _input1.isnull().sum() / len(_input1)
missings[missings > 0.5]
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input0 = _input0.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
numerical_features = [col for col in _input1.columns if _input1[col].dtype != 'object']
categorical_features = [col for col in _input1.columns if _input1[col].dtype == 'object']
from scipy.stats import shapiro

def check_normality(data):
    (stat, p) = shapiro(data)
    print('stat = %.2f, P-Value = %.2f' % (stat, p))
    if p > 0.05:
        print('Normal Distribution')
    else:
        print('Not Normal.')
check_normality(_input1['SalePrice'])
sns.distplot(_input1['SalePrice'])
sns.distplot(np.log1p(_input1['SalePrice']))
for col in _input1[numerical_features].columns:
    print(f'shapiro-wilk test for {col}')
    check_normality(_input1[col])
    print('=============================')
_input1[numerical_features].corr()['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(25, 25))
sns.heatmap(_input1[numerical_features].corr(), annot=True)
for val in ['Id', 'GarageYrBlt', 'GarageArea', '1stFlrSF']:
    numerical_features.remove(val)
y = _input1['SalePrice']
X = _input1[numerical_features].drop('SalePrice', axis=1)
X['OverallQual^2'] = X['OverallQual'] ** 2
X['OverallQual^3'] = X['OverallQual'] ** 3
X['OverallQual^1/2'] = np.sqrt(X['OverallQual'])
skewed_features = [col for col in X.columns if X[col].skew() > 0.5]
print(len(skewed_features))
y = y.apply(lambda x: np.log1p(x))
X[skewed_features] = X[skewed_features].apply(lambda x: np.log1p(x))
X = X.fillna(X.median(), inplace=False)
test_numerical = [col for col in _input0.columns if _input0[col].dtype != 'object' and col not in ['Id', 'GarageYrBlt', 'GarageArea', '1stFlrSF']]
X_test = _input0[test_numerical]
all_features = pd.concat([X, X_test])
scaler = StandardScaler()