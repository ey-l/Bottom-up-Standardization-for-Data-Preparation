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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.info()
train_df.describe(exclude=['int', 'float'])
train_df.describe(exclude='object')
missings = train_df.isnull().sum() / len(train_df)
missings[missings > 0.5]
train_df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
test_df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
numerical_features = [col for col in train_df.columns if train_df[col].dtype != 'object']
categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'object']
from scipy.stats import shapiro

def check_normality(data):
    (stat, p) = shapiro(data)
    print('stat = %.2f, P-Value = %.2f' % (stat, p))
    if p > 0.05:
        print('Normal Distribution')
    else:
        print('Not Normal.')
check_normality(train_df['SalePrice'])
sns.distplot(train_df['SalePrice'])

sns.distplot(np.log1p(train_df['SalePrice']))

for col in train_df[numerical_features].columns:
    print(f'shapiro-wilk test for {col}')
    check_normality(train_df[col])
    print('=============================')
train_df[numerical_features].corr()['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(25, 25))
sns.heatmap(train_df[numerical_features].corr(), annot=True)

for val in ['Id', 'GarageYrBlt', 'GarageArea', '1stFlrSF']:
    numerical_features.remove(val)
y = train_df['SalePrice']
X = train_df[numerical_features].drop('SalePrice', axis=1)
X['OverallQual^2'] = X['OverallQual'] ** 2
X['OverallQual^3'] = X['OverallQual'] ** 3
X['OverallQual^1/2'] = np.sqrt(X['OverallQual'])
skewed_features = [col for col in X.columns if X[col].skew() > 0.5]
print(len(skewed_features))
y = y.apply(lambda x: np.log1p(x))
X[skewed_features] = X[skewed_features].apply(lambda x: np.log1p(x))
X.fillna(X.median(), inplace=True)
test_numerical = [col for col in test_df.columns if test_df[col].dtype != 'object' and col not in ['Id', 'GarageYrBlt', 'GarageArea', '1stFlrSF']]
X_test = test_df[test_numerical]
all_features = pd.concat([X, X_test])
scaler = StandardScaler()