import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test
(train.shape, test.shape)
train.info()
print()
print('------------------------------------------------------------')
print('---------------------- Training set ------------------------')
print('------------------------------------------------------------')
print()
count = 1
for i in train.columns:
    if train[i].isnull().sum() > 0:
        if train[i].dtypes == 'object':
            print(i + ' is a Categorical Variable')
            print('Total null values:', train[i].isnull().sum())
            print('Null values as a % of total:', round(train[i].isnull().sum() * 100 / train['SalePrice'].count(), 1))
            print('Categorical Variable No: ' + str(count))
            count = count + 1
            print()
print()
print('------------------------------------------------------------')
print('------------------------ -------- --------------------------')
print('------------------------------------------------------------')
print()
print()
print('------------------------------------------------------------')
print('------------------------ Test set --------------------------')
print('------------------------------------------------------------')
print()
testcount = 1
for i in test.columns:
    if test[i].isnull().sum() > 0:
        if test[i].dtypes == 'object':
            print(i + ' is a Categorical Variable')
            print('Total null values:', test[i].isnull().sum())
            print('Null values as a % of total:', round(test[i].isnull().sum() * 100 / train['SalePrice'].count(), 1))
            print('Categorical Variable No: ' + str(testcount))
            testcount = testcount + 1
            print()
print()
print('------------------------------------------------------------')
print('------------------------ -------- --------------------------')
print('------------------------------------------------------------')
print()
train = train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
test = test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
(train.shape, test.shape)
train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))
test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))
cats = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
for i in train[cats]:
    if train[i].dtype:
        train[i] = le.fit_transform(train[i].values)
        print(i + ' has been label encoded')
for i in test[cats]:
    if test[i].dtype:
        test[i] = le.fit_transform(test[i].values)
        print(i + ' has been label encoded')
(train.shape, test.shape)
y_train = train['SalePrice']
y_test = train['SalePrice']
features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'GarageQual', 'LandSlope', 'Neighborhood', 'SaleType', 'SaleCondition']
X_train = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
model = linear_model.Lasso(alpha=0.1)