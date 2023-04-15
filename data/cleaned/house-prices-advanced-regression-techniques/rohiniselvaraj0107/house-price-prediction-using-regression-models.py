import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.describe()
test.describe()
train.shape
test.shape
train_missing = []
for col in train.columns:
    if train[col].isna().sum() != 0:
        missing = train[col].isna().sum()
        print(f'{col: <{20}} {missing}')
        if missing > train.shape[0] / 3:
            train_missing.append(col)
test_missing = []
for col in test.columns:
    if test[col].isna().sum() != 0:
        missing = test[col].isna().sum()
        print(f'{col: <{20}} {missing}')
        if missing > test.shape[0] / 3:
            test_missing.append(col)
cols = train.columns
no_of_cols = len(cols)
print('Number of columns: {}'.format(no_of_cols))
numeric_cols = [col for col in train.columns if train[col].dtype in ('int32', 'int64', 'float64')]
print('Numeric columns: {}'.format(numeric_cols))
print('No of numeric columns: {}'.format(len(numeric_cols)))
categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
print('Categorical columns: {}'.format(categorical_cols))
print('No of categorical columns: {}'.format(len(categorical_cols)))
cols = test.columns
no_of_cols = len(cols)
print('Number of columns: {}'.format(no_of_cols))
numeric_cols = [col for col in test.columns if test[col].dtype in ('int32', 'int64', 'float64')]
print('Numeric columns: {}'.format(numeric_cols))
print('No of numeric columns: {}'.format(len(numeric_cols)))
categorical_cols = [col for col in test.columns if test[col].dtype == 'object']
print('Categorical columns: {}'.format(categorical_cols))
print('No of categorical columns: {}'.format(len(categorical_cols)))
plt.figure(figsize=(14, 12))
sns.heatmap(train.corr(), vmax=0.8, square=True, cmap='Blues')
corr_cols = train.corr().nlargest(15, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(12, 10))
sns.heatmap(train[corr_cols].corr(), annot=True, vmax=0.8, square=True, cmap='Blues')
for col in train[categorical_cols]:
    if train[col].isnull().sum() > 0:
        print(col, train[col].unique())
col_fillna = ('Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature')
for col in col_fillna:
    train[col].fillna('no', inplace=True)
    test[col].fillna('no', inplace=True)
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
total_missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
total_missing
total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum() / test.isnull().count()).sort_values(ascending=False)
total_missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
total_missing
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
test.isnull().sum().sort_values(ascending=False)[:8]
for col in test.columns:
    if test[col].dtype == 'object' and test[col].isnull().sum():
        test.fillna('no', inplace=True)
test.isnull().sum().sum()
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
train[categorical_cols] = oe.fit_transform(train[categorical_cols])
test[categorical_cols] = oe.fit_transform(test[categorical_cols])
x = train.drop('SalePrice', axis=1)
y = train.SalePrice.copy()
from sklearn.feature_selection import mutual_info_classif
important = mutual_info_classif(x, y)
important_features = pd.Series(important, train.columns[:len(train.columns) - 1])
important_features
feature_cols = []
for (key, val) in important_features.items():
    if np.abs(val) > 0.5:
        feature_cols.append(key)
print(feature_cols)
print(len(feature_cols))
x = x[feature_cols]
xtest = test[feature_cols]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')