import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.info()
print(train.shape)
print(test.shape)
null_cols = (train.isnull().sum() / len(train)).sort_values(ascending=False)[:20]
null_cols
null_cols = null_cols[null_cols > 0.1].index
null_cols
null_cols = list(null_cols)
null_cols
null_cols.append('Id')
null_cols
train_processed = train.drop(null_cols, axis=1)
test_processed = test.drop(null_cols, axis=1)
print(train_processed.shape)
print(test_processed.shape)
correlation = pd.get_dummies(train_processed, drop_first=True).corr()['SalePrice']
correlation = abs(correlation)
correlation
low_corr = correlation[correlation < 0.2].index
low_corr
pd.get_dummies(train_processed, drop_first=True).shape
train_processed = pd.get_dummies(train_processed, drop_first=True)
test_processed = pd.get_dummies(test_processed, drop_first=True)
print(train_processed.shape)
print(test_processed.shape)
missing_cols = set(train_processed.columns) - set(test_processed.columns)
missing_cols
for col in missing_cols:
    test_processed[col] = 0
missing_cols = set(train_processed.columns) - set(test_processed.columns)
missing_cols
for col in missing_cols:
    test_processed[col] = 0
print(train_processed.shape)
print(test_processed.shape)
train_processed.drop(low_corr, axis=1, inplace=True)
test_processed.drop(low_corr, axis=1, inplace=True)
print(train_processed.shape)
print(test_processed.shape)
X = train_processed.drop('SalePrice', axis=1)
Y = train_processed.SalePrice
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
imputer = SimpleImputer()
scaler = StandardScaler()
preprocess = Pipeline([('imputer', imputer), ('scaler', scaler)])
lr = LinearRegression()
pipeline = Pipeline([('preprocess', preprocess), ('lr', lr)])
(X_train, X_test, y_train, y_test) = train_test_split(X, Y, train_size=0.8)