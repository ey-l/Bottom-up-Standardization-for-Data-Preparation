from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import mode
import xgboost as xgb
import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.info()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 15))
sns.heatmap(_input1.isna())
drop_list = ['Alley', 'PoolQC', 'MiscFeature', 'Fireplaces', 'Fence']
_input1 = _input1.drop(drop_list, axis=1, inplace=False)
plt.figure(figsize=(30, 20))
sns.heatmap(_input1.isna())
_input0 = _input0.drop(drop_list, axis=1, inplace=False)
_input0.head()
plt.figure(figsize=(30, 20))
sns.heatmap(_input1.corr(), annot=True)
for column in _input1.columns:
    if _input1[column].dtype == 'object':
        label = LabelEncoder()
        _input1[column] = label.fit_transform(_input1[column].values)
    if column != 'SalePrice' and _input0[column].dtype == 'object':
        label = LabelEncoder()
        _input0[column] = label.fit_transform(_input0[column].values)
_input0.head()
from sklearn.impute import KNNImputer
train_columns = _input1.columns
impute = KNNImputer(n_neighbors=5)
_input1 = impute.fit_transform(_input1)
_input1 = pd.DataFrame(_input1, columns=train_columns)
_input1
test_columns = _input0.columns
impute = KNNImputer()
_input0 = impute.fit_transform(_input0)
_input0 = pd.DataFrame(_input0, columns=test_columns)
droplist = ['Id', 'Utilities']
for column1 in _input1.columns:
    for column2 in _input1.columns:
        if abs(_input1[column1].corr(_input1[column2])) > 0.8 and column1 != column2:
            droplist.append(column1)
_input1 = _input1.drop(droplist, axis=1)
_input0 = _input0.drop(droplist, axis=1)
target = _input1['SalePrice']
feature = _input1.drop(['SalePrice'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(feature, target, test_size=0.2, random_state=12)
X_test = X_test.drop('Street', axis=1)
X_train = X_train.drop('Street', axis=1)
model = xgb.XGBRegressor(max_depth=4, n_estimators=400, learning_rate=0.1, min_child_weight=20)