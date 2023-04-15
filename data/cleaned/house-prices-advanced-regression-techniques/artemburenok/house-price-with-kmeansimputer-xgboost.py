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
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
train_data.info()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 15))
sns.heatmap(train_data.isna())

drop_list = ['Alley', 'PoolQC', 'MiscFeature', 'Fireplaces', 'Fence']
train_data.drop(drop_list, axis=1, inplace=True)
plt.figure(figsize=(30, 20))
sns.heatmap(train_data.isna())

test_data.drop(drop_list, axis=1, inplace=True)
test_data.head()
plt.figure(figsize=(30, 20))
sns.heatmap(train_data.corr(), annot=True)

for column in train_data.columns:
    if train_data[column].dtype == 'object':
        label = LabelEncoder()
        train_data[column] = label.fit_transform(train_data[column].values)
    if column != 'SalePrice' and test_data[column].dtype == 'object':
        label = LabelEncoder()
        test_data[column] = label.fit_transform(test_data[column].values)
test_data.head()
from sklearn.impute import KNNImputer
train_columns = train_data.columns
impute = KNNImputer(n_neighbors=5)
train_data = impute.fit_transform(train_data)
train_data = pd.DataFrame(train_data, columns=train_columns)
train_data
test_columns = test_data.columns
impute = KNNImputer()
test_data = impute.fit_transform(test_data)
test_data = pd.DataFrame(test_data, columns=test_columns)
droplist = ['Id', 'Utilities']
for column1 in train_data.columns:
    for column2 in train_data.columns:
        if abs(train_data[column1].corr(train_data[column2])) > 0.8 and column1 != column2:
            droplist.append(column1)
train_data = train_data.drop(droplist, axis=1)
test_data = test_data.drop(droplist, axis=1)
target = train_data['SalePrice']
feature = train_data.drop(['SalePrice'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(feature, target, test_size=0.2, random_state=12)
X_test = X_test.drop('Street', axis=1)
X_train = X_train.drop('Street', axis=1)
model = xgb.XGBRegressor(max_depth=4, n_estimators=400, learning_rate=0.1, min_child_weight=20)