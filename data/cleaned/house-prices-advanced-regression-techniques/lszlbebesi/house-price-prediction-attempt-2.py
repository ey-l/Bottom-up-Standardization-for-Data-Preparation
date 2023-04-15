import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
numeric_but_string = ['MSSubClass', 'YrSold', 'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'Fireplaces', 'GarageCars', 'KitchenAbvGr']
numerical_cols = [col for col in train.columns[1:-2] if (train.dtypes[col] == 'int64' or train.dtypes[col] == 'float64') and col not in numeric_but_string]
target_col = 'SalePrice_log'
train['SalePrice_log'] = np.log(train['SalePrice'])
train_tf = train[['Id']].copy()
test_tf = test[['Id']].copy()
std_map = {}
numerical_variables_revisited = list()
for col in train.columns:
    if col not in ['Id']:
        if (train.dtypes[col] == 'int64' or train.dtypes[col] == 'float64') and col not in numeric_but_string:
            train_tf[col] = train[col].copy()
            if train.dtypes[col] == 'int64':
                print(col + ' ' + str(len(train[col].unique())))
            if col not in ['SalePrice_log', 'SalePrice']:
                test_tf[col] = test[col].copy()
                numerical_variables_revisited.append(col)
        else:
            train_tf = pd.concat([train_tf, pd.get_dummies(train[col], prefix=col, prefix_sep='_')], axis=1)
            test_tf = pd.concat([test_tf, pd.get_dummies(test[col], prefix=col, prefix_sep='_')], axis=1)
    if col == 'PoolArea':
        train_tf['Pool_Exists'] = np.where(train['PoolArea'] > 0, 1, 0)
        test_tf['Pool_Exists'] = np.where(test['PoolArea'] > 0, 1, 0)
train_tf.drop(columns=['Id'], inplace=True)
test_tf.drop(columns=['Id'], inplace=True)
for col in train_tf.columns:
    if col not in test_tf.columns:
        test_tf[col] = 0
for col in test_tf.columns:
    if col not in train_tf.columns:
        train_tf[col] = 0
train_target = train_tf[[target_col]].copy()
train_tf = train_tf[test_tf.columns].copy()
saved_cols = train_tf.columns
scaler = StandardScaler()