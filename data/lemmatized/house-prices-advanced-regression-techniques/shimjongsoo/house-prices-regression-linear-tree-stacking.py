import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
train_pp = _input1.drop('Id', axis=1)
train_pp
drop_col = train_pp.isnull().sum().sort_values(ascending=False).head(5).index
drop_col
train_pp = train_pp.drop(drop_col, axis=1)
train_pp
sns.distplot(train_pp['SalePrice'], hist=True, kde=True, color='green')
train_pp['SalePrice'] = np.log1p(train_pp['SalePrice'])
sns.distplot(train_pp['SalePrice'], hist=True, kde=True, color='green')
train_pp = train_pp.fillna(train_pp.mean(), inplace=False)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0
test_pp = _input0.drop('Id', axis=1)
test_pp
drop_col = test_pp.isnull().sum().sort_values(ascending=False).head(5).index
drop_col
test_pp = test_pp.drop(drop_col, axis=1)
test_pp
test_pp = test_pp.fillna(test_pp.mean(), inplace=False)

def check_train_test_column_values(train, test, column):
    print('{} Column에 대한 train_test_values_check 입니다======================='.format(column))
    train_colset = set(_input1[column])
    test_colset = set(_input0[column])
    print(f'Train-set에 있는 고유한 value 개수 : {len(train_colset)}')
    print(f'Test-set에 있는 고유한 value 개수 : {len(test_colset)}')
    print('=' * 80)
    common_colset = train_colset.intersection(test_colset)
    print(f'Train/Test-set에 공통으로 포함되어 있는 value 개수 : {len(common_colset)}')
    if len(common_colset) > 100:
        pass
    else:
        try:
            print(f'Train/Test-set에 공통으로 포함되어 있는 value : {sorted(common_colset)}')
        except:
            print(f'Train/Test-set에 공통으로 포함되어 있는 value : {common_colset}')
    print('=' * 80)
    train_only_colset = train_colset.difference(test_colset)
    print(f'Train-set에만 있는 value는 총 {len(train_only_colset)} 개 입니다.')
    if len(train_only_colset) > 100:
        pass
    else:
        try:
            print(f'Train-set에만 있는 value는 : {sorted(train_only_colset)}')
        except:
            print(f'Train-set에만 있는 value는 : {train_only_colset}')
    print('=' * 80)
    test_only_colset = test_colset.difference(train_colset)
    print(f'Test-set에만 있는 value는 총 {len(test_only_colset)} 개 입니다.')
    if len(test_only_colset) > 100:
        pass
    else:
        try:
            print(f'Test-set에만 있는 value는 : {sorted(test_only_colset)}')
        except:
            print(f'Test-set에만 있는 value는 : {test_only_colset}')
    print(' ')
obj_cols = []
for col in train_pp.columns:
    if train_pp[col].dtypes == 'object':
        obj_cols.append(col)
for col in obj_cols:
    check_train_test_column_values(train_pp, test_pp, col)
train_fin = train_pp.drop(obj_cols, axis=1)
test_fin = test_pp.drop(obj_cols, axis=1)
concat_data = pd.concat([train_pp[obj_cols], test_pp[obj_cols]], axis=0)
concat_data[:1460]
dummy_data = pd.get_dummies(concat_data)
dummy_data.shape
train_dummy = dummy_data[:1460]
test_dummy = dummy_data[1460:]
train_fin = pd.concat([train_fin, train_dummy], axis=1)
test_fin = pd.concat([test_fin, test_dummy], axis=1)
X = train_fin.drop('SalePrice', axis=1)
y = train_fin.SalePrice
_input0 = test_fin
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.metrics import mean_squared_error

def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print('{0} log transform RMSE: {1}'.format(model.__class__.__name__, np.round(rmse, 3)))
    return rmse

def get_rmses(models):
    rmses = []
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lr_reg = LinearRegression()