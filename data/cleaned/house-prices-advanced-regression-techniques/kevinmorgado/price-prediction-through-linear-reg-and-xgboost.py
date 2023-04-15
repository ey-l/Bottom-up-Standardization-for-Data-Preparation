import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test
special_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for i in special_cols:
    null_cols = train[train[i].isna()].index
    for n in null_cols:
        train.loc[n, i] = 'None'
for i in special_cols:
    null_cols = test[test[i].isna()].index
    for n in null_cols:
        test.loc[n, i] = 'None'
null_cols = train[train['LotFrontage'].isna()].index
for n in null_cols:
    train.loc[n, 'LotFrontage'] = 0
null_cols = train[train['GarageYrBlt'].isna()].index
for n in null_cols:
    train.loc[n, 'GarageYrBlt'] = train['GarageYrBlt'].mode()[0]
null_cols = train[train['MasVnrType'].isna()].index
for n in null_cols:
    train.loc[n, 'MasVnrType'] = 'None'
null_cols = train[train['MasVnrArea'].isna()].index
for n in null_cols:
    train.loc[n, 'MasVnrArea'] = 0
null_cols = test[test['LotFrontage'].isna()].index
for n in null_cols:
    test.loc[n, 'LotFrontage'] = 0
null_cols = test[test['GarageYrBlt'].isna()].index
for n in null_cols:
    test.loc[n, 'GarageYrBlt'] = test['GarageYrBlt'].mode()[0]
null_cols = test[test['MasVnrType'].isna()].index
for n in null_cols:
    test.loc[n, 'MasVnrType'] = 'None'
null_cols = test[test['MasVnrArea'].isna()].index
for n in null_cols:
    test.loc[n, 'MasVnrArea'] = 0
null_cols = train[train['Electrical'].isna()].index
for n in null_cols:
    train.loc[n, 'Electrical'] = train['Electrical'].mode()[0]
Rem_null_test = []
for i in test.columns:
    if test[i].isna().value_counts()[0] - len(test[i]) < 0:
        Rem_null_test.append(i)
for col in Rem_null_test:
    if test[col].dtype == 'O':
        null_cols = test[test[col].isna()].index
        for n in null_cols:
            test.loc[n, col] = test[col].mode()[0]
    else:
        null_cols = test[test[col].isna()].index
        for n in null_cols:
            test.loc[n, col] = test[col].mean()
yes_null_cols_train = []
for i in test.columns:
    if train[i].isna().value_counts()[0] - len(train[i]) < 0:
        yes_null_cols_train.append(i)
len(yes_null_cols_train)
yes_null_cols_test = []
for i in test.columns:
    if test[i].isna().value_counts()[0] - len(test[i]) < 0:
        yes_null_cols_test.append(i)
len(yes_null_cols_test)
formula = 'SalePrice ~ '
columns = ''
for i in train.columns[1:-1]:
    if columns == '':
        columns = columns + "Q('" + str(i) + "')"
    else:
        columns = columns + " + Q('" + str(i) + "')"
columns
final_formula = formula + columns