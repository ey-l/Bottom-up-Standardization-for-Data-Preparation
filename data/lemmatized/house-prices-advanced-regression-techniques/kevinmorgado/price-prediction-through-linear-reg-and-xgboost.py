import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0
special_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for i in special_cols:
    null_cols = _input1[_input1[i].isna()].index
    for n in null_cols:
        _input1.loc[n, i] = 'None'
for i in special_cols:
    null_cols = _input0[_input0[i].isna()].index
    for n in null_cols:
        _input0.loc[n, i] = 'None'
null_cols = _input1[_input1['LotFrontage'].isna()].index
for n in null_cols:
    _input1.loc[n, 'LotFrontage'] = 0
null_cols = _input1[_input1['GarageYrBlt'].isna()].index
for n in null_cols:
    _input1.loc[n, 'GarageYrBlt'] = _input1['GarageYrBlt'].mode()[0]
null_cols = _input1[_input1['MasVnrType'].isna()].index
for n in null_cols:
    _input1.loc[n, 'MasVnrType'] = 'None'
null_cols = _input1[_input1['MasVnrArea'].isna()].index
for n in null_cols:
    _input1.loc[n, 'MasVnrArea'] = 0
null_cols = _input0[_input0['LotFrontage'].isna()].index
for n in null_cols:
    _input0.loc[n, 'LotFrontage'] = 0
null_cols = _input0[_input0['GarageYrBlt'].isna()].index
for n in null_cols:
    _input0.loc[n, 'GarageYrBlt'] = _input0['GarageYrBlt'].mode()[0]
null_cols = _input0[_input0['MasVnrType'].isna()].index
for n in null_cols:
    _input0.loc[n, 'MasVnrType'] = 'None'
null_cols = _input0[_input0['MasVnrArea'].isna()].index
for n in null_cols:
    _input0.loc[n, 'MasVnrArea'] = 0
null_cols = _input1[_input1['Electrical'].isna()].index
for n in null_cols:
    _input1.loc[n, 'Electrical'] = _input1['Electrical'].mode()[0]
Rem_null_test = []
for i in _input0.columns:
    if _input0[i].isna().value_counts()[0] - len(_input0[i]) < 0:
        Rem_null_test.append(i)
for col in Rem_null_test:
    if _input0[col].dtype == 'O':
        null_cols = _input0[_input0[col].isna()].index
        for n in null_cols:
            _input0.loc[n, col] = _input0[col].mode()[0]
    else:
        null_cols = _input0[_input0[col].isna()].index
        for n in null_cols:
            _input0.loc[n, col] = _input0[col].mean()
yes_null_cols_train = []
for i in _input0.columns:
    if _input1[i].isna().value_counts()[0] - len(_input1[i]) < 0:
        yes_null_cols_train.append(i)
len(yes_null_cols_train)
yes_null_cols_test = []
for i in _input0.columns:
    if _input0[i].isna().value_counts()[0] - len(_input0[i]) < 0:
        yes_null_cols_test.append(i)
len(yes_null_cols_test)
formula = 'SalePrice ~ '
columns = ''
for i in _input1.columns[1:-1]:
    if columns == '':
        columns = columns + "Q('" + str(i) + "')"
    else:
        columns = columns + " + Q('" + str(i) + "')"
columns
final_formula = formula + columns