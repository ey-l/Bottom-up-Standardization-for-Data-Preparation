import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_subm = pd.DataFrame(_input0['Id'])
_input0
_input1.describe()
_input1['LotShape'] = _input1['LotShape'].map({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
_input1['Utilities'] = _input1['Utilities'].map({'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0})
_input1['LandSlope'] = _input1['LandSlope'].map({'Gtl': 2, 'Mod': 1, 'Sev': 0})
_input1['ExterQual'] = _input1['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input1['ExterCond'] = _input1['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input1['BsmtQual'] = _input1['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input1['BsmtCond'] = _input1['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input1['BsmtExposure'] = _input1['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
_input1['HeatingQC'] = _input1['HeatingQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input1['Electrical'] = _input1['Electrical'].map({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0})
_input1['KitchenQual'] = _input1['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input1['Functional'] = _input1['Functional'].map({'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1})
_input1['FireplaceQu'] = _input1['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input1['GarageFinish'] = _input1['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0})
_input1['GarageQual'] = _input1['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input1['GarageCond'] = _input1['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input1['PavedDrive'] = _input1['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})
_input1['PoolQC'] = _input1['PoolQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input1['Fence'] = _input1['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
_input0['LotShape'] = _input0['LotShape'].map({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
_input0['Utilities'] = _input0['Utilities'].map({'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0})
_input0['LandSlope'] = _input0['LandSlope'].map({'Gtl': 2, 'Mod': 1, 'Sev': 0})
_input0['ExterQual'] = _input0['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input0['ExterCond'] = _input0['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input0['BsmtQual'] = _input0['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input0['BsmtCond'] = _input0['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input0['BsmtExposure'] = _input0['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
_input0['HeatingQC'] = _input0['HeatingQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input0['Electrical'] = _input0['Electrical'].map({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0})
_input0['KitchenQual'] = _input0['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
_input0['Functional'] = _input0['Functional'].map({'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1})
_input0['FireplaceQu'] = _input0['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input0['GarageFinish'] = _input0['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0})
_input0['GarageQual'] = _input0['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input0['GarageCond'] = _input0['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input0['PavedDrive'] = _input0['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})
_input0['PoolQC'] = _input0['PoolQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
_input0['Fence'] = _input0['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
for col in _input1.loc[:, _input1.dtypes == 'object'].columns:
    if len(_input1[col].unique()) == 2:
        _input1 = pd.get_dummies(_input1, columns=[col], drop_first=True)
        _input0 = pd.get_dummies(_input0, columns=[col], drop_first=True)
    elif len(_input1[col].unique()) <= 10:
        _input1 = pd.get_dummies(_input1, columns=[col], drop_first=False)
        _input0 = pd.get_dummies(_input0, columns=[col], drop_first=False)
    else:
        X_temp = _input1[[col, 'Id']]
        X_temp = X_temp.groupby(col, as_index=False).count()
        X_temp = X_temp.sort_values('Id', ascending=False)
        X_temp = X_temp[X_temp['Id'] > 1].head(10)
        for val in X_temp[col]:
            _input1[col + '_' + val] = _input1[col].apply(lambda x: 1 if x == val else 0)
            _input0[col + '_' + val] = _input0[col].apply(lambda x: 1 if x == val else 0)
col_diff_list = [x for x in list(_input1.columns) if x not in list(_input0.columns)]
_input0[col_diff_list] = 0
cols_num = _input1.select_dtypes(include=np.number).columns.tolist()
cols_num.remove('SalePrice')
cols_num.remove('Id')
print('Total columns:', len(cols_num))
cols_mean = []
cols_mode = []
for col in cols_num:
    if len(_input1[col].unique()) < 0.1 * len(_input1):
        cols_mode.append(col)
    else:
        cols_mean.append(col)
print('Mode columns:', len(cols_mode))
print('Mean columns:', len(cols_mean))

def mround(number, multiple):
    return multiple * round(number / multiple)
_input1['seg'] = _input1['SalePrice'].apply(lambda x: str(mround(x, 50000)))
_input1[['seg', 'SalePrice']].groupby('seg').count()
X = _input1[cols_num]
y = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, stratify=_input1['seg'])
X_train.columns
imp_mean = SimpleImputer(strategy='mean')