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
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_subm = pd.DataFrame(df_test['Id'])
df_test
df_train.describe()
df_train['LotShape'] = df_train['LotShape'].map({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
df_train['Utilities'] = df_train['Utilities'].map({'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0})
df_train['LandSlope'] = df_train['LandSlope'].map({'Gtl': 2, 'Mod': 1, 'Sev': 0})
df_train['ExterQual'] = df_train['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_train['ExterCond'] = df_train['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_train['BsmtQual'] = df_train['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_train['BsmtCond'] = df_train['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_train['BsmtExposure'] = df_train['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
df_train['BsmtFinType2'] = df_train['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
df_train['HeatingQC'] = df_train['HeatingQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_train['Electrical'] = df_train['Electrical'].map({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0})
df_train['KitchenQual'] = df_train['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_train['Functional'] = df_train['Functional'].map({'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1})
df_train['FireplaceQu'] = df_train['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_train['GarageFinish'] = df_train['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0})
df_train['GarageQual'] = df_train['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_train['GarageCond'] = df_train['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_train['PavedDrive'] = df_train['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})
df_train['PoolQC'] = df_train['PoolQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_train['Fence'] = df_train['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
df_test['LotShape'] = df_test['LotShape'].map({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
df_test['Utilities'] = df_test['Utilities'].map({'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0})
df_test['LandSlope'] = df_test['LandSlope'].map({'Gtl': 2, 'Mod': 1, 'Sev': 0})
df_test['ExterQual'] = df_test['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_test['ExterCond'] = df_test['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_test['BsmtQual'] = df_test['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_test['BsmtCond'] = df_test['BsmtCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_test['BsmtExposure'] = df_test['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0})
df_test['HeatingQC'] = df_test['HeatingQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_test['Electrical'] = df_test['Electrical'].map({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0})
df_test['KitchenQual'] = df_test['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})
df_test['Functional'] = df_test['Functional'].map({'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1})
df_test['FireplaceQu'] = df_test['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_test['GarageFinish'] = df_test['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0})
df_test['GarageQual'] = df_test['GarageQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_test['GarageCond'] = df_test['GarageCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_test['PavedDrive'] = df_test['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})
df_test['PoolQC'] = df_test['PoolQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
df_test['Fence'] = df_test['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
for col in df_train.loc[:, df_train.dtypes == 'object'].columns:
    if len(df_train[col].unique()) == 2:
        df_train = pd.get_dummies(df_train, columns=[col], drop_first=True)
        df_test = pd.get_dummies(df_test, columns=[col], drop_first=True)
    elif len(df_train[col].unique()) <= 10:
        df_train = pd.get_dummies(df_train, columns=[col], drop_first=False)
        df_test = pd.get_dummies(df_test, columns=[col], drop_first=False)
    else:
        X_temp = df_train[[col, 'Id']]
        X_temp = X_temp.groupby(col, as_index=False).count()
        X_temp = X_temp.sort_values('Id', ascending=False)
        X_temp = X_temp[X_temp['Id'] > 1].head(10)
        for val in X_temp[col]:
            df_train[col + '_' + val] = df_train[col].apply(lambda x: 1 if x == val else 0)
            df_test[col + '_' + val] = df_test[col].apply(lambda x: 1 if x == val else 0)
col_diff_list = [x for x in list(df_train.columns) if x not in list(df_test.columns)]
df_test[col_diff_list] = 0
cols_num = df_train.select_dtypes(include=np.number).columns.tolist()
cols_num.remove('SalePrice')
cols_num.remove('Id')
print('Total columns:', len(cols_num))
cols_mean = []
cols_mode = []
for col in cols_num:
    if len(df_train[col].unique()) < 0.1 * len(df_train):
        cols_mode.append(col)
    else:
        cols_mean.append(col)
print('Mode columns:', len(cols_mode))
print('Mean columns:', len(cols_mean))

def mround(number, multiple):
    return multiple * round(number / multiple)
df_train['seg'] = df_train['SalePrice'].apply(lambda x: str(mround(x, 50000)))
df_train[['seg', 'SalePrice']].groupby('seg').count()
X = df_train[cols_num]
y = df_train['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, stratify=df_train['seg'])
X_train.columns
imp_mean = SimpleImputer(strategy='mean')