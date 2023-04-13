import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1.head()
_input1.info()
_input1.describe()
_input1['SalePrice'].hist()
_input1.hist(bins=int(np.sqrt(len(_input1))), figsize=(20, 20))
_input1.corr()['SalePrice'].sort_values(ascending=False)
import seaborn as sns
sns.heatmap(_input1.corr(), cmap='YlGnBu')
features_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
_input1['Alley'].unique()
from sklearn.impute import SimpleImputer
imp_none = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='None')
df_concat = imp_none.fit_transform(_input1[features_NA])
_input1[features_NA] = df_concat
_input1['Alley'].unique()
_input1.head()
_input1[['MiscFeature']].info()
features_with_large_nans = _input1.columns[_input1.isna().mean() * 100 > 5]
features_with_large_nans
_input1 = _input1.drop(features_with_large_nans, axis=1, inplace=False)
features_one_hot = ['Alley', 'MSZoning', 'LandContour', 'Neighborhood', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'GarageType', 'SaleType', 'SaleCondition', 'Condition1', 'Condition2', 'RoofStyle', 'CentralAir', 'BldgType', 'Functional', 'Exterior2nd', 'LotConfig']
Nans_features = _input1[features_one_hot].columns[_input1[features_one_hot].isna().any()].tolist()
Nans_features
imp = SimpleImputer(strategy='most_frequent')
df_concat = imp.fit_transform(_input1[Nans_features])
_input1[Nans_features] = df_concat
_input1[Nans_features].isna().any()
from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False)
for feat in features_one_hot:
    one_hot = en.fit_transform(_input1[feat].values.reshape(-1, 1))
    dff = pd.DataFrame(one_hot, columns=en.categories_, index=_input1.index)
    _input1[en.categories_[0]] = dff
    _input1 = _input1.drop(feat, axis=1, inplace=False)
features_ordinal_enc = ['LotShape', 'Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Electrical', 'Fence', 'Street', 'PavedDrive']
_input1[features_ordinal_enc].isnull().any()
Nans_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Electrical', 'Fence']
imp = SimpleImputer(strategy='most_frequent')
df_concat = imp.fit_transform(_input1[Nans_features])
_input1[Nans_features] = df_concat
_input1[Nans_features].isna().any()
_input1['LotShape'] = _input1['LotShape'].map({'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3})
_input1['Utilities'] = _input1['Utilities'].map({'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3})
_input1['ExterQual'] = _input1['ExterQual'].map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['ExterCond'] = _input1['ExterCond'].map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['BsmtQual'] = _input1['BsmtQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['BsmtCond'] = _input1['BsmtCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['BsmtExposure'] = _input1['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
_input1['HeatingQC'] = _input1['HeatingQC'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['KitchenQual'] = _input1['KitchenQual'].map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['FireplaceQu'] = _input1['FireplaceQu'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['GarageFinish'] = _input1['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
_input1['GarageQual'] = _input1['GarageQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['GarageCond'] = _input1['GarageCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['PoolQC'] = _input1['PoolQC'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['Electrical'] = _input1['Electrical'].map({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0})
_input1['Fence'] = _input1['Fence'].map({'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})
_input1['Street'] = _input1['Street'].map({'Grvl': 0, 'Pave': 1})
_input1['PavedDrive'] = _input1['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
_input1
_input1.columns[_input1.isna().any()].tolist()
_input1[['MasVnrArea']].info()
_input1['MasVnrArea'].hist(figsize=(15, 5))
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].median(), inplace=False)
_input1['SalePrice']
y = _input1['SalePrice']
X = _input1.drop('SalePrice', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(min_samples_split=3)