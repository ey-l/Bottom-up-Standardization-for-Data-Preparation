import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
X_full = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_test_full = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
X_full.head()
X_full.info()
X_full.describe()
X_full['SalePrice'].hist()
X_full.hist(bins=int(np.sqrt(len(X_full))), figsize=(20, 20))
X_full.corr()['SalePrice'].sort_values(ascending=False)
import seaborn as sns
sns.heatmap(X_full.corr(), cmap='YlGnBu')
features_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
X_full['Alley'].unique()
from sklearn.impute import SimpleImputer
imp_none = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='None')
df_concat = imp_none.fit_transform(X_full[features_NA])
X_full[features_NA] = df_concat
X_full['Alley'].unique()
X_full.head()
X_full[['MiscFeature']].info()
features_with_large_nans = X_full.columns[X_full.isna().mean() * 100 > 5]
features_with_large_nans
X_full.drop(features_with_large_nans, axis=1, inplace=True)
features_one_hot = ['Alley', 'MSZoning', 'LandContour', 'Neighborhood', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'GarageType', 'SaleType', 'SaleCondition', 'Condition1', 'Condition2', 'RoofStyle', 'CentralAir', 'BldgType', 'Functional', 'Exterior2nd', 'LotConfig']
Nans_features = X_full[features_one_hot].columns[X_full[features_one_hot].isna().any()].tolist()
Nans_features
imp = SimpleImputer(strategy='most_frequent')
df_concat = imp.fit_transform(X_full[Nans_features])
X_full[Nans_features] = df_concat
X_full[Nans_features].isna().any()
from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False)
for feat in features_one_hot:
    one_hot = en.fit_transform(X_full[feat].values.reshape(-1, 1))
    dff = pd.DataFrame(one_hot, columns=en.categories_, index=X_full.index)
    X_full[en.categories_[0]] = dff
    X_full.drop(feat, axis=1, inplace=True)
features_ordinal_enc = ['LotShape', 'Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Electrical', 'Fence', 'Street', 'PavedDrive']
X_full[features_ordinal_enc].isnull().any()
Nans_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Electrical', 'Fence']
imp = SimpleImputer(strategy='most_frequent')
df_concat = imp.fit_transform(X_full[Nans_features])
X_full[Nans_features] = df_concat
X_full[Nans_features].isna().any()
X_full['LotShape'] = X_full['LotShape'].map({'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3})
X_full['Utilities'] = X_full['Utilities'].map({'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3})
X_full['ExterQual'] = X_full['ExterQual'].map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
X_full['ExterCond'] = X_full['ExterCond'].map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
X_full['BsmtQual'] = X_full['BsmtQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
X_full['BsmtCond'] = X_full['BsmtCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
X_full['BsmtExposure'] = X_full['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
X_full['HeatingQC'] = X_full['HeatingQC'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
X_full['KitchenQual'] = X_full['KitchenQual'].map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
X_full['FireplaceQu'] = X_full['FireplaceQu'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
X_full['GarageFinish'] = X_full['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
X_full['GarageQual'] = X_full['GarageQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
X_full['GarageCond'] = X_full['GarageCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
X_full['PoolQC'] = X_full['PoolQC'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
X_full['Electrical'] = X_full['Electrical'].map({'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0})
X_full['Fence'] = X_full['Fence'].map({'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})
X_full['Street'] = X_full['Street'].map({'Grvl': 0, 'Pave': 1})
X_full['PavedDrive'] = X_full['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
X_full
X_full.columns[X_full.isna().any()].tolist()
X_full[['MasVnrArea']].info()
X_full['MasVnrArea'].hist(figsize=(15, 5))
X_full['MasVnrArea'].fillna(X_full['MasVnrArea'].median(), inplace=True)
X_full['SalePrice']
y = X_full['SalePrice']
X = X_full.drop('SalePrice', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(min_samples_split=3)