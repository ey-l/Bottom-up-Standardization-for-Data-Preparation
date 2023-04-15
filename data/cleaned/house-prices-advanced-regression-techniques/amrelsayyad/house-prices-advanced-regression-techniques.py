import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler

print('Reading data...')
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
df.info()
df.head()
df.loc[:, df.isna().mean() > 0.4].isna().mean()
df = df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
X = df.drop('SalePrice', axis=1)
y = np.log(df['SalePrice'])
nominal_features = ['MSSubClass', 'MSZoning', 'Street', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'SaleType', 'SaleCondition', 'GarageType']
ordinal_features = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'Electrical', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 'GarageQual', 'GarageCond']
continuous_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
discrete_features = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']
for col in nominal_features + ordinal_features:
    X[col] = X[col].fillna('None')
for col in continuous_features + discrete_features:
    X[col] = X[col].fillna(0)
print('\nOne-hot encoding...\n')
dummies = pd.get_dummies(X[nominal_features]).sort_index()
X = pd.concat([X, dummies], axis=1)
X = X.drop(nominal_features, axis=1)
X.info()
X.head()
rating = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_encoding = {'LotShape': {'None': 0, 'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}, 'Utilities': {'None': 0, 'ElO': 1, 'NoSeWa': 2, 'NoSeWr': 3, 'AllPub': 4}, 'LandSlope': {'None': 0, 'Gtl': 1, 'Mod': 2, 'Sev': 3}, 'ExterQual': rating, 'ExterCond': rating, 'BsmtQual': rating, 'BsmtCond': rating, 'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 'HeatingQC': rating, 'CentralAir': {'None': 0, 'N': 1, 'Y': 2}, 'Electrical': {'None': 0, 'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 'KitchenQual': rating, 'Functional': {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}, 'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 'GarageQual': rating, 'GarageCond': rating, 'PavedDrive': {'None': 0, 'N': 1, 'P': 2, 'Y': 3}}
print('\nOrdinal encoding...\n')
X = X.replace(ordinal_encoding)
X.info()
X.head()
print('\nScaling features...\n')