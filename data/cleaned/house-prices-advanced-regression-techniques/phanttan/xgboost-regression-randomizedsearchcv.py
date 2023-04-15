import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
xgb.set_config(verbosity=0)
warnings.filterwarnings('ignore')
data_dir = Path('_data/input/house-prices-advanced-regression-techniques/')
train_full = pd.read_csv(data_dir / 'train.csv', index_col='Id')
test_full = pd.read_csv(data_dir / 'test.csv', index_col='Id')
df = pd.concat([train_full, test_full])
X = df.copy()
y = X.pop('SalePrice')
X['Exterior2nd'] = X['Exterior2nd'].replace({'Brk Cmn': 'BrkComm'})
X['GarageYrBlt'] = X['GarageYrBlt'].where(X.GarageYrBlt <= 2010, X.YearBuilt)
X.rename(columns={'1stFlrSF': 'FirstFlrSF', '2ndFlrSF': 'SecondFlrSF', '3SsnPorch': 'Threeseasonporch'}, inplace=True)
features_nom = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
for name in features_nom:
    X[name] = X[name].astype('category')
    if 'None' not in X[name].cat.categories:
        X[name].cat.add_categories('None', inplace=True)
five_levels = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
ten_levels = list(range(10))
features_ordered = {'OverallQual': ten_levels, 'OverallCond': ten_levels, 'ExterQual': five_levels, 'ExterCond': five_levels, 'BsmtQual': five_levels, 'BsmtCond': five_levels, 'HeatingQC': five_levels, 'KitchenQual': five_levels, 'FireplaceQu': five_levels, 'GarageQual': five_levels, 'GarageCond': five_levels, 'PoolQC': five_levels, 'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'], 'LandSlope': ['Sev', 'Mod', 'Gtl'], 'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'], 'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], 'Functional': ['Sal', 'Sev', 'Maj1', 'Maj2', 'Mod', 'Min2', 'Min1', 'Typ'], 'GarageFinish': ['Unf', 'RFn', 'Fin'], 'PavedDrive': ['N', 'P', 'Y'], 'Utilities': ['NoSeWa', 'NoSewr', 'AllPub'], 'CentralAir': ['N', 'Y'], 'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], 'Fence': ['MnWw', 'GdWo', 'MnPrv', 'GdPrv']}
features_ordered = {key: ['None'] + value for (key, value) in features_ordered.items()}
for (name, levels) in features_ordered.items():
    X[name] = X[name].astype(CategoricalDtype(levels, ordered=True))
X['OverallQual'] = X['OverallQual'].apply(str)
X['OverallCond'] = X['OverallCond'].apply(str)
for name in X.select_dtypes('number'):
    X[name] = X[name].fillna(0)
for name in X.select_dtypes('category'):
    X[name] = X[name].fillna('None')
X_threshold = X.loc[train_full.index]
for colname in X_threshold.select_dtypes(['object', 'category']):
    (X_threshold[colname], _) = X_threshold[colname].factorize()
varThres = VarianceThreshold(threshold=0.99 * (1 - 0.99))