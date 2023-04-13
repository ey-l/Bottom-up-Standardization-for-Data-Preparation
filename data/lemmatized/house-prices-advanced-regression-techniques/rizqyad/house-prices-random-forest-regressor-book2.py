import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.model_selection import train_test_split
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1.info()
_input1.head()
_input0.head()
_input1 = _input1.dropna(axis=0, subset=['SalePrice'], inplace=False)
y = _input1.SalePrice
_input1 = _input1.drop(['SalePrice'], axis=1, inplace=False)
print(_input1.shape)
missing_val_count_by_column = _input1.isnull().sum()
print(missing_val_count_by_column[missing_val_count_by_column > 0])
droped_cols = [col for col in _input1.columns if _input1[col].isnull().sum() > len(_input1) * 0.8]
print('droped columns:', droped_cols)
X_train_reduced = _input1.drop(droped_cols, axis=1)
(X_train, X_valid, y_train, y_valid) = train_test_split(X_train_reduced, y, train_size=0.8, test_size=0.2, random_state=0)
ordinal_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'BldgType', 'HouseStyle', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
onehot_cols = ['Condition1', 'Condition2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating']
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
my_cols = numerical_cols + ordinal_cols + onehot_cols
X_train_selected = X_train[my_cols].copy()
X_valid_selected = X_valid[my_cols].copy()
print('Used columns:', my_cols)
all_cols = _input1.columns.values.tolist()
not_used_cols = [item for item in all_cols if item not in my_cols]
print('Not used columns:', not_used_cols)
MSZoning = ['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM', 'NA']
Street = ['Grvl', 'Pave', 'NA']
LotShape = ['Reg', 'IR1', 'IR2', 'IR3', 'NA']
LandContour = ['Lvl', 'Bnk', 'HLS', 'Low', 'NA']
Utilities = ['AllPub', 'NoSewr', 'NoSeWa', 'ELO', 'NA']
LotConfig = ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3', 'NA']
LandSlope = ['Gtl', 'Mod', 'Sev', 'NA']
BldgType = ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs', 'NA']
HouseStyle = ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl', 'NA']
ExterQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
ExterCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
Foundation = ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood', 'NA']
BsmtQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
BsmtCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
BsmtExposure = ['Gd', 'Av', 'Mn', 'No', 'NA']
BsmtFinType1 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
BsmtFinType2 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
HeatingQC = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
CentralAir = ['N', 'Y', 'NA']
Electrical = ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', 'NA']
KitchenQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
Functional = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal', 'NA']
FireplaceQu = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
GarageType = ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA']
GarageFinish = ['Fin', 'RFn', 'Unf', 'NA']
GarageQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
GarageCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
PavedDrive = ['Y', 'P', 'N', 'NA']
SaleType = ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth', 'NA']
SaleCondition = ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial', 'NA']
pos_Ord = [MSZoning, Street, LotShape, LandContour, Utilities, LotConfig, LandSlope, BldgType, HouseStyle, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, SaleType, SaleCondition]
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
numerical_transformer = SimpleImputer(strategy='constant')
ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='NA')), ('ordinal', OrdinalEncoder(categories=pos_Ord))])
onehot_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('od', ordinal_transformer, ordinal_cols), ('oh', onehot_transformer, onehot_cols)])
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def score_model(X_train, X_valid, y_train, y_valid, n_estimators):
    new_model = RandomForestRegressor(n_estimators, random_state=0)
    new_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', new_model)])