import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, PowerTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, f_classif
from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submit = pd.DataFrame(_input0['Id'])
_input0 = _input0.set_index('Id')
_input1.select_dtypes(exclude='object').hist(figsize=(25, 19))
plt.figure(figsize=(25, 19))
sns.heatmap(_input1.corr(), annot=True)
null = _input1.loc[:, _input1.isnull().sum() > 500]
_input1 = _input1.drop(null, axis=1)
_input1 = _input1.drop_duplicates()
_input1 = _input1.drop(['YearRemodAdd', '3SsnPorch', 'PoolArea', 'MiscVal', 'LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', 'BsmtFinSF2', 'LotArea', 'BsmtHalfBath', 'GarageCond', 'GarageQual', 'GarageFinish', 'KitchenQual', 'CentralAir', 'HeatingQC', 'RoofStyle', 'MSZoning', 'LandContour', 'LotConfig', 'Condition1'], axis=1)
null = _input0.loc[:, _input0.isnull().sum() > 500]
_input0 = _input0.drop(null, axis=1)
_input0 = _input0.drop_duplicates()
_input0 = _input0.drop(['YearRemodAdd', '3SsnPorch', 'PoolArea', 'MiscVal', 'LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', 'BsmtFinSF2', 'LotArea', 'BsmtHalfBath', 'GarageCond', 'GarageQual', 'GarageFinish', 'KitchenQual', 'CentralAir', 'HeatingQC', 'RoofStyle', 'MSZoning', 'LandContour', 'LotConfig', 'Condition1'], axis=1)
reg1 = _input1.select_dtypes(exclude='object')
reg2 = _input0.select_dtypes(exclude='object')

def year_columns(year):
    a = ''
    if year <= 1900:
        a = 'too_old'
    elif year <= 1950:
        a = 'old'
    elif year <= 1980:
        a = 'middle'
    else:
        a = 'new'
    return a
_input1['YearBuilt'] = _input1['YearBuilt'].map(year_columns)
_input0['YearBuilt'] = _input0['YearBuilt'].map(year_columns)
cat1 = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond']
encode = OrdinalEncoder()
cat_ordinal = pd.DataFrame(encode.fit_transform(_input1[cat1].astype(str)), columns=cat1)
for i in cat_ordinal.columns:
    _input1[i] = cat_ordinal[i]
    _input0[i] = cat_ordinal[i]
_input1 = _input1.drop(['Heating', 'Electrical'], axis=1)
_input0 = _input0.drop(['Heating', 'Electrical'], axis=1)
_input1 = pd.get_dummies(_input1, columns=['Street', 'BldgType', 'YearBuilt', 'MasVnrType', 'BsmtExposure', 'PavedDrive'], drop_first=True)
_input0 = pd.get_dummies(_input0, columns=['Street', 'BldgType', 'YearBuilt', 'MasVnrType', 'BsmtExposure', 'PavedDrive'], drop_first=True)
encode = LabelEncoder()
for i in _input1.select_dtypes(include='object').columns:
    _input1[i] = encode.fit_transform(_input1[i])
    _input0[i] = encode.fit_transform(_input0[i])
model = XGBRegressor(base_score=0.4, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4603, gamma=0.05, gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.05, max_delta_step=0, max_depth=3, min_child_weight=1.7817, monotone_constraints='()', n_estimators=2200, n_jobs=4, nthread=-1, num_parallel_tree=1, random_state=7, reg_alpha=0.464, reg_lambda=0.8571, scale_pos_weight=1, subsample=0.5213, silent=True, tree_method='exact', validate_parameters=1, verbosity=0)
pipeline = Pipeline(steps=[('impute', IterativeImputer(max_iter=9, imputation_order='arabic')), ('d', SelectKBest(score_func=f_regression, k=55)), ('e', SelectKBest(score_func=f_classif, k=52)), ('model', model)])
x = _input1.drop('SalePrice', axis=1)
y = np.log(_input1['SalePrice'])
(xtrain, xvalid, ytrain, yvalid) = train_test_split(x, y, test_size=0.25)