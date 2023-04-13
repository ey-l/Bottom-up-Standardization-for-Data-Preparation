import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import shap
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.isnull().sum()[_input1.isnull().sum() != 0]
train_df = _input1.drop(['Id', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'], axis=1)
test_df = _input0.drop(['Id', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'], axis=1)

def encode_target_smooth(data, target, categ_variables, smooth):
    """    
    Apply target encoding with smoothing.
    
    Parameters
    ----------
    data: pd.DataFrame
    target: str, dependent variable
    categ_variables: list of str, variables to encode
    smooth: int, number of observations to weigh global average with
    
    Returns
    --------
    encoded_dataset: pd.DataFrame
    code_map: dict, mapping to be used on validation/test datasets 
    defaul_map: dict, mapping to replace previously unseen values with
    """
    train_target = data.copy()
    code_map = dict()
    default_map = dict()
    for v in categ_variables:
        prior = data[target].mean()
        n = data.groupby(v).size()
        mu = data.groupby(v)[target].mean()
        mu_smoothed = (n * mu + smooth * prior) / (n + smooth)
        train_target.loc[:, v] = train_target[v].map(mu_smoothed)
        code_map[v] = mu_smoothed
        default_map[v] = prior
    return (train_target, code_map, default_map)
Tr_mean = pd.concat([train_df.loc[:, train_df.dtypes == object], train_df['SalePrice']], axis=1)
Te_mean = test_df.loc[:, test_df.dtypes == object]
cat_vars = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
(train_target_smooth, target_map, default_map) = encode_target_smooth(Tr_mean, 'SalePrice', cat_vars, 500)
test_target_smooth = Te_mean.copy()
for v in cat_vars:
    test_target_smooth.loc[:, v] = test_target_smooth[v].map(target_map[v])
no_Tr_mean = train_df.loc[:, train_df.dtypes != object]
no_Tr_mean = no_Tr_mean.drop(['SalePrice'], axis=1)
no_Te_mean = test_df.loc[:, train_df.dtypes != object]
df_train = pd.concat([train_target_smooth, no_Tr_mean], axis=1)
df_test = pd.concat([test_target_smooth, no_Te_mean], axis=1)
df_train = df_train.fillna(df_train.mean())
df_test = df_test.fillna(df_test.mean())
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
from sklearn.model_selection import GridSearchCV

def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)