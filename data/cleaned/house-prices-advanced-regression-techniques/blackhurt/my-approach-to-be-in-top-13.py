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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submit = pd.DataFrame(test['Id'])
test = test.set_index('Id')
train.select_dtypes(exclude='object').hist(figsize=(25, 19))

plt.figure(figsize=(25, 19))
sns.heatmap(train.corr(), annot=True)
null = train.loc[:, train.isnull().sum() > 500]
train = train.drop(null, axis=1)
train = train.drop_duplicates()
train = train.drop(['YearRemodAdd', '3SsnPorch', 'PoolArea', 'MiscVal', 'LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', 'BsmtFinSF2', 'LotArea', 'BsmtHalfBath', 'GarageCond', 'GarageQual', 'GarageFinish', 'KitchenQual', 'CentralAir', 'HeatingQC', 'RoofStyle', 'MSZoning', 'LandContour', 'LotConfig', 'Condition1'], axis=1)
null = test.loc[:, test.isnull().sum() > 500]
test = test.drop(null, axis=1)
test = test.drop_duplicates()
test = test.drop(['YearRemodAdd', '3SsnPorch', 'PoolArea', 'MiscVal', 'LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', 'BsmtFinSF2', 'LotArea', 'BsmtHalfBath', 'GarageCond', 'GarageQual', 'GarageFinish', 'KitchenQual', 'CentralAir', 'HeatingQC', 'RoofStyle', 'MSZoning', 'LandContour', 'LotConfig', 'Condition1'], axis=1)
reg1 = train.select_dtypes(exclude='object')
reg2 = test.select_dtypes(exclude='object')

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
train['YearBuilt'] = train['YearBuilt'].map(year_columns)
test['YearBuilt'] = test['YearBuilt'].map(year_columns)
cat1 = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond']
encode = OrdinalEncoder()
cat_ordinal = pd.DataFrame(encode.fit_transform(train[cat1].astype(str)), columns=cat1)
for i in cat_ordinal.columns:
    train[i] = cat_ordinal[i]
    test[i] = cat_ordinal[i]
train = train.drop(['Heating', 'Electrical'], axis=1)
test = test.drop(['Heating', 'Electrical'], axis=1)
train = pd.get_dummies(train, columns=['Street', 'BldgType', 'YearBuilt', 'MasVnrType', 'BsmtExposure', 'PavedDrive'], drop_first=True)
test = pd.get_dummies(test, columns=['Street', 'BldgType', 'YearBuilt', 'MasVnrType', 'BsmtExposure', 'PavedDrive'], drop_first=True)
encode = LabelEncoder()
for i in train.select_dtypes(include='object').columns:
    train[i] = encode.fit_transform(train[i])
    test[i] = encode.fit_transform(test[i])
model = XGBRegressor(base_score=0.4, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4603, gamma=0.05, gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.05, max_delta_step=0, max_depth=3, min_child_weight=1.7817, monotone_constraints='()', n_estimators=2200, n_jobs=4, nthread=-1, num_parallel_tree=1, random_state=7, reg_alpha=0.464, reg_lambda=0.8571, scale_pos_weight=1, subsample=0.5213, silent=True, tree_method='exact', validate_parameters=1, verbosity=0)
pipeline = Pipeline(steps=[('impute', IterativeImputer(max_iter=9, imputation_order='arabic')), ('d', SelectKBest(score_func=f_regression, k=55)), ('e', SelectKBest(score_func=f_classif, k=52)), ('model', model)])
x = train.drop('SalePrice', axis=1)
y = np.log(train['SalePrice'])
(xtrain, xvalid, ytrain, yvalid) = train_test_split(x, y, test_size=0.25)