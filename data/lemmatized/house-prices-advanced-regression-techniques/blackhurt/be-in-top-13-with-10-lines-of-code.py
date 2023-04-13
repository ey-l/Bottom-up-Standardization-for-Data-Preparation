import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer, RobustScaler, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectKBest, f_classif, SelectKBest, mutual_info_regression, f_regression
from sklearn.impute import IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submit = pd.DataFrame(_input0['Id'])
null = _input1.loc[:, _input1.isnull().sum() > 500]
_input1 = _input1.drop(null, inplace=False, axis=1)
null = _input0.loc[:, _input0.isnull().sum() > 500]
_input0 = _input0.drop(null, inplace=False, axis=1)
encode = LabelEncoder()
for i in _input1.select_dtypes(include='object').columns:
    _input1[i] = encode.fit_transform(_input1[i])
    _input0[i] = encode.fit_transform(_input0[i])
y = np.log(_input1['SalePrice'])
_input0 = _input0.drop('Id', axis=1)
model = XGBRegressor(base_score=0.4, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4603, gamma=0.05, gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.05, max_delta_step=0, max_depth=3, min_child_weight=1.7817, monotone_constraints='()', n_estimators=2200, n_jobs=4, nthread=-1, num_parallel_tree=1, random_state=7, reg_alpha=0.464, reg_lambda=0.8571, scale_pos_weight=1, subsample=0.5213, silent=True, tree_method='exact', validate_parameters=1, verbosity=0)
for i in range(1, 2):
    pipeline = Pipeline(steps=[('impute', IterativeImputer(max_iter=4)), ('model', model)])
    (xtrain, xvalid, ytrain, yvalid) = train_test_split(_input1.drop('SalePrice', axis=1), y, test_size=0.2)