import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submit = pd.DataFrame(test['Id'])
test = test.set_index('Id')
train.select_dtypes(exclude='object').hist(bins=10, figsize=(25, 19))

plt.figure(figsize=(25, 19))
sns.heatmap(train.corr('spearman'), annot=True)

null = train.loc[:, train.isnull().sum() > 500]
train = train.drop(null, axis=1)
test = test.drop(null, axis=1)
encode = LabelEncoder()
for i in train.select_dtypes(include='object'):
    train[i] = encode.fit_transform(train[i])
    test[i] = encode.fit_transform(test[i])
model = XGBRegressor(base_score=0.4, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4603, gamma=0.05, gpu_id=-1, importance_type='gain', interaction_constraints='', learning_rate=0.05, max_delta_step=0, max_depth=3, min_child_weight=1.7817, monotone_constraints='()', n_estimators=2200, n_jobs=4, nthread=-1, num_parallel_tree=1, random_state=7, reg_alpha=0.464, reg_lambda=0.8571, scale_pos_weight=1, subsample=0.5213, silent=True, tree_method='exact', validate_parameters=1, verbosity=0)
pipeline = Pipeline(steps=[('impute', IterativeImputer(max_iter=4)), ('model', model)])