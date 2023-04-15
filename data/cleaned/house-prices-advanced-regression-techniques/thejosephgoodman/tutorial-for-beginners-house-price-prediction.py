import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from scipy import stats

def score_dataset(X, y, model=XGBRegressor(random_state=0)):
    score = cross_val_score(model, X, y, scoring='neg_mean_squared_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

def drop_outliers(col, zscore=3):
    return col[np.abs(stats.zscore(col)) < zscore]
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
cols_to_drop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
train = train.drop(cols_to_drop, axis=1)
X = train.drop(['SalePrice'], axis=1)
y = train.SalePrice
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns
cat_X = X[cat_cols]
num_X = X[num_cols]
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_X_prep = onehot_encoder.fit_transform(cat_X)
for col in num_X:
    num_X[col].fillna(num_X[col].mean(), inplace=True)
for col in num_X:
    num_X.loc[:, col] = drop_outliers(num_X[col])
num_X = num_X.dropna(axis=0)
X_prep = np.concatenate((num_X, cat_X_prep[num_X.index]), axis=1)
y_prep = y[num_X.index]
score_dataset(X_prep, y_prep, XGBRegressor(random_state=0, learning_rate=0.1, n_jobs=-1))
X_test = test.drop(columns=cols_to_drop, axis=1)
cat_X_test = X_test[cat_cols]
num_X_test = X_test[num_cols]
cat_X_test_prep = onehot_encoder.transform(cat_X_test)
for col in num_X_test:
    num_X_test[col].fillna(num_X_test[col].mean(), inplace=True)
X_test_prep = np.concatenate((num_X_test, cat_X_test_prep), axis=1)
my_model = XGBRegressor(learning_rate=0.1, n_jobs=-1)