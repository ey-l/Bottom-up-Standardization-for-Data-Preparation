import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 100)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.shape
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
_input0.shape
all_data = pd.concat((_input1, _input0)).reset_index(drop=True)
all_data.shape
null_value = all_data.isna().sum().sort_values(ascending=False)
null_value[null_value > 0]
all_data = all_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'Id'], axis=1, inplace=False)
null_value = all_data.isna().sum().sort_values(ascending=False)
null_value[null_value > 0]
null_list = []
for i in null_value[null_value > 0].keys():
    null_list.append(i)
null_list.remove('SalePrice')
null_list
null_cat = []
null_num = []
for col in null_list:
    if all_data[col].dtype in [object, bool] and len(all_data[col].unique()) <= 50:
        null_cat.append(col)
    else:
        null_num.append(col)
(null_cat, null_num)
for col in null_cat:
    all_data[col] = all_data[col].fillna(max(list(all_data[col]), key=list(all_data[col]).count))
for col in null_num:
    all_data[col] = all_data[col].fillna(all_data[col].median())
null_value = all_data.isna().sum().sort_values(ascending=False)
null_value[null_value > 0]
categorical_col = []
for col in all_data.columns:
    if all_data[col].dtype in [object, bool] and len(all_data[col].unique()) <= 50:
        categorical_col.append(col)
categorical_col
for col in categorical_col:
    all_data[col] = all_data[col].astype('category').cat.codes
all_data.head()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto')
array_hot_encoded = ohe.fit_transform(all_data[categorical_col]).toarray()
data_hot_encoded = pd.DataFrame(array_hot_encoded, index=all_data.index)
data_other_cols = all_data.drop(columns=categorical_col)
all_data = pd.concat([data_hot_encoded, data_other_cols], axis=1)
all_data.head()
all_data.columns = all_data.columns.astype(str)
(_input1.shape, _input0.shape, all_data.shape)
_input1 = all_data.iloc[:1460, :]
_input0 = all_data.iloc[-1459:, :]
(_input1.shape, _input0.shape)
_input0 = _input0.drop(['SalePrice'], axis=1, inplace=False)
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
clf = LocalOutlierFactor(contamination=0.02)
outliers = clf.fit_predict(_input1)
train_data_cleaned = _input1[np.where(outliers == 1, True, False)]
x_train = train_data_cleaned.drop('SalePrice', axis=1)
y_train = train_data_cleaned[['SalePrice']]
(x_train.shape, y_train.shape)
from sklearn.model_selection import GridSearchCV
cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
from xgboost import XGBRegressor
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
xgb = XGBRegressor(**other_params)
xgb = GridSearchCV(estimator=xgb, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)