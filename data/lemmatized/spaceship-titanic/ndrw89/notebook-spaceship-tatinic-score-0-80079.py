import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input0.shape)
_input0.head()
_input1['Transported'] = _input1['Transported'].astype(int)
print(_input1.shape)
print(_input1.isnull().sum())
obj_col_name = _input1.select_dtypes('object').columns
print(obj_col_name)
obj_col_name = obj_col_name.drop('Name')
obj_col = _input1[obj_col_name]
obj_cols_with_missing = [col for col in obj_col if _input1[col].isnull().any()]
obj_cols_with_missing
print(_input1[obj_cols_with_missing].isnull().sum())
print(_input0[obj_cols_with_missing].isnull().sum())
data_test = _input1[obj_cols_with_missing].fillna('unknown')
target_test = _input0[obj_cols_with_missing].fillna('unknown')
print(data_test.shape)
print(target_test.shape)
_input1 = _input1.drop(obj_cols_with_missing, axis=1)
_input0 = _input0.drop(obj_cols_with_missing, axis=1)
_input1 = pd.concat([_input1, data_test], axis=1)
_input0 = pd.concat([_input0, target_test], axis=1)
print(_input1.shape)
print(_input0.shape)
low_cardinality_cols = [col for col in obj_cols_with_missing if data_test[col].nunique() < 10]
print(low_cardinality_cols)
high_cardinality_cols = list(set(obj_cols_with_missing) - set(low_cardinality_cols))
print(high_cardinality_cols)
test = _input1['Cabin'].str.split('/', n=2, expand=True)
test = test.fillna('unknown')
_input1['c1'] = test[0]
_input1['c2'] = test[1]
_input1['c3'] = test[2]
t_test = _input0['Cabin'].str.split('/', n=2, expand=True)
t_test = t_test.fillna('unknown')
_input0['c1'] = t_test[0]
_input0['c2'] = t_test[1]
_input0['c3'] = t_test[2]
_input1['VIP'] = _input1['VIP'].replace('unknown', 'False')
_input1['CryoSleep'] = _input1['CryoSleep'].replace('unknown', 'False')
_input1['VIP'] = _input1['VIP'].astype(bool)
_input1['CryoSleep'] = _input1['CryoSleep'].astype(bool)
_input1['VIP'] = _input1['VIP'].astype(int)
_input1['CryoSleep'] = _input1['CryoSleep'].astype(int)
_input0['VIP'] = _input0['VIP'].replace('unknown', 'False')
_input0['CryoSleep'] = _input0['CryoSleep'].replace('unknown', 'False')
_input0['VIP'] = _input0['VIP'].astype(bool)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(bool)
_input0['VIP'] = _input0['VIP'].astype(int)
_input0['CryoSleep'] = _input0['CryoSleep'].astype(int)
cabin_cardinality_cols = [col for col in ['c1', 'c2', 'c3'] if _input1[col].nunique() < 10]
low_cardinality_cols = cabin_cardinality_cols + low_cardinality_cols
print(low_cardinality_cols)
_input1.head()
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(_input1[['c1', 'c3', 'HomePlanet', 'Destination']]))
OH_cols_target = pd.DataFrame(OH_encoder.fit_transform(_input0[['c1', 'c3', 'HomePlanet', 'Destination']]))
OH_cols_train.index = _input1.index
OH_cols_target.index = _input0.index
OH_X_train = pd.concat([_input1, OH_cols_train], axis=1)
OH_X_test = pd.concat([_input0, OH_cols_target], axis=1)
OH_X_train = OH_X_train.drop(['c1', 'c3', 'HomePlanet', 'Destination'], axis=1)
OH_X_test = OH_X_test.drop(['c1', 'c3', 'HomePlanet', 'Destination'], axis=1)
print(OH_X_train.shape)
print(OH_X_test.shape)
OH_X_train.head()
oh_obj_col_name = OH_X_train.select_dtypes('object').columns
print(oh_obj_col_name)
OH_X_train['c2'] = OH_X_train['c2'].replace('unknown', '0')
OH_X_train['c2'] = OH_X_train['c2'].astype(int)
OH_X_test['c2'] = OH_X_test['c2'].replace('unknown', '0')
OH_X_test['c2'] = OH_X_test['c2'].astype(int)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
y = OH_X_train.Transported
X = OH_X_train.drop(['Transported', 'Name', 'Cabin', 'PassengerId'], axis=1)
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
OH_X_test = OH_X_test.drop(['Name', 'Cabin', 'PassengerId'], axis=1)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc_params = {'gamma': [0.5, 1, 1.5], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'max_depth': [3, 4, 5], 'n_estimators': [80, 100, 150]}
xgbc_cv_model = GridSearchCV(xgbc, xgbc_params, cv=10, n_jobs=-1)