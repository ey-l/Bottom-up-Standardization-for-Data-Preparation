import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data.head()
target = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(target.shape)
target.head()
data['Transported'] = data['Transported'].astype(int)
print(data.shape)
print(data.isnull().sum())
obj_col_name = data.select_dtypes('object').columns
print(obj_col_name)
obj_col_name = obj_col_name.drop('Name')
obj_col = data[obj_col_name]
obj_cols_with_missing = [col for col in obj_col if data[col].isnull().any()]
obj_cols_with_missing
print(data[obj_cols_with_missing].isnull().sum())
print(target[obj_cols_with_missing].isnull().sum())
data_test = data[obj_cols_with_missing].fillna('unknown')
target_test = target[obj_cols_with_missing].fillna('unknown')
print(data_test.shape)
print(target_test.shape)
data = data.drop(obj_cols_with_missing, axis=1)
target = target.drop(obj_cols_with_missing, axis=1)
data = pd.concat([data, data_test], axis=1)
target = pd.concat([target, target_test], axis=1)
print(data.shape)
print(target.shape)
low_cardinality_cols = [col for col in obj_cols_with_missing if data_test[col].nunique() < 10]
print(low_cardinality_cols)
high_cardinality_cols = list(set(obj_cols_with_missing) - set(low_cardinality_cols))
print(high_cardinality_cols)
test = data['Cabin'].str.split('/', n=2, expand=True)
test = test.fillna('unknown')
data['c1'] = test[0]
data['c2'] = test[1]
data['c3'] = test[2]
t_test = target['Cabin'].str.split('/', n=2, expand=True)
t_test = t_test.fillna('unknown')
target['c1'] = t_test[0]
target['c2'] = t_test[1]
target['c3'] = t_test[2]
data['VIP'] = data['VIP'].replace('unknown', 'False')
data['CryoSleep'] = data['CryoSleep'].replace('unknown', 'False')
data['VIP'] = data['VIP'].astype(bool)
data['CryoSleep'] = data['CryoSleep'].astype(bool)
data['VIP'] = data['VIP'].astype(int)
data['CryoSleep'] = data['CryoSleep'].astype(int)
target['VIP'] = target['VIP'].replace('unknown', 'False')
target['CryoSleep'] = target['CryoSleep'].replace('unknown', 'False')
target['VIP'] = target['VIP'].astype(bool)
target['CryoSleep'] = target['CryoSleep'].astype(bool)
target['VIP'] = target['VIP'].astype(int)
target['CryoSleep'] = target['CryoSleep'].astype(int)
cabin_cardinality_cols = [col for col in ['c1', 'c2', 'c3'] if data[col].nunique() < 10]
low_cardinality_cols = cabin_cardinality_cols + low_cardinality_cols
print(low_cardinality_cols)
data.head()
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(data[['c1', 'c3', 'HomePlanet', 'Destination']]))
OH_cols_target = pd.DataFrame(OH_encoder.fit_transform(target[['c1', 'c3', 'HomePlanet', 'Destination']]))
OH_cols_train.index = data.index
OH_cols_target.index = target.index
OH_X_train = pd.concat([data, OH_cols_train], axis=1)
OH_X_test = pd.concat([target, OH_cols_target], axis=1)
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