import numpy as np
import pandas as pd
import math
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()

def transform_col_cabin(X_raw):
    list_index = []
    list_deck = []
    list_num = []
    list_side = []
    for i in range(len(_input1.index.values)):
        splitted = str(_input1['Cabin'][i]).split('/')
        if splitted == float('NaN'):
            list_deck.append(float('NaN'))
            list_num.append(float('NaN'))
            list_side.append(float('NaN'))
            continue
        elif len(splitted) != 3:
            list_deck.append(float('NaN'))
            list_num.append(float('NaN'))
            list_side.append(float('NaN'))
            continue
        else:
            list_deck.append(splitted[0])
            list_num.append(int(splitted[1]))
            list_side.append(splitted[2])
    new_cabin_columns = pd.DataFrame({'deck': list_deck, 'num': list_num, 'side': list_side}, index=_input1.index.values.tolist())
    df = pd.concat([_input1, new_cabin_columns], axis=1)
    return df
X_raw_transformed = transform_col_cabin(_input1)
X_raw_test_transformed = transform_col_cabin(_input0)

def remove_specified_cols(X_raw_transformed, X_raw_test_transformed):
    cols_drop = ['Cabin', 'Name']
    y = X_raw_transformed['Transported']
    X_prepared = X_raw_transformed.drop(cols_drop, axis=1)
    X_prepared = X_prepared.drop('Transported', axis=1)
    X_test_prepared = X_raw_test_transformed.drop(cols_drop, axis=1)
    return (y, X_prepared, X_test_prepared)
(y, X_prepared, X_test_prepared) = remove_specified_cols(X_raw_transformed, X_raw_test_transformed)

def define_col_types(X):
    cols_cat = [cname for cname in X.columns if X[cname].dtype == 'object']
    cols_num = [cname for cname in X.columns if X[cname].dtype in ['float64']]
    return (cols_num, cols_cat)
(numerical_cols, categorical_cols) = define_col_types(X_prepared)
print(numerical_cols)
print(categorical_cols)
X_prepared.info()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
num_imputer = SimpleImputer()
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer_test = SimpleImputer()
cat_imputer_test = SimpleImputer(strategy='most_frequent')
imputed_X_num = pd.DataFrame(num_imputer.fit_transform(X_prepared[numerical_cols]))
imputed_X_cat = pd.DataFrame(cat_imputer.fit_transform(X_prepared[categorical_cols]))
imputed_X_test_num = pd.DataFrame(num_imputer_test.fit_transform(X_test_prepared[numerical_cols]))
imputed_X_test_cat = pd.DataFrame(cat_imputer_test.fit_transform(X_test_prepared[categorical_cols]))
imputed_X_num.columns = X_prepared[numerical_cols].columns
imputed_X_cat.columns = X_prepared[categorical_cols].columns
imputed_X_test_num.columns = X_test_prepared[numerical_cols].columns
imputed_X_test_cat.columns = X_test_prepared[categorical_cols].columns
X_full_ready = imputed_X_num.join(imputed_X_cat, how='outer')
X_test_ready = imputed_X_test_num.join(imputed_X_test_cat, how='outer')
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X_full_ready, y, test_size=0.25, random_state=0)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = XGBClassifier(booster='gbtree', learning_rate=0.02, n_estimators=200, n_jobs=4)