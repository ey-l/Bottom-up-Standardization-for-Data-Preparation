import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')

def transform_col_cabin(X):
    list_index = []
    list_deck = []
    list_num = []
    list_side = []
    for i in range(len(X.index.values)):
        splitted = str(X['Cabin'][i]).split('/')
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
    new_cabin_columns = pd.DataFrame({'deck': list_deck, 'num': list_num, 'side': list_side}, index=X.index.values.tolist())
    df = pd.concat([X, new_cabin_columns], axis=1)
    return df
_input1 = transform_col_cabin(_input1)
_input0 = transform_col_cabin(_input0)
cols_drop = ['Cabin', 'Name']
X = _input1.drop(cols_drop, axis=1)
y = _input1['Transported']
_input0 = _input0.drop(cols_drop, axis=1)

def define_col_types(X):
    cols_cat = [cname for cname in X.columns if X[cname].dtype == 'object']
    cols_num = [cname for cname in X.columns if X[cname].dtype in ['float64']]
    return (cols_num, cols_cat)
(numerical_cols, categorical_cols) = define_col_types(X)
(numerical_cols_test, categorical_cols_test) = define_col_types(_input0)
print(numerical_cols)
print(categorical_cols)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
num_imputer = SimpleImputer()
cat_imputer = SimpleImputer(strategy='most_frequent')
imputed_X_num = pd.DataFrame(num_imputer.fit_transform(_input1[numerical_cols]))
imputed_X_cat = pd.DataFrame(cat_imputer.fit_transform(_input1[categorical_cols]))
imputed_test_num = pd.DataFrame(num_imputer.fit_transform(_input0[numerical_cols]))
imputed_test_cat = pd.DataFrame(cat_imputer.fit_transform(_input0[categorical_cols]))
imputed_X_num.columns = X[numerical_cols].columns
imputed_X_cat.columns = X[categorical_cols].columns
imputed_test_num.columns = _input0[numerical_cols_test].columns
imputed_test_cat.columns = _input0[categorical_cols_test].columns
X = imputed_X_num.join(imputed_X_cat, how='outer')
_input0 = imputed_test_num.join(imputed_test_cat, how='outer')
X.info()
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.125, random_state=0)

def score_dataset(model, X_train, X_valid, y_train, y_valid):
    model = model()