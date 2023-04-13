import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor, ExtraTreesRegressor
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1
_input1.corr()['SalePrice']

def string_remover(df, list1=[], drop=[]):
    a = df.select_dtypes(include='object')
    for i in a.columns:
        for x in a.index:
            try:
                c = list1.index(a[i].iloc[x:x + 1][x])
                a[i].iloc[x:x + 1][x] = c
            except:
                list1.append(a[i].iloc[x:x + 1][x])
                a[i].iloc[x:x + 1][x] = len(list1) - 1
    a.fillna(len(list1))
    d = df.select_dtypes(exclude='object').fillna(0)
    try:
        return pd.concat([d, a], axis=1).fillna(0).drop([drop], axis=1)
    except:
        return pd.concat([d, a], axis=1).fillna(0)
b = []
_input1 = string_remover(_input1, list1=b)
_input0 = string_remover(_input0, list1=b)
(X_train, X_test, y_train, y_test) = train_test_split(_input1.drop('SalePrice', axis=1), _input1['SalePrice'], test_size=0.33, random_state=500)
forest = ExtraTreesRegressor()
from sklearn.feature_selection import SelectFromModel
select_from_model = SelectFromModel(forest)