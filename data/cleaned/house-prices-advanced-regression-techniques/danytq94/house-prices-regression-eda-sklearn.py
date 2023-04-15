import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
import seaborn as sb
sb.set()

def nan_values(raw_data, typ):
    for i in raw_data.columns:
        if str(raw_data[i].dtypes) == str(typ):
            for j in range(len(raw_data[i].unique())):
                if str(raw_data[i].unique()[j]) == '':
                    print(i)

def unique_columns(raw_data, typ):
    s = 0
    for i in raw_data.columns:
        if str(raw_data[i].dtypes) == str(typ):
            if s == 0:
                print('<Unique variables of', typ, 'features>')
                print('#\t   Column\t       Count\t     variables')
                print('-\t  -------\t      ------\t    ----------\n')
            s += 1
            print(s, '\t', i, ' \t\t', len(raw_data[i].unique()), ' \t', raw_data[i].unique())

def null_columns(raw_data):
    print('<No of null rows in each column>\n', 'Columns\t No\t Type\n', '--------\t ----    -----')
    for i in raw_data.columns:
        if raw_data[i].isnull().sum() != 0:
            print(i, '  \t', raw_data[i].isnull().sum(), '\t', raw_data[i].dtype)
    print('\nTotal rows:', len(raw_data))

def columns_type(raw_data, typ):
    columns = []
    s = 0
    for i in raw_data.columns:
        if str(raw_data[i].dtypes) == str(typ):
            if s == 0:
                print('<', typ, 'columns>')
                print('#\t   Column\t')
                print('-\t  -------\t')
            s += 1
            print(s, '\t', i)
            columns.append(i)

def unique_col_comparison(data1, data2, typ):
    s = 0
    for i in data1.columns:
        if str(data1[i].dtypes) == str(typ):
            if s == 0:
                print('<Unique variables of', typ, 'features>')
                print('#\t   Column\t       Count Train\t    Count Test')
                print('-\t  -------\t      ---------\t\t    ---------\n')
            s += 1
            if len(data1[i].unique()) != len(data2[i].unique()):
                print(s, '\t', i, ' \t\t', len(data1[i].unique()), ' \t\t\t', len(data2[i].unique()))

def scaleMinMax(data):
    for col in num_col + ord_col:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
all_rmse_train = {}
all_acc_train = {}

def models_fit(x_train, y_train, x_test):
    all_regr_models = [LinearRegression(), Ridge(), RidgeCV(), LassoCV(max_iter=100000), ElasticNetCV()]
    for model in all_regr_models:
        model_name = model.__class__.__name__
        print('♦ ', model_name)