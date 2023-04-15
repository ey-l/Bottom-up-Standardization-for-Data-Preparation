


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head(5)
test.head(5)
target_train = train['SalePrice']
train = train.drop(['Id', 'SalePrice'], axis=1)
test = test.drop('Id', axis=1)
data0 = pd.concat([train, test])
data0 = data0.reset_index(drop=True)
data0.isna().sum()
data1 = data0.copy()
data_num = data0.select_dtypes(np.number)
num_columns = data_num.loc[:, data_num.isna().sum() > 0].columns
num_columns

def knn_impute(X, column):
    from sklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor()
    X_num_temp = X.select_dtypes(np.number)
    X_num = X_num_temp.loc[:, X_num_temp.isna().sum() == 0]
    ind_fit = np.where(X[column].isna() == False)[0]
    ind_impute = np.where(X[column].isna())[0]