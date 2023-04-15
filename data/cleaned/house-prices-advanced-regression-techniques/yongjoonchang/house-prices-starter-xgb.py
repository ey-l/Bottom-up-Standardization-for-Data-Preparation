import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name='', null_name=''):
    if id_name == '':
        df = df.reset_index().rename(columns={'index': 'id'})
        id_name = 'id'
    else:
        id_name = id_name
    if null_name != '':
        df[df == null_name] = np.nan
    (X_train, X_test) = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return (X_train, X_test, y_train, y_test)
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
(X_train, X_test, y_train, y_test) = exam_data_load(df, target='SalePrice', id_name='Id')
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.head()
y_train.head()
X_train.info()
column_num = X_train.select_dtypes(exclude=['object']).columns
column_cat = X_train.select_dtypes(include=['object']).columns
X_train[column_num].info()
X_train[column_num].describe()
from sklearn.impute import SimpleImputer
imp = SimpleImputer()
X_train_num = imp.fit_transform(X_train[column_num])
X_test_num = imp.transform(X_test[column_num])
X_train_num = pd.DataFrame(X_train_num, columns=column_num, index=X_train[column_num].index)
X_test_num = pd.DataFrame(X_test_num, columns=column_num, index=X_test[column_num].index)
column_num_outlier = []

def PrintMeanMax(column):
    max_mean_ratio = round(X_train[column].max() / X_train[column].mean(), 2)
    if max_mean_ratio > 10.0:
        column_num_outlier.append(column)
        print(column, 'max/mean:', max_mean_ratio)
for column in column_num:
    PrintMeanMax(column)

def DropOutlierViaIQR(column):
    iqr = X_train_num[column].quantile(0.75) - X_train_num[column].quantile(0.25)
    condi_low = X_train_num[column].quantile(0.25) - 1.5 * iqr <= X_train_num[column]
    condi_high = X_train_num[column] <= X_train_num[column].quantile(0.75) + 1.5 * iqr
    X_train_num[column] = X_train_num.loc[condi_low & condi_high, column]
    print(column, X_train_num.loc[condi_low & condi_high, column].shape)
for column in column_num_outlier:
    DropOutlierViaIQR(column)
X_train_num.shape
X_train_num = X_train_num.dropna()
y_train_num = y_train.loc[X_train_num.index]
X_train_num.shape
from sklearn.model_selection import train_test_split
(X_split_train, X_split_val, y_split_train, y_split_val) = train_test_split(X_train_num, y_train_num, test_size=0.1, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()