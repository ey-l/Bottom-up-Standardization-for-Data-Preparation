import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1[_input1.columns[1:]].corr()['SalePrice'][:]
_input1.describe()
_input1.isnull().sum().sort_values(ascending=False)[0:20]
_input0.isnull().sum().sort_values(ascending=False)[0:40]
for col in _input1:
    if _input1[col].dtype == 'object':
        _input1[col] = _input1[col].fillna(_input1[col].mode()[0])
    else:
        _input1[col] = _input1[col].fillna(round(_input1[col].mean()), inplace=False)
for col in _input0:
    if _input0[col].dtype == 'object':
        _input0[col] = _input0[col].fillna(_input0[col].mode()[0])
    else:
        _input0[col] = _input0[col].fillna(round(_input0[col].mean()), inplace=False)
_input1.isnull().any().any()
_input0.isnull().any().any()
_input1 = _input1.drop(columns=['Id'], axis=1, inplace=False)
_input0 = _input0.drop(columns=['Id'], axis=1, inplace=False)
import seaborn as sns
sns.pairplot(_input1, x_vars=['SalePrice'], y_vars=['TotalBsmtSF', 'YearBuilt', '1stFlrSF', 'GrLivArea', '2ndFlrSF', 'GarageArea'], corner=True)

def cat_onehotencoder(df_concat):
    df_temp = df_concat
    for col in df_temp:
        if df_temp[col].dtype == 'object':
            df1 = pd.get_dummies(df_concat[col], drop_first=True)
            df_concat = df_concat.drop([col], axis=1, inplace=False)
            df_concat = pd.concat([df_concat, df1], axis=1)
    return df_concat
y = _input1.iloc[:, -1].values
df_t = _input1
y
_input1 = _input1.drop(columns=['SalePrice'], axis=0, inplace=False)
df_concat = pd.concat([_input1, _input0], axis=0)
df_final = cat_onehotencoder(df_concat)
df_final = df_final.loc[:, ~df_final.columns.duplicated()]
df_final.shape
df_final
import seaborn as sns
correlations = _input1[_input1.columns].corr(method='pearson')
sns.heatmap(correlations, cmap='YlGnBu', annot=True)
import heapq
print('Absolute overall correlations')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()
print(correlations_abs_sum, '\n')
print('Weakest correlations')
print('-' * 30)
print(correlations_abs_sum.nsmallest(5))
train = df_final.iloc[:1460, :]
test = df_final.iloc[1460:, :]
X = train.iloc[:, :].values
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
gbreg = GradientBoostingRegressor()
parameters = {'n_estimators': [100, 200, 300, 600], 'max_depth': [3, 4, 6, 7]}
gbreg = GridSearchCV(gbreg, param_grid=parameters)