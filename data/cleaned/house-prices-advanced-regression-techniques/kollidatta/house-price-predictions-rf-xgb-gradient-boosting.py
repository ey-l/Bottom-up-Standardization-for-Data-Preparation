import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train[df_train.columns[1:]].corr()['SalePrice'][:]
df_train.describe()
df_train.isnull().sum().sort_values(ascending=False)[0:20]
df_test.isnull().sum().sort_values(ascending=False)[0:40]
for col in df_train:
    if df_train[col].dtype == 'object':
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    else:
        df_train[col].fillna(round(df_train[col].mean()), inplace=True)
for col in df_test:
    if df_test[col].dtype == 'object':
        df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
    else:
        df_test[col].fillna(round(df_test[col].mean()), inplace=True)
df_train.isnull().any().any()
df_test.isnull().any().any()
df_train.drop(columns=['Id'], axis=1, inplace=True)
df_test.drop(columns=['Id'], axis=1, inplace=True)
import seaborn as sns
sns.pairplot(df_train, x_vars=['SalePrice'], y_vars=['TotalBsmtSF', 'YearBuilt', '1stFlrSF', 'GrLivArea', '2ndFlrSF', 'GarageArea'], corner=True)

def cat_onehotencoder(df_concat):
    df_temp = df_concat
    for col in df_temp:
        if df_temp[col].dtype == 'object':
            df1 = pd.get_dummies(df_concat[col], drop_first=True)
            df_concat.drop([col], axis=1, inplace=True)
            df_concat = pd.concat([df_concat, df1], axis=1)
    return df_concat
y = df_train.iloc[:, -1].values
df_t = df_train
y
df_train.drop(columns=['SalePrice'], axis=0, inplace=True)
df_concat = pd.concat([df_train, df_test], axis=0)
df_final = cat_onehotencoder(df_concat)
df_final = df_final.loc[:, ~df_final.columns.duplicated()]
df_final.shape
df_final
import seaborn as sns
correlations = df_train[df_train.columns].corr(method='pearson')
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