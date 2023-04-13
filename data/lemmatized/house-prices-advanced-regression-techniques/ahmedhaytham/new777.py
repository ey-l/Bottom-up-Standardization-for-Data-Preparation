import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(len(_input1.columns))
print(len(_input0.columns))
_input1.head()
print(_input1.isnull().sum().sort_values(ascending=False))
_input1 = _input1.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
_input1.describe()
categorical_data = _input1.select_dtypes(['object']).columns
_input1[categorical_data] = _input1[categorical_data].fillna(_input1[categorical_data].mode().iloc[0])
_input1[categorical_data].mode()
print(_input1.isnull().sum().sort_values(ascending=False))
trai = _input1.drop('Id', axis=1)
numerical_data = trai.select_dtypes(['float64', 'int64']).columns
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=15)
for i in numerical_data:
    _input1[i] = imputer.fit_transform(_input1[[i]])
_input1.hist(figsize=(20, 20), bins=20)
category_columns = _input1.select_dtypes(['object']).columns
print(category_columns)
_input1[category_columns] = _input1[category_columns].astype('category').apply(lambda x: x.cat.codes)
float_columns = _input1.select_dtypes(['float64']).columns
print(float_columns)
_input1['LotFrontage'] = pd.to_numeric(_input1['LotFrontage'], errors='coerce')
_input1['MasVnrArea'] = pd.to_numeric(_input1['MasVnrArea'], errors='coerce')
_input1['GarageYrBlt'] = pd.to_numeric(_input1['GarageYrBlt'], errors='coerce')
_input1['SalePrice'] = pd.to_numeric(_input1['SalePrice'], errors='coerce')
_input1 = _input1.astype('float64')
sns.displot(_input1['SalePrice'])
correlation_matrix = _input1.corr()
correlation_matrix['SalePrice'].sort_values(ascending=False)
correlation_num = 30
correlation_cols = correlation_matrix.nlargest(correlation_num, 'SalePrice')['SalePrice'].index
correlation_mat_sales = np.corrcoef(_input1[correlation_cols].values.T)
sns.set(font_scale=1.25)
(f, ax) = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(correlation_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=correlation_cols.values, xticklabels=correlation_cols.values)
y = _input1['SalePrice']
x = _input1.drop(columns=['SalePrice', 'Id'])
print(len(x.columns))
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=60, shuffle=True)
print(len(X_train))
print(len(X_test))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn.metrics as sm
forest_model = RandomForestRegressor(n_estimators=150, random_state=42)