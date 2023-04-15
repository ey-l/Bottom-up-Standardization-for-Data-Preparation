import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import sklearn.metrics as sm
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
pd.options.display.max_rows = 1500
pd.options.display.max_columns = 100
train_set = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_set = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(len(train_set))
print(len(test_set))
train_set.head()
print(len(train_set.columns))
train_set.info()
train_set.describe()
print(train_set.dtypes)
train_set.isnull().sum().sort_values(ascending=False)

def remove_null_columns(dataset):
    dataset = dataset.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
    return dataset

def clean_categorical(dataset):
    dataset = pd.DataFrame(dataset)
    categorical_data = dataset.select_dtypes(['object']).columns
    dataset[categorical_data] = dataset[categorical_data].fillna(dataset[categorical_data].mode().iloc[0])
    dataset[categorical_data].mode()
    dataset[categorical_data] = dataset[categorical_data].astype('category').apply(lambda x: x.cat.codes)
    return dataset

def clean_numerical(dataset):
    numerical_data = dataset.select_dtypes(['float64', 'int64']).columns
    dataset[numerical_data] = dataset[numerical_data].fillna(dataset[numerical_data].mean())
    dataset[numerical_data].mean()
    return dataset

def clean_dataset(dataset):
    dataset = remove_null_columns(dataset)
    dataset = clean_categorical(dataset)
    dataset = clean_numerical(dataset)
    return dataset
train_set = clean_dataset(train_set)
train_set.head()
print(len(train_set.columns))
print(train_set.isnull().sum().sort_values(ascending=False))
train_set.head()
train_set.hist(figsize=(30, 30), bins=20)


def remove_irrelavant(dataset):
    dataset = dataset.drop(columns=['LowQualFinSF', 'PoolArea', 'MiscVal', '3SsnPorch', 'Heating', 'RoofMatl', 'Condition2', 'Utilities', 'Street'])
    return dataset
train_set = remove_irrelavant(train_set)
print(len(train_set.columns))
print(train_set.dtypes)

def convert_float(dataset):
    float_columns = dataset.select_dtypes(['float64']).columns
    dataset[float_columns] = dataset[float_columns].applymap(np.int64)
    dataset = dataset.astype('int64')
    return dataset
train_set = convert_float(train_set)
print(train_set.dtypes)
train_set['SalePrice'].describe()
sns.displot(train_set['SalePrice'])
correlation_matrix = train_set.corr()
correlation_matrix['SalePrice'].sort_values(ascending=False)
correlation_num = 30
correlation_cols = correlation_matrix.nlargest(correlation_num, 'SalePrice')['SalePrice'].index
correlation_mat_sales = np.corrcoef(train_set[correlation_cols].values.T)
sns.set(font_scale=1.25)
(f, ax) = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(correlation_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=correlation_cols.values, xticklabels=correlation_cols.values)

grph = pd.concat([train_set['SalePrice'], train_set['GrLivArea']], axis=1)
grph.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
train_set = train_set.drop(train_set[(train_set['GrLivArea'] > 4000) & (train_set['SalePrice'] < 300000)].index)
grph = pd.concat([train_set['SalePrice'], train_set['GrLivArea']], axis=1)
grph.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
train_set.shape
y = train_set['SalePrice']
x = train_set.drop(columns=['SalePrice', 'Id'])
print(len(train_set.columns))
(X_train, X_test, Y_train, Y_test) = train_test_split(x, y, test_size=0.3, random_state=60, shuffle=True)
print(len(X_train))
print(len(X_test))
linear_model = LinearRegression()