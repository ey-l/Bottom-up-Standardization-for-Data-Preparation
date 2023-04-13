import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
import warnings
print('Python: {}'.format(sys.version))
print('numpy: {}'.format(np.__version__))
print('pandas: {}'.format(pd.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('\n\n============== [ DataFrames ] ==================\n\n')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def warns(*args, **kwargs):
    pass
warnings.warn = warns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train Shape: ', _input1.shape, '\nTest Shape: ', _input0.shape)
_input1.head()
sns.distplot(_input1['SalePrice'])
sns.displot(_input1['SalePrice'])
maxPrice = _input1['SalePrice'].max()
print('Maximum Sale Price: ', maxPrice)
data = pd.concat([_input1['SalePrice'], _input1['OverallQual']], axis=1)
(f, ax) = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=_input1['OverallQual'], y='SalePrice', data=data)
fig.axis(ymin=0, ymax=maxPrice)

def get_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
train_missing_values = get_missing_values(_input1)
train_missing_values.head(20)
test_missing_values = get_missing_values(_input0)
test_missing_values.head(20)
(fig, ax) = plt.subplots(figsize=(12, 6))
plt.title('Missing Value in Train DataFrame', fontsize=18)
sns.heatmap(_input1.isnull())
(fig, ax) = plt.subplots(figsize=(12, 6))
plt.title('Missing Value in Test DataFrame', fontsize=18)
sns.heatmap(_input0.isnull())
le = LabelEncoder()
dtypeVal = ['object', 'int64', 'float64']

def fillna_encoder(df):
    for x in dtypeVal:
        if x == 'object':
            obj_df = df.select_dtypes(include=[x]).copy().reset_index()
            obj_df = obj_df.fillna('Not Listed', inplace=False)
            obj_df = obj_df.astype(str)
            obj_df = obj_df.apply(le.fit_transform)
        elif x == 'int64':
            int_df = df.select_dtypes(include=[x]).copy().reset_index()
            int_df = int_df.fillna(0, inplace=False)
        elif x == 'float64':
            float_df = df.select_dtypes(include=[x]).copy().reset_index()
            float_df = float_df.fillna(0, inplace=False)
    all_df = obj_df.merge(int_df, on='index').merge(float_df, on='index')
    return all_df
xtrain = fillna_encoder(_input1)
xtrain
xtest = fillna_encoder(_input0)
xtest
(fig, ax) = plt.subplots(figsize=(12, 6))
plt.title('Missing Value in Train DataFrame', fontsize=18)
sns.heatmap(xtrain.isnull())
(fig, ax) = plt.subplots(figsize=(12, 6))
plt.title('Missing Value in Test DataFrame', fontsize=18)
sns.heatmap(xtest.isnull())
print('Train DataFrame Null Values: ', xtrain.isnull().sum().sum(), '\nTest DataFrame Null Values: ', xtest.isnull().sum().sum())
id = xtest.Id
y = xtrain['SalePrice'].values
X = xtrain.drop(['Id', 'index', 'SalePrice'], axis=1)
X_test = xtest.drop(['Id', 'index'], axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.1, random_state=1)
(X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape)
from sklearn.preprocessing import StandardScaler, MinMaxScaler