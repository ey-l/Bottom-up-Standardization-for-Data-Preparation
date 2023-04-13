import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
len(_input1.select_dtypes(include='object').columns)
len(_input1.select_dtypes(exclude='object').columns)
_input1.describe()
j = 1
a = len(_input1.select_dtypes(exclude='object').columns) // 3 + 1
plt.figure(figsize=(20, a * 5))
for i in _input1.select_dtypes(exclude='object'):
    plt.subplot(a, 3, j)
    sns.distplot(_input1[i])
    plt.axvline(_input1[i].min(), c='b', label='min')
    plt.axvline(_input1[i].quantile(0.25), c='orange', label='25%')
    plt.axvline(_input1[i].median(), c='y', label='median')
    plt.axvline(_input1[i].mean(), c='g', label='mean')
    plt.axvline(_input1[i].quantile(0.75), c='brown', label='75%')
    plt.axvline(_input1[i].max(), c='r', label='max')
    j = j + 1
plt.legend()
a = len(_input1.select_dtypes(include='object').columns) // 3 + 1
j = 1
plt.figure(figsize=(20, a * 5))
for i in _input1.select_dtypes(include='object'):
    plt.subplot(a, 3, j)
    sns.boxplot(y=_input1['SalePrice'], x=_input1[i])
    j = j + 1
a = len(_input1.select_dtypes(include='object').columns) // 3 + 1
j = 1
plt.figure(figsize=(20, a * 5))
for i in _input1.select_dtypes(include='object'):
    plt.subplot(a, 3, j)
    sns.violinplot(y=_input1['SalePrice'], x=_input1[i])
    j = j + 1
plt.figure(figsize=(25, 20))
sns.heatmap(_input1.corr(), annot=True, fmt='.1f')
_input1.head()
(_input1.shape, _input0.shape)
_input1.dtypes
na = _input1.isna().sum() / len(_input1)
na[na > 0.5]
_input1 = _input1.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
_input0 = _input0.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
a = _input1.corr()['SalePrice']
cols = list(a[a > 0.6].index)
print(cols)
print(a[a > 0.6])
sns.pairplot(data=_input1[cols + ['ExterQual']], hue='ExterQual')
plt.figure(figsize=(20, 10))
sns.jointplot(x=_input1['GrLivArea'], y=_input1['SalePrice'], hue=_input1['KitchenQual'])
obj = list(_input1.select_dtypes(include='object').columns)
len(obj)
for i in obj:
    _input1[i] = _input1[i].fillna('aaaaaaaa')
    _input1[i] = _input1[i].astype(str)
for i in obj:
    _input0[i] = _input0[i].fillna('aaaaaaaa')
    _input0[i] = _input0[i].astype(str)
import scipy.stats as stats
col = []
pval = []
for i in obj:
    a = [_input1[_input1[i] == j]['SalePrice'] for j in _input1[i].unique()]
    print(i, len(a), len(_input1[i].unique()), end=' | ')
    col.append(i)
    pval.append(stats.f_oneway(*a).pvalue)
anval = pd.DataFrame({'col': col, 'pval': pval})
obj_col = list(anval.sort_values('pval')[0:65]['col'])
len(obj_col)
(_input1.shape, _input0.shape)
c = _input1.corr()['SalePrice']
num_col = list(c[abs(c) > 0.0].index)
len(num_col)
col = num_col + obj_col
_input1 = _input1[col]
_input0 = _input0[[i for i in col if i != 'SalePrice']]
(_input1.shape, _input0.shape)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in obj_col:
    _input1[i] = label_encoder.fit_transform(_input1[i])
    _input0[i] = label_encoder.fit_transform(_input0[i])
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
_input1 = _input1.astype(float)
_input1 = pd.DataFrame(imp.fit_transform(_input1), columns=list(_input1.columns))
_input0 = pd.DataFrame(imp.fit_transform(_input0), columns=list(_input0.columns))
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
X = _input1.drop(columns=['SalePrice'])
y = _input1['SalePrice']
from sklearn.preprocessing import MinMaxScaler
sca = MinMaxScaler()
X = pd.DataFrame(sca.fit_transform(X), columns=list(X.columns))
_input0 = pd.DataFrame(sca.fit_transform(_input0), columns=list(_input0.columns))
from sklearn.model_selection import GridSearchCV
model = CatBoostRegressor(iterations=10000, verbose=False)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)