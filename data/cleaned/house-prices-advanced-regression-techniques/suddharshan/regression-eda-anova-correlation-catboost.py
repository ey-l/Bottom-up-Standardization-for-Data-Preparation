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
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
len(df.select_dtypes(include='object').columns)
len(df.select_dtypes(exclude='object').columns)
df.describe()
j = 1
a = len(df.select_dtypes(exclude='object').columns) // 3 + 1
plt.figure(figsize=(20, a * 5))
for i in df.select_dtypes(exclude='object'):
    plt.subplot(a, 3, j)
    sns.distplot(df[i])
    plt.axvline(df[i].min(), c='b', label='min')
    plt.axvline(df[i].quantile(0.25), c='orange', label='25%')
    plt.axvline(df[i].median(), c='y', label='median')
    plt.axvline(df[i].mean(), c='g', label='mean')
    plt.axvline(df[i].quantile(0.75), c='brown', label='75%')
    plt.axvline(df[i].max(), c='r', label='max')
    j = j + 1
plt.legend()
a = len(df.select_dtypes(include='object').columns) // 3 + 1
j = 1
plt.figure(figsize=(20, a * 5))
for i in df.select_dtypes(include='object'):
    plt.subplot(a, 3, j)
    sns.boxplot(y=df['SalePrice'], x=df[i])
    j = j + 1
a = len(df.select_dtypes(include='object').columns) // 3 + 1
j = 1
plt.figure(figsize=(20, a * 5))
for i in df.select_dtypes(include='object'):
    plt.subplot(a, 3, j)
    sns.violinplot(y=df['SalePrice'], x=df[i])
    j = j + 1
plt.figure(figsize=(25, 20))
sns.heatmap(df.corr(), annot=True, fmt='.1f')
df.head()
(df.shape, df1.shape)
df.dtypes
na = df.isna().sum() / len(df)
na[na > 0.5]
df = df.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
df1 = df1.drop(columns=['PoolQC', 'Id', 'Alley', 'Fence', 'MiscFeature'])
a = df.corr()['SalePrice']
cols = list(a[a > 0.6].index)
print(cols)
print(a[a > 0.6])
sns.pairplot(data=df[cols + ['ExterQual']], hue='ExterQual')
plt.figure(figsize=(20, 10))
sns.jointplot(x=df['GrLivArea'], y=df['SalePrice'], hue=df['KitchenQual'])
obj = list(df.select_dtypes(include='object').columns)
len(obj)
for i in obj:
    df[i] = df[i].fillna('aaaaaaaa')
    df[i] = df[i].astype(str)
for i in obj:
    df1[i] = df1[i].fillna('aaaaaaaa')
    df1[i] = df1[i].astype(str)
import scipy.stats as stats
col = []
pval = []
for i in obj:
    a = [df[df[i] == j]['SalePrice'] for j in df[i].unique()]
    print(i, len(a), len(df[i].unique()), end=' | ')
    col.append(i)
    pval.append(stats.f_oneway(*a).pvalue)
anval = pd.DataFrame({'col': col, 'pval': pval})
obj_col = list(anval.sort_values('pval')[0:65]['col'])
len(obj_col)
(df.shape, df1.shape)
c = df.corr()['SalePrice']
num_col = list(c[abs(c) > 0.0].index)
len(num_col)
col = num_col + obj_col
df = df[col]
df1 = df1[[i for i in col if i != 'SalePrice']]
(df.shape, df1.shape)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in obj_col:
    df[i] = label_encoder.fit_transform(df[i])
    df1[i] = label_encoder.fit_transform(df1[i])
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
df = df.astype(float)
df = pd.DataFrame(imp.fit_transform(df), columns=list(df.columns))
df1 = pd.DataFrame(imp.fit_transform(df1), columns=list(df1.columns))
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
from sklearn.preprocessing import MinMaxScaler
sca = MinMaxScaler()
X = pd.DataFrame(sca.fit_transform(X), columns=list(X.columns))
df1 = pd.DataFrame(sca.fit_transform(df1), columns=list(df1.columns))
from sklearn.model_selection import GridSearchCV
model = CatBoostRegressor(iterations=10000, verbose=False)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)