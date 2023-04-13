import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew, norm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Data is loaded')
print('Train: ', _input1.shape[0], 'sales, and ', _input1.shape[1], 'features')
print('Test: ', _input0.shape[0], 'sales, and ', _input0.shape[1], 'features')
_input1.head()
_input1.info()
sns.set_style('white')
sns.set_color_codes(palette='deep')
(f, ax) = plt.subplots(figsize=(8, 7))
sns.distplot(_input1['SalePrice'], color='b')
ax.xaxis.grid(False)
ax.set(ylabel='Frequency')
ax.set(xlabel='SalePrice')
ax.set(title='SalePrice distribution')
sns.despine(trim=True, left=True)
print('Skewness: %f' % _input1['SalePrice'].skew())
corr = _input1.corr()
plt.subplots(figsize=(15, 12))
sns.heatmap(corr, vmax=0.9, cmap='coolwarm', square=True)
_input1.corr()['SalePrice'].sort_values()
sns.scatterplot(data=_input1, x='GrLivArea', y='SalePrice')
plt.axhline(y=300000, color='r')
plt.axvline(x=4550, color='r')
_input1[(_input1['GrLivArea'] > 4500) & (_input1['SalePrice'] < 300000)][['SalePrice', 'GrLivArea']]
sns.scatterplot(data=_input1, x='OverallQual', y='SalePrice')
plt.axvline(x=4.9, color='r')
plt.axhline(y=650000, color='r')
_input1[(_input1['OverallQual'] < 5) & (_input1['SalePrice'] < 200000)][['SalePrice', 'OverallQual']]

def missing_percent(train_df):
    nan_percent = 100 * (_input1.isnull().sum() / len(_input1))
    nan_percent = nan_percent[nan_percent > 0].sort_values()
    return nan_percent
nan_percent = missing_percent(_input1)
nan_percent
sns.set_style('whitegrid')
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar()
_input1['MSSubClass'] = _input1['MSSubClass'].apply(str)
_input1['YrSold'] = _input1['YrSold'].astype(str)
_input1['MoSold'] = _input1['MoSold'].astype(str)
sns.set_style('white')
sns.set_color_codes(palette='deep')
(f, ax) = plt.subplots(figsize=(8, 7))
sns.distplot(_input1['SalePrice'], color='b')
ax.xaxis.grid(False)
ax.set(ylabel='Frequency')
ax.set(xlabel='SalePrice')
ax.set(title='SalePrice distribution')
sns.despine(trim=True, left=True)
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
sns.set_style('white')
sns.set_color_codes(palette='deep')
(f, ax) = plt.subplots(figsize=(8, 7))
sns.distplot(_input1['SalePrice'], fit=norm, color='b')