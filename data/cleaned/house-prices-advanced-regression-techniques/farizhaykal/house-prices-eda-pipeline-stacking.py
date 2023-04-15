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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Data is loaded')
print('Train: ', train_df.shape[0], 'sales, and ', train_df.shape[1], 'features')
print('Test: ', test_df.shape[0], 'sales, and ', test_df.shape[1], 'features')
train_df.head()
train_df.info()
sns.set_style('white')
sns.set_color_codes(palette='deep')
(f, ax) = plt.subplots(figsize=(8, 7))
sns.distplot(train_df['SalePrice'], color='b')
ax.xaxis.grid(False)
ax.set(ylabel='Frequency')
ax.set(xlabel='SalePrice')
ax.set(title='SalePrice distribution')
sns.despine(trim=True, left=True)

print('Skewness: %f' % train_df['SalePrice'].skew())
corr = train_df.corr()
plt.subplots(figsize=(15, 12))
sns.heatmap(corr, vmax=0.9, cmap='coolwarm', square=True)
train_df.corr()['SalePrice'].sort_values()
sns.scatterplot(data=train_df, x='GrLivArea', y='SalePrice')
plt.axhline(y=300000, color='r')
plt.axvline(x=4550, color='r')
train_df[(train_df['GrLivArea'] > 4500) & (train_df['SalePrice'] < 300000)][['SalePrice', 'GrLivArea']]
sns.scatterplot(data=train_df, x='OverallQual', y='SalePrice')
plt.axvline(x=4.9, color='r')
plt.axhline(y=650000, color='r')
train_df[(train_df['OverallQual'] < 5) & (train_df['SalePrice'] < 200000)][['SalePrice', 'OverallQual']]

def missing_percent(train_df):
    nan_percent = 100 * (train_df.isnull().sum() / len(train_df))
    nan_percent = nan_percent[nan_percent > 0].sort_values()
    return nan_percent
nan_percent = missing_percent(train_df)
nan_percent
sns.set_style('whitegrid')
missing = train_df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
train_df['MSSubClass'] = train_df['MSSubClass'].apply(str)
train_df['YrSold'] = train_df['YrSold'].astype(str)
train_df['MoSold'] = train_df['MoSold'].astype(str)
sns.set_style('white')
sns.set_color_codes(palette='deep')
(f, ax) = plt.subplots(figsize=(8, 7))
sns.distplot(train_df['SalePrice'], color='b')
ax.xaxis.grid(False)
ax.set(ylabel='Frequency')
ax.set(xlabel='SalePrice')
ax.set(title='SalePrice distribution')
sns.despine(trim=True, left=True)

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
sns.set_style('white')
sns.set_color_codes(palette='deep')
(f, ax) = plt.subplots(figsize=(8, 7))
sns.distplot(train_df['SalePrice'], fit=norm, color='b')