import pandas as pd
import numpy as np
import random
import math
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xg
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
corrmat = train_data.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

(fig, ax) = plt.subplots()
ax.scatter(x=train_data['GrLivArea'], y=train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

(fig, ax) = plt.subplots()
ax.scatter(x=train_data['OverallQual'], y=train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)

(fig, ax) = plt.subplots()
ax.scatter(x=train_data['GarageArea'], y=train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)

train_data = train_data.drop(train_data[(train_data['OverallQual'] > 9) & (train_data['SalePrice'] < 220000)].index)
train_data = train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index)
train_data = train_data.drop(train_data[(train_data['GarageArea'] > 1200) & (train_data['SalePrice'] < 300000)].index)
train_data.isnull().sum().sort_values(ascending=False).head(20)
Target = 'SalePrice'
train_data.dropna(axis=0, subset=[Target], inplace=True)
all_data = pd.concat([train_data, test_data], keys=['train', 'test'])
print('У тренировочного датасета {} рядов и {} признаков'.format(train_data.shape[0], train_data.shape[1]))
print('У тестового датасета {} рядов и {} признаков'.format(test_data.shape[0], test_data.shape[1]))
print('Объединённый датасет содержит в себе {} рядов и {} признаков'.format(all_data.shape[0], all_data.shape[1]))
all_data = all_data.drop(columns=['Id'], axis=1)

def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)
    temp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return temp.loc[temp['Total'] > 0]
missingValuesInfo(train_data)

def HandleMissingValues(df):
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in df.columns if df[cname].dtype == 'object']
    values = {}
    for a in cat_cols:
        values[a] = 'UNKNOWN'
    for a in num_cols:
        mean1 = df[a].mean()
        std1 = df[a].std()
        values[a] = random.randint(int(mean1 - std1), int(mean1 + std1))
    df.fillna(value=values, inplace=True)
HandleMissingValues(all_data)
all_data.isnull().sum().sum()

def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == 'object']

def PerformOneHotEncoding(df, columnsToEncode):
    return pd.get_dummies(df, columns=columnsToEncode)
cat_cols = getObjectColumnsList(all_data)
all_data = PerformOneHotEncoding(all_data, cat_cols)
all_data.head()
train_data = all_data.loc['train']
test_data = all_data.loc['test']
(train_data.shape, test_data.shape)
target = train_data['SalePrice']
train_data = train_data.drop(['SalePrice'], axis=1)
test_data = test_data.drop(['SalePrice'], axis=1)
(X, y) = (train_data, target)
(train_data.shape, test_data.shape)
gbr_reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber')