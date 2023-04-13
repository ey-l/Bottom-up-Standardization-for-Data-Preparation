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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
corrmat = _input1.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['OverallQual'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['GarageArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
_input1 = _input1.drop(_input1[(_input1['OverallQual'] > 9) & (_input1['SalePrice'] < 220000)].index)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
_input1 = _input1.drop(_input1[(_input1['GarageArea'] > 1200) & (_input1['SalePrice'] < 300000)].index)
_input1.isnull().sum().sort_values(ascending=False).head(20)
Target = 'SalePrice'
_input1 = _input1.dropna(axis=0, subset=[Target], inplace=False)
all_data = pd.concat([_input1, _input0], keys=['train', 'test'])
print('У тренировочного датасета {} рядов и {} признаков'.format(_input1.shape[0], _input1.shape[1]))
print('У тестового датасета {} рядов и {} признаков'.format(_input0.shape[0], _input0.shape[1]))
print('Объединённый датасет содержит в себе {} рядов и {} признаков'.format(all_data.shape[0], all_data.shape[1]))
all_data = all_data.drop(columns=['Id'], axis=1)

def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)
    temp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return temp.loc[temp['Total'] > 0]
missingValuesInfo(_input1)

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
    df = df.fillna(value=values, inplace=False)
HandleMissingValues(all_data)
all_data.isnull().sum().sum()

def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == 'object']

def PerformOneHotEncoding(df, columnsToEncode):
    return pd.get_dummies(df, columns=columnsToEncode)
cat_cols = getObjectColumnsList(all_data)
all_data = PerformOneHotEncoding(all_data, cat_cols)
all_data.head()
_input1 = all_data.loc['train']
_input0 = all_data.loc['test']
(_input1.shape, _input0.shape)
target = _input1['SalePrice']
_input1 = _input1.drop(['SalePrice'], axis=1)
_input0 = _input0.drop(['SalePrice'], axis=1)
(X, y) = (_input1, target)
(_input1.shape, _input0.shape)
gbr_reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber')