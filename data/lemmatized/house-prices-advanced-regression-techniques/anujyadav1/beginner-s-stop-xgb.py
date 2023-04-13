import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
c_test = _input0.copy()
c_train = _input1.copy()
_input1.head(10)
_input0.head(10)
_input1.info()
_input0.info()
c_train.describe()
c_train.describe(exclude='number')
c_train['train'] = 1
c_test['train'] = 0
df = pd.concat([c_train, c_test], axis=0, sort=False)
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
SalePrice = pd.DataFrame(corrmat['SalePrice'].sort_values(ascending=False))
SalePrice
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
df = df.drop(['GarageArea', 'TotRmsAbvGrd', '1stFlrSF'], axis=1)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols], height=2.5)
df = df.drop(['Street', 'LandContour', 'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 'ExterCond', 'BsmtCond', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageQual', 'GarageCond', 'PavedDrive', 'Condition1', 'SaleType'], axis=1)
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
df = df.drop(missing_data[missing_data['Total'] > 5].index, axis=1)
print(df.isnull().sum().max())
df.isnull().sum()
numeric_missed = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars']
for feature in numeric_missed:
    df[feature] = df[feature].fillna(0, inplace=False)
categorical_missed = ['Exterior1st', 'Exterior2nd', 'MSZoning', 'KitchenQual']
for feature in categorical_missed:
    df[feature] = df[feature].fillna(df[feature].mode()[0], inplace=False)
df.isnull().sum().max()
plt.figure(figsize=(12, 7))
sns.displot(_input1['SalePrice']).set(ylabel=None, xlabel=None)
plt.title('House price distribution histogram', fontsize=18)
from scipy.stats import skew
df = pd.get_dummies(df)
df
df = df.drop(['Id'], axis=1)
df_train = pd.DataFrame(df[:1460])
df_test = pd.DataFrame(df[1460:2920])
target = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(df_train, target, test_size=0.33, random_state=0)
xgb = XGBRegressor(learning_rate=0.1, n_estimators=100, reg_alpha=0.001, reg_lambda=1e-06, n_jobs=-1, min_child_weight=3, subsample=0.9, max_depth=5, colsample_bytree=0.2)
rf = RandomForestRegressor()