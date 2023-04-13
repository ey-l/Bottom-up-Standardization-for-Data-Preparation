import numpy as np
import pandas as pd
import scipy as sp
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import category_encoders as ce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', encoding='latin-1')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', encoding='latin-1')
_input1.head()
_input0.head()
_input1.info()
_input1.shape
train2 = _input1.copy()
test2 = _input0.copy()
missing_values = pd.concat([_input1.isnull().sum(), _input0.isnull().sum()], axis=1, keys=['TRAIN', 'TEST'])
missing_values[missing_values.sum(axis=1) > 0]
merged_data = train2.append(test2)
merged_data = merged_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Id'], axis=1)
for col in ['LotFrontage', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'FireplaceQu', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'Fence']:
    merged_data[col] = merged_data[col].fillna(0)
else:
    merged_data[col] = merged_data[col].fillna('None')
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'Fence']
encoder = ce.OrdinalEncoder(cols=ordinal_cols)
merged_data = encoder.fit_transform(merged_data)
merged_data.head()
train2 = merged_data.iloc[:len(train2)]
train2_quant = train2.select_dtypes(exclude=['object'])
plt.subplots(figsize=(55, 40))
sns.heatmap(train2_quant.corr('pearson').abs(), annot=True, square=True, linewidths=0.5)
cor_pairs = train2_quant.corr().unstack()
strong_pairs = cor_pairs[(cor_pairs > 0.8) & (cor_pairs < 1.0)]
strong_pairs
merged_data = merged_data.drop(['GarageYrBlt', '1stFlrSF', 'GarageArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageCond'], axis=1)
sales_cor = train2.corr('pearson').abs()['SalePrice']
sorted_cor_target = sales_cor.sort_values(kind='Quicksort', ascending=False)
sorted_cor_target
train2_quant = train2.select_dtypes(exclude=['object'])
quant_eda = train2.hist(column=train2_quant.columns, figsize=(30, 40))
train_quant_log = train2_quant.copy()
for col in train_quant_log.columns:
    train_quant_log[col] = np.log1p(train_quant_log[col])
train_quant_log.hist(figsize=(30, 40), bins=50, xlabelsize=8, ylabelsize=8)
merged_data['SalePrice'] = np.log1p(merged_data['SalePrice'])
merged_data['LotArea'] = np.log1p(merged_data['LotArea'])
merged_data['BsmtUnfSF'] = np.log1p(merged_data['BsmtUnfSF'])
merged_data['GrLivArea'] = np.log1p(merged_data['GrLivArea'])
object_cols = merged_data.select_dtypes(include=['object']).columns
dummy_cols = []
for col in object_cols:
    if col not in ordinal_cols:
        dummy_cols.append(col)
merged_data = pd.get_dummies(merged_data, columns=dummy_cols)
Train = merged_data.iloc[:len(train2)]
Test = merged_data.iloc[len(train2):].drop('SalePrice', axis=1)
Train.head()
Test.head()
quant_features = list((Train.dtypes != 'object')[Train.dtypes != 'object'].index)
low_cor_features = set()
for i in quant_features:
    if abs(Train[i].corr(Train['SalePrice'])) < 0.02:
        low_cor_features.add(i)
low_cor_features
train1 = Train.copy()
test1 = Test.copy()
train1 = train1.drop(columns=low_cor_features)
test1 = test1.drop(columns=low_cor_features)
X_train = train1.drop('SalePrice', axis=1)
Y_train = train1['SalePrice']
(x_train, x_valid, y_train, y_valid) = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
len(x_train.columns)
linreg = LinearRegression()