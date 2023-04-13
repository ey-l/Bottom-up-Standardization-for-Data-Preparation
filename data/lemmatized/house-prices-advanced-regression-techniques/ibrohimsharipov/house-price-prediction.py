import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1
print(_input1.shape, _input0.shape)
_input1.info()
_input1.isna().sum().sort_values(ascending=False).head(20)
df = _input1.isna().sum().sum()
df
len(_input1) / df * 100
plt.figure(figsize=(15, 8))
sns.heatmap(_input1.isna(), cmap='Paired')
_input1['PoolQC'] = _input1['PoolQC'].fillna('None')
_input1['MiscFeature'] = _input1['MiscFeature'].fillna('None')
_input1['Alley'] = _input1['Alley'].fillna('None')
_input1['Fence'] = _input1['Fence'].fillna('None')
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('None')
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(0)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].value_counts().idxmax())
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].value_counts().idxmax())
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].value_counts().idxmax())
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].value_counts().idxmax())
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].value_counts().idxmax())
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].value_counts().idxmax())
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].value_counts().idxmax())
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].value_counts().idxmax())
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].value_counts().idxmax())
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].value_counts().idxmax())
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].value_counts().idxmax())
_input1.isna().sum().sort_values(ascending=False).head(20)
_input1.columns
plt.figure(figsize=(15, 8))
sns.heatmap(_input1.isna(), cmap='Paired')
_input1.shape
corr = _input1.corr()
corr = corr.sort_values(['SalePrice'], ascending=False, inplace=False)
corr.SalePrice.head(10)
plt.figure(figsize=(25, 20))
plt.subplot(5, 5, 1)
plt.title('OverallQual')
sns.barplot(x='OverallQual', y='SalePrice', data=_input1)
plt.subplot(5, 5, 2)
plt.title('GrLivArea')
plt.scatter(x='GrLivArea', y='SalePrice', data=_input1)
plt.subplot(5, 5, 3)
plt.title('GarageCars')
sns.barplot(x='GarageCars', y='SalePrice', data=_input1)
plt.subplot(5, 5, 4)
plt.title('GarageArea')
sns.scatterplot(x='GarageArea', y='SalePrice', data=_input1)
plt.subplot(5, 5, 5)
plt.title('TotalBsmtSF')
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=_input1)
plt.subplot(5, 5, 6)
plt.title('1stFlrSF')
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=_input1)
label = LabelEncoder()
for i in _input1.columns:
    if _input1[i].dtypes == object:
        _input1[i] = label.fit_transform(_input1[i])
X = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)

def decision_tree_model(X_train, y_train):
    tree = DecisionTreeRegressor(random_state=1)