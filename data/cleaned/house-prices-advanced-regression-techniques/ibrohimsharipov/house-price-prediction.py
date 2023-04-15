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

d_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
d_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
d_train
print(d_train.shape, d_test.shape)
d_train.info()
d_train.isna().sum().sort_values(ascending=False).head(20)
df = d_train.isna().sum().sum()
df
len(d_train) / df * 100
plt.figure(figsize=(15, 8))
sns.heatmap(d_train.isna(), cmap='Paired')
d_train['PoolQC'] = d_train['PoolQC'].fillna('None')
d_train['MiscFeature'] = d_train['MiscFeature'].fillna('None')
d_train['Alley'] = d_train['Alley'].fillna('None')
d_train['Fence'] = d_train['Fence'].fillna('None')
d_train['FireplaceQu'] = d_train['FireplaceQu'].fillna('None')
d_train['LotFrontage'] = d_train['LotFrontage'].fillna(d_train['LotFrontage'].mean())
d_train['GarageYrBlt'] = d_train['GarageYrBlt'].fillna(0)
d_train['MasVnrArea'] = d_train['MasVnrArea'].fillna(0)
d_train['GarageCond'] = d_train['GarageCond'].fillna(d_train['GarageCond'].value_counts().idxmax())
d_train['GarageType'] = d_train['GarageType'].fillna(d_train['GarageType'].value_counts().idxmax())
d_train['GarageFinish'] = d_train['GarageFinish'].fillna(d_train['GarageFinish'].value_counts().idxmax())
d_train['GarageQual'] = d_train['GarageQual'].fillna(d_train['GarageQual'].value_counts().idxmax())
d_train['BsmtFinType2'] = d_train['BsmtFinType2'].fillna(d_train['BsmtFinType2'].value_counts().idxmax())
d_train['BsmtExposure'] = d_train['BsmtExposure'].fillna(d_train['BsmtExposure'].value_counts().idxmax())
d_train['BsmtQual'] = d_train['BsmtQual'].fillna(d_train['BsmtQual'].value_counts().idxmax())
d_train['BsmtCond'] = d_train['BsmtCond'].fillna(d_train['BsmtCond'].value_counts().idxmax())
d_train['BsmtFinType1'] = d_train['BsmtFinType1'].fillna(d_train['BsmtFinType1'].value_counts().idxmax())
d_train['MasVnrType'] = d_train['MasVnrType'].fillna(d_train['MasVnrType'].value_counts().idxmax())
d_train['Electrical'] = d_train['Electrical'].fillna(d_train['Electrical'].value_counts().idxmax())
d_train.isna().sum().sort_values(ascending=False).head(20)
d_train.columns
plt.figure(figsize=(15, 8))
sns.heatmap(d_train.isna(), cmap='Paired')
d_train.shape
corr = d_train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice.head(10)
plt.figure(figsize=(25, 20))
plt.subplot(5, 5, 1)
plt.title('OverallQual')
sns.barplot(x='OverallQual', y='SalePrice', data=d_train)
plt.subplot(5, 5, 2)
plt.title('GrLivArea')
plt.scatter(x='GrLivArea', y='SalePrice', data=d_train)
plt.subplot(5, 5, 3)
plt.title('GarageCars')
sns.barplot(x='GarageCars', y='SalePrice', data=d_train)
plt.subplot(5, 5, 4)
plt.title('GarageArea')
sns.scatterplot(x='GarageArea', y='SalePrice', data=d_train)
plt.subplot(5, 5, 5)
plt.title('TotalBsmtSF')
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=d_train)
plt.subplot(5, 5, 6)
plt.title('1stFlrSF')
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=d_train)
label = LabelEncoder()
for i in d_train.columns:
    if d_train[i].dtypes == object:
        d_train[i] = label.fit_transform(d_train[i])
X = d_train.drop('SalePrice', axis=1)
y = d_train['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)

def decision_tree_model(X_train, y_train):
    tree = DecisionTreeRegressor(random_state=1)