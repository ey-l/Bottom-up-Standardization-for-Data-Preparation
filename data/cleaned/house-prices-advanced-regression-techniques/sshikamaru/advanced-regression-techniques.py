import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
print(train.shape)
print(test.shape)
train.isnull().sum().sort_values(ascending=False)[0:20]
sns.heatmap(train.isnull(), yticklabels=False, cbar='BuPu')
train.info()
"train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean)\ntrain.drop(['Alley'], axis = 1, inplace=True)\n\ntrain['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])\ntrain['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])\ntrain.drop(['PoolQC'], axis = 1, inplace=True)\ntrain.drop(['Fence'], axis = 1, inplace=True)    \ntrain.drop(['MiscFeature'], axis = 1, inplace=True)"
for i in list(train.columns):
    dtype = train[i].dtype
    values = 0
    if dtype == float or dtype == int:
        method = 'mean'
    else:
        method = 'mode'
    if train[i].notnull().sum() / 1460 <= 0.5:
        train.drop(i, axis=1, inplace=True)
    elif method == 'mean':
        train[i] = train[i].fillna(train[i].mean())
    else:
        train[i] = train[i].fillna(train[i].mode()[0])
        print(train[i])
for i in list(test.columns):
    dtype = test[i].dtype
    values = 0
    if dtype == float or dtype == int:
        method = 'mean'
    else:
        method = 'mode'
    if test[i].notnull().sum() / 1460 <= 0.5:
        test.drop(i, axis=1, inplace=True)
    elif method == 'mean':
        test[i] = test[i].fillna(test[i].mean())
    else:
        test[i] = test[i].fillna(test[i].mode()[0])
train.head()
test.shape
test.drop(columns=['Id'], inplace=True)
train.dropna(inplace=True)
train.drop(columns=['Id'], inplace=True)
print(train.shape)
print(test.shape)
train.isnull().any().any()
train.head()
plt.figure(figsize=(15, 5))
plt.plot(train.SalePrice, linewidth=2, color=next(color_cycle))
plt.title('Distribution Plot for Sales Prices')
plt.ylabel('Sales Price')
plt.figure(figsize=(15, 5))
plt.plot(train.SalePrice.sort_values().reset_index(drop=True), color=next(color_cycle))
plt.title('Distribution Plot for Sales Prices')
plt.ylabel('Sales Price')
fig = px.scatter(train, x=train.index, y='SalePrice', labels={'x': 'Index'}, color=train.MSZoning, template='seaborn', title='Sale Price distriution of MSZoning')
fig.show()
fig = px.scatter(train, x=train.index, y='SalePrice', labels={'x': 'Index'}, color=train.Street, template='seaborn', title='Sale Price distriution ---> Street')
fig.show()
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.scatter(x=train[train.LotConfig == 'FR3'].index, y=train[train.LotConfig == 'FR3'].SalePrice, color=next(color_cycle))
plt.title('SalePrice distribution of FR3 value of LotConfig')
plt.subplot(2, 2, 2)
plt.scatter(x=train[train.LotConfig == 'CulDSac'].index, y=train[train.LotConfig == 'CulDSac'].SalePrice, color=next(color_cycle))
plt.title('SalePrice distribution of CulDSac value of LotConfig')
plt.subplot(2, 2, 3)
plt.scatter(x=train[train.LotConfig == 'Corner'].index, y=train[train.LotConfig == 'Corner'].SalePrice, color=next(color_cycle))
plt.title('SalePrice distribution of Corner value of LotConfig')
plt.subplot(2, 2, 4)
plt.scatter(x=train[train.LotConfig == 'FR2'].index, y=train[train.LotConfig == 'FR2'].SalePrice, color=next(color_cycle))
plt.title('SalePrice distribution of FR2 value of  LotConfig')
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
train_test_data = pd.concat([train, test], axis=0)
print(test.shape)
print(train.shape)
train_test_data.head()
train_test_data.shape

def One_hot_encoding(columns):
    df_final = train_test_data
    i = 0
    for fields in columns:
        df1 = pd.get_dummies(train_test_data[fields], drop_first=True)
        train_test_data.drop([fields], axis=1, inplace=True)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([train_test_data, df_final], axis=1)
    return df_final
train_test_data = One_hot_encoding(columns)
print(train_test_data.shape)
train_test_data.head()
train_test_data.columns.duplicated()
train_test_data = train_test_data.loc[:, ~train_test_data.columns.duplicated()]
train_test_data.shape
from scipy.stats import norm, skew
from scipy import stats
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])
res = stats.probplot(train['SalePrice'], plot=plt)
df_Train = train_test_data.iloc[:1460, :]
df_Test = train_test_data.iloc[1460:, :]
print(df_Test.shape)
df_Test.head()
df_Test.drop(['SalePrice'], axis=1, inplace=True)
X_train_final = df_Train.drop(['SalePrice'], axis=1)
y_train_final = df_Train['SalePrice']
X_train_final.shape
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X_train_final)
my_columns = X_train_final.columns
new_df = pd.DataFrame(X_std, columns=my_columns)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(new_df)
print(df_pca)
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=y_train_final, cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X_train_final, y_train_final)
linreg = LinearRegression()