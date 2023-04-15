import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
train_df.tail(1)
train_df.info()
train_df.columns
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
test_df.head(1)
sample = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample.head(1)
train_df.columns[train_df.isnull().sum() > 0]
import missingno as msno
msno.matrix(train_df)
print('Unique misc features => ', train_df['MiscFeature'].unique())
print('total non-0 values for misc => ', train_df[train_df.MiscVal > 0].shape[0] / train_df.shape[0])
print('Unique fence => ', train_df['Fence'].unique())
train_df.drop(['PoolQC', 'Alley', 'MiscFeature', 'MiscVal', 'Fence'], axis=1, inplace=True)
print('FireplaceQu -> ', train_df['FireplaceQu'].unique())
print('GarageType -> ', train_df['GarageType'].unique())
print('GarageFinish ->', train_df['GarageFinish'].unique())
print('GarageQual -> ', train_df['GarageQual'].unique())
print('GarageCond ->', train_df['GarageCond'].unique())
print('MasVnrType ->', train_df['MasVnrType'].unique())
print('BsmtQual ->', train_df['BsmtQual'].unique())
print('BsmtCond ->', train_df['BsmtCond'].unique())
print('BsmtExposure ->', train_df['BsmtExposure'].unique())
print('BsmtFinType1 ->', train_df['BsmtFinType1'].unique())
print('BsmtFinType2 ->', train_df['BsmtFinType2'].unique())
print('Electrical ->', train_df['Electrical'].unique())
train_df.fillna({'FireplaceQu': 'no', 'GarageType': 'no', 'GarageFinish': 'no', 'GarageQual': 'no', 'GarageCond': 'no', 'GarageYrBlt': 0, 'MasVnrType': 'None', 'BsmtQual': 'no', 'BsmtCond': 'no', 'BsmtExposure': 'no', 'BsmtFinType1': 'no', 'BsmtFinType2': 'no'}, inplace=True)
train_df['LotFrontage'].fillna(int(train_df['LotFrontage'].mean()), inplace=True)
train_df['MasVnrArea'].fillna(0, inplace=True)
train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].astype(int)
train_df.columns[train_df.isnull().sum() > 0]
train_df.describe()
import plotly.express as px
fig = px.scatter(train_df['SalePrice'])
fig.show()
columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
y_train = train_df.pop('SalePrice')
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
sns.distplot(y_train)
plt.axvline(y_train.mean(), color='r')
plt.axvline(y_train.median(), color='b')
plt.subplot(1, 2, 2)
sns.boxplot(y=y_train)

plt.figure(figsize=(16, 10))
sns.heatmap(train_df.corr(), annot=False, cmap='YlGnBu')


def scatter(x, fig):
    plt.subplot(6, 2, fig)
    plt.scatter(train_df[x], y_train)
    plt.title(x + ' vs SalePrice')
    plt.ylabel('SalePrice')
    plt.xlabel(x)
plt.figure(figsize=(10, 20))
scatter('LotArea', 1)
scatter('OverallCond', 2)
scatter('YearBuilt', 3)
scatter('GrLivArea', 4)
scatter('MasVnrArea', 5)
scatter('GarageArea', 6)
plt.tight_layout()
from sklearn.preprocessing import LabelEncoder
col = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
for i in range(len(col)):
    globals()['level_%s' % i] = LabelEncoder()
    train_df[col[i]] = globals()['level_%s' % i].fit_transform(train_df[col[i]])
train_df.head(10)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_df[train_df.columns] = scaler.fit_transform(train_df[train_df.columns])
train_df.head(5)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
X_train = train_df
lm = LinearRegression()