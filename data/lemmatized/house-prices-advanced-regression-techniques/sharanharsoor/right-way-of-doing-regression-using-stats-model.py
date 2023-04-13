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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col=0)
_input1.tail(1)
_input1.info()
_input1.columns
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col=0)
_input0.head(1)
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input2.head(1)
_input1.columns[_input1.isnull().sum() > 0]
import missingno as msno
msno.matrix(_input1)
print('Unique misc features => ', _input1['MiscFeature'].unique())
print('total non-0 values for misc => ', _input1[_input1.MiscVal > 0].shape[0] / _input1.shape[0])
print('Unique fence => ', _input1['Fence'].unique())
_input1 = _input1.drop(['PoolQC', 'Alley', 'MiscFeature', 'MiscVal', 'Fence'], axis=1, inplace=False)
print('FireplaceQu -> ', _input1['FireplaceQu'].unique())
print('GarageType -> ', _input1['GarageType'].unique())
print('GarageFinish ->', _input1['GarageFinish'].unique())
print('GarageQual -> ', _input1['GarageQual'].unique())
print('GarageCond ->', _input1['GarageCond'].unique())
print('MasVnrType ->', _input1['MasVnrType'].unique())
print('BsmtQual ->', _input1['BsmtQual'].unique())
print('BsmtCond ->', _input1['BsmtCond'].unique())
print('BsmtExposure ->', _input1['BsmtExposure'].unique())
print('BsmtFinType1 ->', _input1['BsmtFinType1'].unique())
print('BsmtFinType2 ->', _input1['BsmtFinType2'].unique())
print('Electrical ->', _input1['Electrical'].unique())
_input1 = _input1.fillna({'FireplaceQu': 'no', 'GarageType': 'no', 'GarageFinish': 'no', 'GarageQual': 'no', 'GarageCond': 'no', 'GarageYrBlt': 0, 'MasVnrType': 'None', 'BsmtQual': 'no', 'BsmtCond': 'no', 'BsmtExposure': 'no', 'BsmtFinType1': 'no', 'BsmtFinType2': 'no'}, inplace=False)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(int(_input1['LotFrontage'].mean()), inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0, inplace=False)
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0])
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].astype(int)
_input1.columns[_input1.isnull().sum() > 0]
_input1.describe()
import plotly.express as px
fig = px.scatter(_input1['SalePrice'])
fig.show()
columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
y_train = _input1.pop('SalePrice')
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
sns.distplot(y_train)
plt.axvline(y_train.mean(), color='r')
plt.axvline(y_train.median(), color='b')
plt.subplot(1, 2, 2)
sns.boxplot(y=y_train)
plt.figure(figsize=(16, 10))
sns.heatmap(_input1.corr(), annot=False, cmap='YlGnBu')

def scatter(x, fig):
    plt.subplot(6, 2, fig)
    plt.scatter(_input1[x], y_train)
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
    _input1[col[i]] = globals()['level_%s' % i].fit_transform(_input1[col[i]])
_input1.head(10)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
_input1[_input1.columns] = scaler.fit_transform(_input1[_input1.columns])
_input1.head(5)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
X_train = _input1
lm = LinearRegression()