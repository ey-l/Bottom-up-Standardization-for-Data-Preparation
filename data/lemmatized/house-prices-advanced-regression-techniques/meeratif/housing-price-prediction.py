import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.shape
_input1.dtypes
_input1.info
_input1.columns
_input1.head(10)
_input1.tail()
fig = plot.figure(figsize=(12, 10))
plot.subplot(321)
sns.scatterplot(data=_input1, x='GarageArea', y='SalePrice')
plot.subplot(322)
sns.scatterplot(data=_input1, x='YearBuilt', y='SalePrice')
plot.subplot(323)
sns.scatterplot(data=_input1, x='WoodDeckSF', y='SalePrice')
plot.subplot(324)
sns.scatterplot(data=_input1, x='OverallQual', y='SalePrice')
plot.subplot(325)
sns.scatterplot(data=_input1, x='BsmtUnfSF', y='SalePrice')
plot.subplot(326)
sns.scatterplot(data=_input1, x='TotalBsmtSF', y='SalePrice')
y = _input1.SalePrice
plot.figure(figsize=(7, 7))
sns.boxplot(data=y)
y.describe()
data = _input1
y = y.drop_duplicates()
data.dtypes.value_counts().plot.pie(autopct='%0.2f%%')
obj_col = _input1.select_dtypes(include=['object']).columns
int_col = _input1.select_dtypes(include=['int']).columns
print('Categorical Data Columns', obj_col, '\n', 'Integar Data Columns', int_col)
_input1['LotFrontage'] = int(_input1['LotFrontage'].mean())
_input1['MasVnrArea'] = int(_input1['MasVnrArea'].mean())
_input1['BsmtFinSF1'] = int(_input1['BsmtFinSF1'].mean())
_input1['BsmtFinSF2'] = int(_input1['BsmtFinSF2'].mean())
_input1['BsmtUnfSF'] = int(_input1['BsmtUnfSF'].mean())
_input1['TotalBsmtSF'] = int(_input1['TotalBsmtSF'].mean())
_input1['BsmtHalfBath'] = int(_input1['BsmtHalfBath'].mean())
_input1['GarageYrBlt'] = int(_input1['GarageYrBlt'].mean())
_input1['GarageCars'] = int(_input1['GarageCars'].mean())
_input1['GarageArea'] = int(_input1['GarageArea'].mean())
_input1 = _input1.drop(['Alley'], inplace=False, axis=1)
_input1 = _input1.drop(['MiscFeature'], inplace=False, axis=1)
_input1 = _input1.drop(['Id'], inplace=False, axis=1)
_input1 = _input1.drop(['PoolQC'], inplace=False, axis=1)
_input1 = _input1.drop(['Fence'], inplace=False, axis=1)
_input0 = _input0.drop(['Alley'], inplace=False, axis=1)
_input0 = _input0.drop(['MiscFeature'], inplace=False, axis=1)
_input0 = _input0.drop(['Id'], inplace=False, axis=1)
_input0 = _input0.drop(['PoolQC'], inplace=False, axis=1)
_input0 = _input0.drop(['Fence'], inplace=False, axis=1)
null_p = _input1.isnull().sum() / data.shape[0] * 100
null_p = _input0.isnull().sum() / data.shape[0] * 100
cread_t_col = null_p[null_p > 30].keys()
_input0 = _input0.drop(cread_t_col, 'columns')
create_d_col = null_p[null_p > 30].keys()
_input1 = _input1.drop(create_d_col, 'columns')
_input1['MSZoning'] = _input1['MSZoning'].mode()[0]
_input1['MSSubClass'] = _input1['MSSubClass'].mode()[0]
_input1['BsmtCond'] = _input1['BsmtCond'].mode()[0]
_input1['BsmtQual'] = _input1['BsmtQual'].mode()[0]
_input1['GarageType'] = _input1['GarageType'].mode()[0]
_input1['BsmtCond'] = _input1['BsmtCond'].mode()[0]
_input1['BsmtExposure'] = _input1['BsmtExposure'].mode()[0]
_input1['GarageArea'] = _input1['GarageArea'].mode()[0]
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].mode()[0]
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].mode()[0]
_input1['GarageCond'] = _input1['GarageYrBlt'].mode()[0]
_input1['GarageFinish'] = _input1['GarageYrBlt'].mode()[0]
_input1['Exterior2nd'] = _input1['GarageYrBlt'].mode()[0]
_input0.dropna(axis=0, how='any')
plot.figure(figsize=(15, 5))
_input1.dtypes.value_counts().plot.pie(autopct='%0.2f%%')
plot.figure(figsize=(25, 28))
sns.heatmap(_input1.corr(), cmap='coolwarm', annot=True, linewidth=2)
obj_c = _input1.select_dtypes('object')
obj_c
'plot.figure(figsize=(10,8))\nplot.plot(y)'
_input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
_input1 = _input1.drop('SalePrice', axis=1)
print(_input1.shape, y.shape)
_input1
_input1
d_train_data = _input1.apply(LabelEncoder().fit_transform)
d_test_data = _input0.apply(LabelEncoder().fit_transform)
d_train_data
(x_train, x_test, y_train, y_test) = train_test_split(d_train_data, y, test_size=0.2, random_state=42)
rmr = RandomForestRegressor()