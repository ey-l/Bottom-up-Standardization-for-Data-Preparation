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
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
sub_file = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_data.shape
train_data.dtypes
train_data.info
train_data.columns
train_data.head(10)
train_data.tail()
fig = plot.figure(figsize=(12, 10))
plot.subplot(321)
sns.scatterplot(data=train_data, x='GarageArea', y='SalePrice')
plot.subplot(322)
sns.scatterplot(data=train_data, x='YearBuilt', y='SalePrice')
plot.subplot(323)
sns.scatterplot(data=train_data, x='WoodDeckSF', y='SalePrice')
plot.subplot(324)
sns.scatterplot(data=train_data, x='OverallQual', y='SalePrice')
plot.subplot(325)
sns.scatterplot(data=train_data, x='BsmtUnfSF', y='SalePrice')
plot.subplot(326)
sns.scatterplot(data=train_data, x='TotalBsmtSF', y='SalePrice')
y = train_data.SalePrice
plot.figure(figsize=(7, 7))
sns.boxplot(data=y)
y.describe()
data = train_data
y = y.drop_duplicates()
data.dtypes.value_counts().plot.pie(autopct='%0.2f%%')





obj_col = train_data.select_dtypes(include=['object']).columns
int_col = train_data.select_dtypes(include=['int']).columns
print('Categorical Data Columns', obj_col, '\n', 'Integar Data Columns', int_col)
train_data['LotFrontage'] = int(train_data['LotFrontage'].mean())
train_data['MasVnrArea'] = int(train_data['MasVnrArea'].mean())
train_data['BsmtFinSF1'] = int(train_data['BsmtFinSF1'].mean())
train_data['BsmtFinSF2'] = int(train_data['BsmtFinSF2'].mean())
train_data['BsmtUnfSF'] = int(train_data['BsmtUnfSF'].mean())
train_data['TotalBsmtSF'] = int(train_data['TotalBsmtSF'].mean())
train_data['BsmtHalfBath'] = int(train_data['BsmtHalfBath'].mean())
train_data['GarageYrBlt'] = int(train_data['GarageYrBlt'].mean())
train_data['GarageCars'] = int(train_data['GarageCars'].mean())
train_data['GarageArea'] = int(train_data['GarageArea'].mean())
train_data.drop(['Alley'], inplace=True, axis=1)
train_data.drop(['MiscFeature'], inplace=True, axis=1)
train_data.drop(['Id'], inplace=True, axis=1)
train_data.drop(['PoolQC'], inplace=True, axis=1)
train_data.drop(['Fence'], inplace=True, axis=1)
test_data.drop(['Alley'], inplace=True, axis=1)
test_data.drop(['MiscFeature'], inplace=True, axis=1)
test_data.drop(['Id'], inplace=True, axis=1)
test_data.drop(['PoolQC'], inplace=True, axis=1)
test_data.drop(['Fence'], inplace=True, axis=1)
null_p = train_data.isnull().sum() / data.shape[0] * 100
null_p = test_data.isnull().sum() / data.shape[0] * 100
cread_t_col = null_p[null_p > 30].keys()
test_data = test_data.drop(cread_t_col, 'columns')
create_d_col = null_p[null_p > 30].keys()
train_data = train_data.drop(create_d_col, 'columns')
train_data['MSZoning'] = train_data['MSZoning'].mode()[0]
train_data['MSSubClass'] = train_data['MSSubClass'].mode()[0]
train_data['BsmtCond'] = train_data['BsmtCond'].mode()[0]
train_data['BsmtQual'] = train_data['BsmtQual'].mode()[0]
train_data['GarageType'] = train_data['GarageType'].mode()[0]
train_data['BsmtCond'] = train_data['BsmtCond'].mode()[0]
train_data['BsmtExposure'] = train_data['BsmtExposure'].mode()[0]
train_data['GarageArea'] = train_data['GarageArea'].mode()[0]
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].mode()[0]
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].mode()[0]
train_data['GarageCond'] = train_data['GarageYrBlt'].mode()[0]
train_data['GarageFinish'] = train_data['GarageYrBlt'].mode()[0]
train_data['Exterior2nd'] = train_data['GarageYrBlt'].mode()[0]
test_data.dropna(axis=0, how='any')
plot.figure(figsize=(15, 5))

train_data.dtypes.value_counts().plot.pie(autopct='%0.2f%%')
plot.figure(figsize=(25, 28))
sns.heatmap(train_data.corr(), cmap='coolwarm', annot=True, linewidth=2)
obj_c = train_data.select_dtypes('object')
obj_c
'plot.figure(figsize=(10,8))\nplot.plot(y)'
train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']
train_data = train_data.drop('SalePrice', axis=1)
print(train_data.shape, y.shape)
train_data
train_data
d_train_data = train_data.apply(LabelEncoder().fit_transform)
d_test_data = test_data.apply(LabelEncoder().fit_transform)
d_train_data
(x_train, x_test, y_train, y_test) = train_test_split(d_train_data, y, test_size=0.2, random_state=42)
rmr = RandomForestRegressor()