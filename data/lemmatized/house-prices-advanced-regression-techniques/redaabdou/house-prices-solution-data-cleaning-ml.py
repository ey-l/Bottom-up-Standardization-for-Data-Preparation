import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('_data/input/house-prices-advanced-regression-techniques'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1 = _input1.drop(index=[523, 1298], axis=0)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('th train data has {} rows and {} features'.format(_input1.shape[0], _input1.shape[1]))
print('the test data has {} rows and {} features'.format(_input0.shape[0], _input0.shape[1]))
data = pd.concat([_input1.iloc[:, :-1], _input0], axis=0)
print('tha data has {} rows and {} features'.format(data.shape[0], data.shape[1]))
data.columns
data.info()
num_features = data.select_dtypes(include=['int64', 'float64'])
categorical_features = data.select_dtypes(include='object')
num_features.describe()
categorical_features.describe()
data.isnull().sum().sort_values(ascending=False)[:34]
f = open('_data/input/house-prices-advanced-regression-techniques/data_description.txt', 'r')
data = data.drop(columns=['Id', 'Street', 'PoolQC', 'Utilities'], axis=1)
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
data['LotFrontage'].isnull().sum()
features = ['Electrical', 'KitchenQual', 'SaleType', 'Exterior2nd', 'Exterior1st', 'Alley', 'Fence', 'MiscFeature', 'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
for name in features:
    data[name] = data[name].fillna('Other', inplace=False)
data[features].isnull().sum()
data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
data['Functional'] = data['Functional'].fillna('typ')
"mode=['Electrical','KitchenQual','SaleType','Exterior2nd','Exterior1st']\nfor name in mode:\n    data[name].fillna(data[name].mode()[0],inplace=True)"
zero = ['GarageArea', 'GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars']
for name in zero:
    data[name] = data[name].fillna(0, inplace=False)
data.isnull().sum().sum()
data.loc[data['MSSubClass'] == 60, 'MSSubClass'] = 0
data.loc[(data['MSSubClass'] == 20) | (data['MSSubClass'] == 120), 'MSSubClass'] = 1
data.loc[data['MSSubClass'] == 75, 'MSSubClass'] = 2
data.loc[(data['MSSubClass'] == 40) | (data['MSSubClass'] == 70) | (data['MSSubClass'] == 80), 'MSSubClass'] = 3
data.loc[(data['MSSubClass'] == 50) | (data['MSSubClass'] == 85) | (data['MSSubClass'] == 90) | (data['MSSubClass'] == 160) | (data['MSSubClass'] == 190), 'MSSubClass'] = 4
data.loc[(data['MSSubClass'] == 30) | (data['MSSubClass'] == 45) | (data['MSSubClass'] == 180), 'MSSubClass'] = 5
data.loc[data['MSSubClass'] == 150, 'MSSubClass'] = 6
object_features = data.select_dtypes(include='object').columns
object_features

def dummies(d):
    dummies_df = pd.DataFrame()
    object_features = d.select_dtypes(include='object').columns
    for name in object_features:
        dummies = pd.get_dummies(d[name], drop_first=False)
        dummies = dummies.add_prefix('{}_'.format(name))
        dummies_df = pd.concat([dummies_df, dummies], axis=1)
    return dummies_df
dummies_data = dummies(data)
dummies_data.shape
data = data.drop(columns=object_features, axis=1)
data.columns
final_data = pd.concat([data, dummies_data], axis=1)
final_data.shape
train_data = final_data.iloc[:1458, :]
test_data = final_data.iloc[1458:, :]
print(train_data.shape)
test_data.shape
X = train_data
y = _input1.loc[:, 'SalePrice']
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNet
model_las_cv = LassoCV(alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))