import numpy as np
import pandas as pd
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
import seaborn as sns
sns.heatmap(_input1.isnull(), cbar=False, cmap='PuBu')
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head(20)
_input1 = _input1.drop(['Alley'], axis=1, inplace=False)
_input1 = _input1.drop(['PoolQC'], axis=1, inplace=False)
_input1 = _input1.drop(['MiscFeature'], axis=1, inplace=False)
_input1 = _input1.drop(['Fence'], axis=1, inplace=False)
_input1 = _input1.drop(['FireplaceQu'], axis=1, inplace=False)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mean())
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].mode()[0])
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mean())
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0])
from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in _input1.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    _input1[col] = lencoders[col].fit_transform(_input1[col])
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
x_t = _input1.drop('SalePrice', axis=1)
y_t = _input1['SalePrice']
clf_1 = SelectFromModel(RandomForestClassifier(n_estimators=100, max_features='log2', max_depth=4))
clf_2 = SelectFromModel(RandomForestClassifier(n_estimators=100, max_features='auto', max_depth=4))