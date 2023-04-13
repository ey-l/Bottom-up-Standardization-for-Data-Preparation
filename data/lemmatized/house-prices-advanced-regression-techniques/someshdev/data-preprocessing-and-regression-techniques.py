import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.info()
temp = pd.DataFrame(_input1.isna().sum(), columns=['number'])
temp2 = temp[temp['number'] > 0]
temp2.sort_values(['number'], ascending=False)
_input1 = _input1.drop(['PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'Alley', 'Fence', 'FireplaceQu', 'Fireplaces', 'LotFrontage', 'Id'], axis=1)
_input1 = _input1.drop(['GarageYrBlt'], axis=1)
_input1['GarageType'] = _input1['GarageType'].fillna('no_g')
_input1['GarageFinish'] = _input1['GarageFinish'].fillna('no_g')
_input1['GarageQual'] = _input1['GarageQual'].fillna('no_g')
_input1['GarageCond'] = _input1['GarageCond'].fillna('no_g')
_input1['GarageType'] = _input1['GarageType'].fillna('no_g')
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('no_b')
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna('no_b')
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna('no_b')
_input1['BsmtCond'] = _input1['BsmtCond'].fillna('no_b')
_input1['BsmtQual'] = _input1['BsmtQual'].fillna('no_b')
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('no_m')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0)
temp = pd.DataFrame(_input1.isna().sum(), columns=['number'])
temp2 = temp[temp['number'] > 0]
temp2.sort_values(['number'], ascending=False)
test = _input1[_input1['Electrical'].isna()]
test[['Electrical', 'Utilities']]
_input1['Electrical'].value_counts()
_input1['Electrical'] = _input1['Electrical'].fillna('SBrkr')
_input1['SalePrice']
_input1.isna()
from sklearn.model_selection import train_test_split
target = _input1['SalePrice']
features = _input1.drop('SalePrice', axis=1)
(train_x, test_x, train_y, test_y) = train_test_split(_input1.drop('SalePrice', axis=1), _input1['SalePrice'], test_size=0.2)
from sklearn.preprocessing import OrdinalEncoder
numeric_data = train_x.select_dtypes(include=[np.number])
categorical_data = train_x.select_dtypes(exclude=[np.number])

def encode(dataset):
    temp = []
    frequency = []
    for x in dataset.columns:
        if x.endswith('Qual'):
            temp.append(x)
        elif x.endswith('Cond'):
            temp.append(x)
        elif x.endswith('QC'):
            temp.append(x)
        else:
            frequency.append(x)
    encoder = OrdinalEncoder()