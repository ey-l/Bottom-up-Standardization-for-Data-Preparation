import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
house = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
house.head()
house.info()
temp = pd.DataFrame(house.isna().sum(), columns=['number'])
temp2 = temp[temp['number'] > 0]
temp2.sort_values(['number'], ascending=False)
house = house.drop(['PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'Alley', 'Fence', 'FireplaceQu', 'Fireplaces', 'LotFrontage', 'Id'], axis=1)
house = house.drop(['GarageYrBlt'], axis=1)
house['GarageType'] = house['GarageType'].fillna('no_g')
house['GarageFinish'] = house['GarageFinish'].fillna('no_g')
house['GarageQual'] = house['GarageQual'].fillna('no_g')
house['GarageCond'] = house['GarageCond'].fillna('no_g')
house['GarageType'] = house['GarageType'].fillna('no_g')
house['BsmtExposure'] = house['BsmtExposure'].fillna('no_b')
house['BsmtFinType1'] = house['BsmtFinType1'].fillna('no_b')
house['BsmtFinType2'] = house['BsmtFinType2'].fillna('no_b')
house['BsmtCond'] = house['BsmtCond'].fillna('no_b')
house['BsmtQual'] = house['BsmtQual'].fillna('no_b')
house['MasVnrType'] = house['MasVnrType'].fillna('no_m')
house['MasVnrArea'] = house['MasVnrArea'].fillna(0)
temp = pd.DataFrame(house.isna().sum(), columns=['number'])
temp2 = temp[temp['number'] > 0]
temp2.sort_values(['number'], ascending=False)
test = house[house['Electrical'].isna()]
test[['Electrical', 'Utilities']]
house['Electrical'].value_counts()
house['Electrical'] = house['Electrical'].fillna('SBrkr')
house['SalePrice']
house.isna()
from sklearn.model_selection import train_test_split
target = house['SalePrice']
features = house.drop('SalePrice', axis=1)
(train_x, test_x, train_y, test_y) = train_test_split(house.drop('SalePrice', axis=1), house['SalePrice'], test_size=0.2)
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