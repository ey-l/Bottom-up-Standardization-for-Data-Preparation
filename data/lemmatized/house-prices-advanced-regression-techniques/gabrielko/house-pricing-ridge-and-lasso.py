import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import os
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.shape
_input0.shape
_input1.info()
_input1.describe()
house_numeric = _input1.select_dtypes(include=['float64', 'int64'])
house_numeric.head()
house_numeric.info()
house_numeric = house_numeric.drop(['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold'], axis=1)
house_numeric.head()
house_numeric.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
sns.violinplot(_input1['PoolArea'])
Q1 = _input1['PoolArea'].quantile(0.1)
Q3 = _input1['PoolArea'].quantile(0.9)
IQR = Q3 - Q1
_input1 = _input1[(_input1['PoolArea'] >= Q1 - 1.5 * IQR) & (_input1['PoolArea'] <= Q3 + 1.5 * IQR)]
_input1.shape
sns.violinplot(_input1['MiscVal'])
Q1 = _input1['MiscVal'].quantile(0.1)
Q3 = _input1['MiscVal'].quantile(0.9)
IQR = Q3 - Q1
_input1 = _input1[(_input1['MiscVal'] >= Q1 - 1.5 * IQR) & (_input1['MiscVal'] <= Q3 + 1.5 * IQR)]
_input1.shape
sns.violinplot(_input1['ScreenPorch'])
Q1 = _input1['ScreenPorch'].quantile(0.1)
Q3 = _input1['ScreenPorch'].quantile(0.9)
IQR = Q3 - Q1
_input1 = _input1[(_input1['ScreenPorch'] >= Q1 - 1.5 * IQR) & (_input1['ScreenPorch'] <= Q3 + 1.5 * IQR)]
_input1.shape
sns.violinplot(_input1['LotArea'])
Q1 = _input1['LotArea'].quantile(0.1)
Q3 = _input1['LotArea'].quantile(0.9)
IQR = Q3 - Q1
_input1 = _input1[(_input1['LotArea'] >= Q1 - 1.5 * IQR) & (_input1['LotArea'] <= Q3 + 1.5 * IQR)]
_input1.shape
sns.violinplot(_input1['MasVnrArea'])
Q1 = _input1['MasVnrArea'].quantile(0.1)
Q3 = _input1['MasVnrArea'].quantile(0.9)
IQR = Q3 - Q1
_input1 = _input1[(_input1['MasVnrArea'] >= Q1 - 1.5 * IQR) & (_input1['MasVnrArea'] <= Q3 + 1.5 * IQR)]
_input1.shape
sns.violinplot(_input1['SalePrice'])
Q1 = _input1['SalePrice'].quantile(0.1)
Q3 = _input1['SalePrice'].quantile(0.9)
IQR = Q3 - Q1
_input1 = _input1[(_input1['SalePrice'] >= Q1 - 1.5 * IQR) & (_input1['SalePrice'] <= Q3 + 1.5 * IQR)]
_input1.shape
plt.figure(figsize=(24, 12))
plt.subplot(3, 3, 1)
sns.violinplot(_input1.LotFrontage, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 2)
sns.violinplot(_input1.LotArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 3)
sns.violinplot(_input1.MasVnrArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 4)
sns.violinplot(_input1.BsmtUnfSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 5)
sns.violinplot(_input1.TotalBsmtSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 6)
sns.violinplot(_input1['1stFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 7)
sns.violinplot(_input1['2ndFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 8)
sns.violinplot(_input1.LowQualFinSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 9)
sns.violinplot(_input1.GrLivArea, fill='#A4A4A4', color='red')
cor = house_numeric.corr()
cor
plt.figure(figsize=(18, 10))
sns.heatmap(cor, annot=True)
_input1.info()
_input1.isnull().sum()
round(100 * (_input1.isnull().sum() / len(_input1.index)), 2)
_input1.shape
_input1 = pd.concat((_input1, _input0))
_input1['Alley'] = _input1['Alley'].fillna('No Alley', inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None', inplace=False)
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna('No Fireplace', inplace=False)
_input1['PoolQC'] = _input1['PoolQC'].fillna('No Pool', inplace=False)
_input1['Fence'] = _input1['Fence'].fillna('No Fence', inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0, inplace=False)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(0, inplace=False)
_input1['GarageType'] = _input1['GarageType'].fillna('No Garage', inplace=False)
_input1['GarageFinish'] = _input1['GarageFinish'].fillna('No Garage', inplace=False)
_input1['GarageQual'] = _input1['GarageQual'].fillna('No Garage', inplace=False)
_input1['GarageCond'] = _input1['GarageCond'].fillna('No Garage', inplace=False)
_input1 = _input1.drop('MiscFeature', axis=1)
_input1.isnull().sum()
_input1['YearBuilt'] = 2019 - _input1['YearBuilt']
_input1['YearRemodAdd'] = 2019 - _input1['YearRemodAdd']
_input1['GarageYrBlt'] = 2019 - _input1['GarageYrBlt']
_input1['YrSold'] = 2019 - _input1['YrSold']
_input1['MSSubClass'] = _input1['MSSubClass'].astype('object')
_input1['OverallQual'] = _input1['OverallQual'].astype('object')
_input1['OverallCond'] = _input1['OverallCond'].astype('object')
_input1['BsmtFullBath'] = _input1['BsmtFullBath'].astype('object')
_input1['BsmtHalfBath'] = _input1['BsmtHalfBath'].astype('object')
_input1['FullBath'] = _input1['FullBath'].astype('object')
_input1['HalfBath'] = _input1['HalfBath'].astype('object')
_input1['BedroomAbvGr'] = _input1['BedroomAbvGr'].astype('object')
_input1['KitchenAbvGr'] = _input1['KitchenAbvGr'].astype('object')
_input1['TotRmsAbvGrd'] = _input1['TotRmsAbvGrd'].astype('object')
_input1['Fireplaces'] = _input1['Fireplaces'].astype('object')
_input1['GarageCars'] = _input1['GarageCars'].astype('object')
_input1.shape
final = _input1
varlist1 = ['Street']

def binary_map(x):
    return x.map({'Pave': 1, 'Grvl': 0})
final[varlist1] = final[varlist1].apply(binary_map)
varlist2 = ['Utilities']

def binary_map(x):
    return x.map({'AllPub': 1, 'NoSeWa': 0})
final[varlist2] = final[varlist2].apply(binary_map)
varlist3 = ['CentralAir']

def binary_map(x):
    return x.map({'Y': 1, 'N': 0})
final[varlist3] = final[varlist3].apply(binary_map)
X = final.drop(['Id'], axis=1)
house_categorical = X.select_dtypes(include=['object'])
house_categorical.head()
house_dummies = pd.get_dummies(house_categorical, drop_first=True)
house_dummies.head()
final = final.drop(list(house_categorical.columns), axis=1)
final = pd.concat([final, house_dummies], axis=1)
final.shape
_input0 = final.tail(1459)
_input0.shape
X = final.head(1253)
y = np.log(X.SalePrice)
X = X.drop('SalePrice', 1)
_input0 = _input0.fillna(_input0.interpolate())
X = X.fillna(X.interpolate())
_input0 = _input0.drop('SalePrice', 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()