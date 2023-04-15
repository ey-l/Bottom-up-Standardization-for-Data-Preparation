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
house = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
house.head()
house.shape
test.shape
house.info()
house.describe()
house_numeric = house.select_dtypes(include=['float64', 'int64'])
house_numeric.head()
house_numeric.info()
house_numeric = house_numeric.drop(['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold'], axis=1)
house_numeric.head()
house_numeric.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
sns.violinplot(house['PoolArea'])
Q1 = house['PoolArea'].quantile(0.1)
Q3 = house['PoolArea'].quantile(0.9)
IQR = Q3 - Q1
house = house[(house['PoolArea'] >= Q1 - 1.5 * IQR) & (house['PoolArea'] <= Q3 + 1.5 * IQR)]
house.shape
sns.violinplot(house['MiscVal'])
Q1 = house['MiscVal'].quantile(0.1)
Q3 = house['MiscVal'].quantile(0.9)
IQR = Q3 - Q1
house = house[(house['MiscVal'] >= Q1 - 1.5 * IQR) & (house['MiscVal'] <= Q3 + 1.5 * IQR)]
house.shape
sns.violinplot(house['ScreenPorch'])
Q1 = house['ScreenPorch'].quantile(0.1)
Q3 = house['ScreenPorch'].quantile(0.9)
IQR = Q3 - Q1
house = house[(house['ScreenPorch'] >= Q1 - 1.5 * IQR) & (house['ScreenPorch'] <= Q3 + 1.5 * IQR)]
house.shape
sns.violinplot(house['LotArea'])
Q1 = house['LotArea'].quantile(0.1)
Q3 = house['LotArea'].quantile(0.9)
IQR = Q3 - Q1
house = house[(house['LotArea'] >= Q1 - 1.5 * IQR) & (house['LotArea'] <= Q3 + 1.5 * IQR)]
house.shape
sns.violinplot(house['MasVnrArea'])
Q1 = house['MasVnrArea'].quantile(0.1)
Q3 = house['MasVnrArea'].quantile(0.9)
IQR = Q3 - Q1
house = house[(house['MasVnrArea'] >= Q1 - 1.5 * IQR) & (house['MasVnrArea'] <= Q3 + 1.5 * IQR)]
house.shape
sns.violinplot(house['SalePrice'])
Q1 = house['SalePrice'].quantile(0.1)
Q3 = house['SalePrice'].quantile(0.9)
IQR = Q3 - Q1
house = house[(house['SalePrice'] >= Q1 - 1.5 * IQR) & (house['SalePrice'] <= Q3 + 1.5 * IQR)]
house.shape
plt.figure(figsize=(24, 12))
plt.subplot(3, 3, 1)
sns.violinplot(house.LotFrontage, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 2)
sns.violinplot(house.LotArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 3)
sns.violinplot(house.MasVnrArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 4)
sns.violinplot(house.BsmtUnfSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 5)
sns.violinplot(house.TotalBsmtSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 6)
sns.violinplot(house['1stFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 7)
sns.violinplot(house['2ndFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 8)
sns.violinplot(house.LowQualFinSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 9)
sns.violinplot(house.GrLivArea, fill='#A4A4A4', color='red')

cor = house_numeric.corr()
cor
plt.figure(figsize=(18, 10))
sns.heatmap(cor, annot=True)

house.info()
house.isnull().sum()
round(100 * (house.isnull().sum() / len(house.index)), 2)
house.shape
house = pd.concat((house, test))
house['Alley'].fillna('No Alley', inplace=True)
house['MasVnrType'].fillna('None', inplace=True)
house['FireplaceQu'].fillna('No Fireplace', inplace=True)
house['PoolQC'].fillna('No Pool', inplace=True)
house['Fence'].fillna('No Fence', inplace=True)
house['MasVnrArea'].fillna(0, inplace=True)
house['LotFrontage'].fillna(0, inplace=True)
house['GarageType'].fillna('No Garage', inplace=True)
house['GarageFinish'].fillna('No Garage', inplace=True)
house['GarageQual'].fillna('No Garage', inplace=True)
house['GarageCond'].fillna('No Garage', inplace=True)
house = house.drop('MiscFeature', axis=1)
house.isnull().sum()
house['YearBuilt'] = 2019 - house['YearBuilt']
house['YearRemodAdd'] = 2019 - house['YearRemodAdd']
house['GarageYrBlt'] = 2019 - house['GarageYrBlt']
house['YrSold'] = 2019 - house['YrSold']
house['MSSubClass'] = house['MSSubClass'].astype('object')
house['OverallQual'] = house['OverallQual'].astype('object')
house['OverallCond'] = house['OverallCond'].astype('object')
house['BsmtFullBath'] = house['BsmtFullBath'].astype('object')
house['BsmtHalfBath'] = house['BsmtHalfBath'].astype('object')
house['FullBath'] = house['FullBath'].astype('object')
house['HalfBath'] = house['HalfBath'].astype('object')
house['BedroomAbvGr'] = house['BedroomAbvGr'].astype('object')
house['KitchenAbvGr'] = house['KitchenAbvGr'].astype('object')
house['TotRmsAbvGrd'] = house['TotRmsAbvGrd'].astype('object')
house['Fireplaces'] = house['Fireplaces'].astype('object')
house['GarageCars'] = house['GarageCars'].astype('object')
house.shape
final = house
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
test = final.tail(1459)
test.shape
X = final.head(1253)
y = np.log(X.SalePrice)
X = X.drop('SalePrice', 1)
test = test.fillna(test.interpolate())
X = X.fillna(X.interpolate())
test = test.drop('SalePrice', 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()