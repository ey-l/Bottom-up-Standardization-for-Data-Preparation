import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input1.shape
_input1.info()
_input1.describe()
round(100 * _input1.isnull().sum() / len(_input1.index), 2).sort_values(ascending=False)
_input1.columns[100 * _input1.isnull().sum() / len(_input1.index) > 45]
_input1.loc[_input1['PoolQC'].isnull(), ['PoolQC']] = 'No Pool'
_input1.loc[_input1['Fence'].isnull(), ['Fence']] = 'No Fence'
_input1.loc[_input1['MiscFeature'].isnull(), ['MiscFeature']] = 'none'
_input1.loc[_input1['Alley'].isnull(), ['Alley']] = 'No alley access'
_input1.loc[_input1['FireplaceQu'].isnull(), ['FireplaceQu']] = 'No Fireplace'
_input1.loc[_input1['BsmtQual'].isnull(), ['BsmtQual']] = 'No Basement'
_input1.loc[_input1['BsmtCond'].isnull(), ['BsmtCond']] = 'No Basement'
_input1.loc[_input1['BsmtExposure'].isnull(), ['BsmtExposure']] = 'No Basement'
_input1.loc[_input1['BsmtFinType1'].isnull(), ['BsmtFinType1']] = 'No Basement'
_input1.loc[_input1['BsmtFinType2'].isnull(), ['BsmtFinType2']] = 'No Basement'
_input1.loc[_input1['MasVnrType'].isnull(), ['MasVnrType']] = 'none'
_input1.loc[_input1['MasVnrArea'].isnull(), ['MasVnrArea']] = 0
100 * _input1['LotFrontage'].isnull().sum() / len(_input1.index)
_input1['LotFrontage'] = _input1['LotFrontage'].replace(np.nan, _input1['LotFrontage'].mean(), inplace=False)
_input1.loc[_input1['GarageType'].isnull(), ['GarageType']] = 'No Garage'
_input1.loc[_input1['GarageFinish'].isnull(), ['GarageFinish']] = 'No Garage'
_input1.loc[_input1['GarageQual'].isnull(), ['GarageQual']] = 'No Garage'
_input1.loc[_input1['GarageCond'].isnull(), ['GarageCond']] = 'No Garage'
_input1.loc[_input1['Electrical'].isnull(), ['Electrical']] = 'SBrkr'
_input1.loc[_input1['GarageYrBlt'].isnull(), ['GarageYrBlt']] = 2019
_input1.columns[100 * _input1.isnull().sum() / len(_input1.index) > 0]
_input1.shape
_input1 = _input1.drop_duplicates()
_input1.shape
_input1['SalePrice'].describe()
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input1['TotalSF'] = _input1['TotalBsmtSF'] + _input1['1stFlrSF'] + _input1['2ndFlrSF']
_input1['house_age'] = 2019 - _input1['YearBuilt']
_input1['garage_age'] = 2019 - _input1['GarageYrBlt']
_input1['gap_between_build_remodel'] = _input1['YearRemodAdd'] - _input1['YearBuilt']
_input1['MSSubClass'] = _input1['MSSubClass'].astype('object')
_input1['OverallCond'] = _input1['OverallCond'].astype('object')
_input1['YrSold'] = _input1['YrSold'].astype('object')
_input1['MoSold'] = _input1['MoSold'].astype('object')
house_numeric = _input1.select_dtypes(include=['float64', 'int64'])
house_numeric.head()
house_numeric.columns
house_categorical = _input1.select_dtypes(include=['object'])
house_categorical.columns
print(len(house_categorical.columns))
print(len(house_numeric.columns))
corr = house_numeric.corr()
corr
plt.figure(figsize=(20, 15))
sns.heatmap(corr, cmap='coolwarm', annot=True)
sns.set()
plt.figure(figsize=(40, 30))
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'KitchenAbvGr', 'Fireplaces', 'WoodDeckSF', 'PoolArea', 'TotalSF']
sns.pairplot(_input1[cols], size=2.5)
sns.distplot(_input1['SalePrice'])
house_numeric = house_numeric.drop(['YearBuilt', 'YearRemodAdd', 'Fireplaces'], axis=1, inplace=False)
house_numeric.head()
house_numeric.shape
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
sns.violinplot(_input1.TotalSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 6)
sns.violinplot(_input1['1stFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 7)
sns.violinplot(_input1['2ndFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 8)
sns.violinplot(_input1.LowQualFinSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 9)
sns.violinplot(_input1.GrLivArea, fill='#A4A4A4', color='red')
plt.figure(figsize=(24, 12))
plt.subplot(3, 3, 1)
sns.violinplot(_input1.GrLivArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 2)
sns.violinplot(_input1.TotRmsAbvGrd, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 3)
sns.violinplot(_input1.house_age, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 4)
sns.violinplot(_input1.garage_age, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 5)
sns.violinplot(_input1.PoolArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 6)
sns.violinplot(_input1.MiscVal, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 7)
sns.violinplot(_input1.EnclosedPorch, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 8)
sns.violinplot(_input1.GarageArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 9)
sns.violinplot(_input1.SalePrice, fill='#A4A4A4', color='red')
Q1 = _input1.LotFrontage.quantile(0.25)
Q3 = _input1.LotFrontage.quantile(0.75)
IQR = Q3 - Q1
_input1 = _input1[(_input1.LotFrontage >= Q1 - 1.5 * IQR) & (_input1.LotFrontage <= Q3 + 1.5 * IQR)]
Q1 = _input1.LotArea.quantile(0.25)
Q3 = _input1.LotArea.quantile(0.75)
IQR = Q3 - Q1
_input1 = _input1[(_input1.LotArea >= Q1 - 1.5 * IQR) & (_input1.LotArea <= Q3 + 1.5 * IQR)]
Q1 = _input1.PoolArea.quantile(0.25)
Q3 = _input1.PoolArea.quantile(0.75)
IQR = Q3 - Q1
_input1 = _input1[(_input1.PoolArea >= Q1 - 1.5 * IQR) & (_input1.PoolArea <= Q3 + 1.5 * IQR)]
Q1 = _input1.MiscVal.quantile(0.25)
Q3 = _input1.MiscVal.quantile(0.75)
IQR = Q3 - Q1
_input1 = _input1[(_input1.MiscVal >= Q1 - 1.5 * IQR) & (_input1.MiscVal <= Q3 + 1.5 * IQR)]
_input1.shape
X = _input1.drop(['SalePrice'], axis=1)
y = _input1['SalePrice']
_input1['CentralAir'] = _input1['CentralAir'].map({'Y': 1, 'N': 0})
house_categorical_df = X.select_dtypes(include=['object'])
house_categorical_df.columns
house_df_dummies = pd.get_dummies(house_categorical_df, drop_first=True)
house_df_dummies.head()
X = X.drop(list(house_categorical_df.columns), axis=1)
X = pd.concat([X, house_df_dummies], axis=1)
X.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale_var = X.columns
X[scale_var] = scaler.fit_transform(X[scale_var])
X.describe()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn import metrics
lm = LinearRegression()