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
house = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
house.head()
house.shape
house.info()
house.describe()
round(100 * house.isnull().sum() / len(house.index), 2).sort_values(ascending=False)
house.columns[100 * house.isnull().sum() / len(house.index) > 45]
house.loc[house['PoolQC'].isnull(), ['PoolQC']] = 'No Pool'
house.loc[house['Fence'].isnull(), ['Fence']] = 'No Fence'
house.loc[house['MiscFeature'].isnull(), ['MiscFeature']] = 'none'
house.loc[house['Alley'].isnull(), ['Alley']] = 'No alley access'
house.loc[house['FireplaceQu'].isnull(), ['FireplaceQu']] = 'No Fireplace'
house.loc[house['BsmtQual'].isnull(), ['BsmtQual']] = 'No Basement'
house.loc[house['BsmtCond'].isnull(), ['BsmtCond']] = 'No Basement'
house.loc[house['BsmtExposure'].isnull(), ['BsmtExposure']] = 'No Basement'
house.loc[house['BsmtFinType1'].isnull(), ['BsmtFinType1']] = 'No Basement'
house.loc[house['BsmtFinType2'].isnull(), ['BsmtFinType2']] = 'No Basement'
house.loc[house['MasVnrType'].isnull(), ['MasVnrType']] = 'none'
house.loc[house['MasVnrArea'].isnull(), ['MasVnrArea']] = 0
100 * house['LotFrontage'].isnull().sum() / len(house.index)
house['LotFrontage'].replace(np.nan, house['LotFrontage'].mean(), inplace=True)
house.loc[house['GarageType'].isnull(), ['GarageType']] = 'No Garage'
house.loc[house['GarageFinish'].isnull(), ['GarageFinish']] = 'No Garage'
house.loc[house['GarageQual'].isnull(), ['GarageQual']] = 'No Garage'
house.loc[house['GarageCond'].isnull(), ['GarageCond']] = 'No Garage'
house.loc[house['Electrical'].isnull(), ['Electrical']] = 'SBrkr'
house.loc[house['GarageYrBlt'].isnull(), ['GarageYrBlt']] = 2019
house.columns[100 * house.isnull().sum() / len(house.index) > 0]
house.shape
house = house.drop_duplicates()
house.shape
house['SalePrice'].describe()
house.drop(['Id'], axis=1, inplace=True)
house['TotalSF'] = house['TotalBsmtSF'] + house['1stFlrSF'] + house['2ndFlrSF']
house['house_age'] = 2019 - house['YearBuilt']
house['garage_age'] = 2019 - house['GarageYrBlt']
house['gap_between_build_remodel'] = house['YearRemodAdd'] - house['YearBuilt']
house['MSSubClass'] = house['MSSubClass'].astype('object')
house['OverallCond'] = house['OverallCond'].astype('object')
house['YrSold'] = house['YrSold'].astype('object')
house['MoSold'] = house['MoSold'].astype('object')
house_numeric = house.select_dtypes(include=['float64', 'int64'])
house_numeric.head()
house_numeric.columns
house_categorical = house.select_dtypes(include=['object'])
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
sns.pairplot(house[cols], size=2.5)

sns.distplot(house['SalePrice'])
house_numeric.drop(['YearBuilt', 'YearRemodAdd', 'Fireplaces'], axis=1, inplace=True)
house_numeric.head()
house_numeric.shape
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
sns.violinplot(house.TotalSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 6)
sns.violinplot(house['1stFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 7)
sns.violinplot(house['2ndFlrSF'], fill='#A4A4A4', color='red')
plt.subplot(3, 3, 8)
sns.violinplot(house.LowQualFinSF, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 9)
sns.violinplot(house.GrLivArea, fill='#A4A4A4', color='red')

plt.figure(figsize=(24, 12))
plt.subplot(3, 3, 1)
sns.violinplot(house.GrLivArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 2)
sns.violinplot(house.TotRmsAbvGrd, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 3)
sns.violinplot(house.house_age, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 4)
sns.violinplot(house.garage_age, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 5)
sns.violinplot(house.PoolArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 6)
sns.violinplot(house.MiscVal, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 7)
sns.violinplot(house.EnclosedPorch, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 8)
sns.violinplot(house.GarageArea, fill='#A4A4A4', color='red')
plt.subplot(3, 3, 9)
sns.violinplot(house.SalePrice, fill='#A4A4A4', color='red')

Q1 = house.LotFrontage.quantile(0.25)
Q3 = house.LotFrontage.quantile(0.75)
IQR = Q3 - Q1
house = house[(house.LotFrontage >= Q1 - 1.5 * IQR) & (house.LotFrontage <= Q3 + 1.5 * IQR)]
Q1 = house.LotArea.quantile(0.25)
Q3 = house.LotArea.quantile(0.75)
IQR = Q3 - Q1
house = house[(house.LotArea >= Q1 - 1.5 * IQR) & (house.LotArea <= Q3 + 1.5 * IQR)]
Q1 = house.PoolArea.quantile(0.25)
Q3 = house.PoolArea.quantile(0.75)
IQR = Q3 - Q1
house = house[(house.PoolArea >= Q1 - 1.5 * IQR) & (house.PoolArea <= Q3 + 1.5 * IQR)]
Q1 = house.MiscVal.quantile(0.25)
Q3 = house.MiscVal.quantile(0.75)
IQR = Q3 - Q1
house = house[(house.MiscVal >= Q1 - 1.5 * IQR) & (house.MiscVal <= Q3 + 1.5 * IQR)]
house.shape
X = house.drop(['SalePrice'], axis=1)
y = house['SalePrice']
house['CentralAir'] = house['CentralAir'].map({'Y': 1, 'N': 0})
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