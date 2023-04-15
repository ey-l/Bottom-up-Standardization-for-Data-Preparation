import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from scipy.stats import skew

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
housing = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
housing.head()
housing.shape
test.shape
housing.info()
housing.describe([0.25, 0.5, 0.75, 0.99])
housing.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
housing.isnull().sum().sort_values(ascending=False).head(20)
housing[housing.isnull().sum(axis=1) > 1]
round(housing.isnull().sum() * 100 / housing.shape[0], 2).sort_values(ascending=False).head(20)
threshold = 10
drop_cols = round(housing.isnull().sum() * 100 / housing.shape[0], 2)[round(housing.isnull().sum() * 100 / housing.shape[0], 2) > threshold].index.tolist()
drop_cols
housing.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)
housing.shape
housing.head()
round(housing.isnull().sum() * 100 / housing.shape[0], 2)[round(housing.isnull().sum() * 100 / housing.shape[0], 2) > 0].sort_values(ascending=False)
round(test.isnull().sum() * 100 / test.shape[0], 2)[round(test.isnull().sum() * 100 / test.shape[0], 2) > 0].sort_values(ascending=False)
housing['GarageYrBlt'] = 2021 - housing['GarageYrBlt']
housing['YearBuilt'] = 2021 - housing['YearBuilt']
housing['YearRemodAdd'] = 2021 - housing['YearRemodAdd']
housing['YrSold'] = 2021 - housing['YrSold']
housing[['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']].head()
test['GarageYrBlt'] = 2021 - test['GarageYrBlt']
test['YearBuilt'] = 2021 - test['YearBuilt']
test['YearRemodAdd'] = 2021 - test['YearRemodAdd']
test['YrSold'] = 2021 - test['YrSold']
test[['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']].head()
housing['GarageFinish'].fillna('No Garage', inplace=True)
housing['GarageType'].fillna('No Garage', inplace=True)
housing['GarageQual'].fillna('No Garage', inplace=True)
housing['GarageCond'].fillna('No Garage', inplace=True)
housing['GarageYrBlt'].fillna(-1, inplace=True)
housing['BsmtExposure'].fillna('No Basement', inplace=True)
housing['BsmtFinType1'].fillna('No Basement', inplace=True)
housing['BsmtQual'].fillna('No Basement', inplace=True)
housing['BsmtCond'].fillna('No Basement', inplace=True)
housing['BsmtFinType2'].fillna('No Basement', inplace=True)
housing['MasVnrType'].fillna('None', inplace=True)
housing['MasVnrArea'].fillna(0, inplace=True)
housing.dropna(axis=0, inplace=True)
housing.shape
test['GarageFinish'].fillna('No Garage', inplace=True)
test['GarageType'].fillna('No Garage', inplace=True)
test['GarageQual'].fillna('No Garage', inplace=True)
test['GarageCond'].fillna('No Garage', inplace=True)
test['GarageYrBlt'].fillna(-1, inplace=True)
test['BsmtExposure'].fillna('No Basement', inplace=True)
test['BsmtFinType1'].fillna('No Basement', inplace=True)
test['BsmtQual'].fillna('No Basement', inplace=True)
test['BsmtCond'].fillna('No Basement', inplace=True)
test['BsmtFinType2'].fillna('No Basement', inplace=True)
test['MasVnrType'].fillna('None', inplace=True)
test['MasVnrArea'].fillna(0, inplace=True)
test.shape
housing.describe([0.25, 0.5, 0.75, 0.99])
num_col = list(housing.dtypes[housing.dtypes != 'object'].index)

def drop_outliers(x):
    list = []
    for col in num_col:
        Q1 = x[col].quantile(0.25)
        Q3 = x[col].quantile(0.99)
        IQR = Q3 - Q1
        x = x[(x[col] >= Q1 - 1.5 * IQR) & (x[col] <= Q3 + 1.5 * IQR)]
    return x
housing = drop_outliers(housing)
housing.shape
housing.describe([0.25, 0.5, 0.75, 0.99])
housing.drop(columns=['PoolArea'], inplace=True)
test.drop(columns=['PoolArea'], inplace=True)
plt.title('SalePrice')
sns.distplot(housing['SalePrice'], bins=10)

housing['SalePrice'] = np.log1p(housing['SalePrice'])
plt.title('SalePrice')
sns.distplot(housing['SalePrice'], bins=10)

num_vars = list(housing.dtypes[housing.dtypes != 'object'].index)
housing[num_vars].head()
plt.figure(figsize=(10, 5))
sns.pairplot(housing, x_vars=['MSSubClass', 'LotArea', 'MasVnrArea'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(housing, x_vars=['OverallQual', 'OverallCond', 'OpenPorchSF'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(housing, x_vars=['BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(housing, x_vars=['1stFlrSF', '2ndFlrSF', 'GrLivArea'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(housing, x_vars=['BsmtFullBath', 'FullBath', 'HalfBath'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(housing, x_vars=['BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(housing, x_vars=['GarageCars', 'GarageArea', 'WoodDeckSF'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(housing, x_vars=['3SsnPorch', 'MiscVal', 'KitchenAbvGr'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')

housing.drop(columns=['MSSubClass', '3SsnPorch', 'MiscVal'], inplace=True)
housing.drop(columns=['KitchenAbvGr', 'BsmtFullBath', 'BsmtHalfBath'], inplace=True)
housing['GrLivArea'] = housing['GrLivArea'].clip(0, 3000)
housing['TotalBsmtSF'] = housing['TotalBsmtSF'].clip(0, 3000)
housing['1stFlrSF'] = housing['1stFlrSF'].clip(0, 3000)
housing['GarageArea'] = housing['GarageArea'].clip(0, 1200)
housing['BsmtFinSF1'] = housing['BsmtFinSF1'].clip(0, 2500)
housing['OpenPorchSF'] = housing['OpenPorchSF'].clip(0, 400)
housing['LotArea'] = housing['LotArea'].clip(0, 60000)
housing.shape
test.drop(columns=['MSSubClass', '3SsnPorch', 'MiscVal'], inplace=True)
test.drop(columns=['KitchenAbvGr', 'BsmtFullBath', 'BsmtHalfBath'], inplace=True)
test['GrLivArea'] = test['GrLivArea'].clip(0, 3000)
test['TotalBsmtSF'] = test['TotalBsmtSF'].clip(0, 3000)
test['1stFlrSF'] = test['1stFlrSF'].clip(0, 3000)
test['GarageArea'] = test['GarageArea'].clip(0, 1200)
test['BsmtFinSF1'] = test['BsmtFinSF1'].clip(0, 2500)
test['OpenPorchSF'] = test['OpenPorchSF'].clip(0, 400)
test['LotArea'] = test['LotArea'].clip(0, 60000)
test.shape
plt.figure(figsize=(30, 20))
sns.heatmap(housing.corr(), annot=True, cmap='YlGnBu')

housing.drop(columns=['GarageArea', 'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'GarageYrBlt', 'YearRemodAdd'], inplace=True)
housing.shape
test.drop(columns=['GarageArea', 'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'GarageYrBlt', 'YearRemodAdd'], inplace=True)
test.shape
numerical_columns = housing.select_dtypes(include=['int64', 'float64'])
skewness_of_feats = numerical_columns.apply(lambda x: skew(x)).sort_values(ascending=False)
print(skewness_of_feats)
housing['LowQualFinSF'] = np.log1p(housing['LowQualFinSF'])
housing['LotArea'] = np.log1p(housing['LotArea'])
housing['BsmtFinSF2'] = np.log1p(housing['BsmtFinSF2'])
housing['ScreenPorch'] = np.log1p(housing['ScreenPorch'])
housing['EnclosedPorch'] = np.log1p(housing['EnclosedPorch'])
housing['MasVnrArea'] = np.log1p(housing['MasVnrArea'])
housing['OpenPorchSF'] = np.log1p(housing['OpenPorchSF'])
housing['WoodDeckSF'] = np.log1p(housing['WoodDeckSF'])
housing['BsmtUnfSF'] = np.log1p(housing['BsmtUnfSF'])
test['LowQualFinSF'] = np.log1p(test['LowQualFinSF'])
test['LotArea'] = np.log1p(test['LotArea'])
test['BsmtFinSF2'] = np.log1p(test['BsmtFinSF2'])
test['ScreenPorch'] = np.log1p(test['ScreenPorch'])
test['EnclosedPorch'] = np.log1p(test['EnclosedPorch'])
test['MasVnrArea'] = np.log1p(test['MasVnrArea'])
test['OpenPorchSF'] = np.log1p(test['OpenPorchSF'])
test['WoodDeckSF'] = np.log1p(test['WoodDeckSF'])
test['BsmtUnfSF'] = np.log1p(test['BsmtUnfSF'])
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
sns.boxplot(x='MSZoning', y='SalePrice', data=housing)
plt.subplot(3, 3, 2)
sns.boxplot(x='BldgType', y='SalePrice', data=housing)
plt.subplot(3, 3, 3)
sns.boxplot(x='Street', y='SalePrice', data=housing)
plt.subplot(3, 3, 4)
sns.boxplot(x='LotShape', y='SalePrice', data=housing)
plt.subplot(3, 3, 5)
sns.boxplot(x='HouseStyle', y='SalePrice', data=housing)
plt.subplot(3, 3, 6)
sns.boxplot(x='Utilities', y='SalePrice', data=housing)
plt.subplot(3, 3, 7)
sns.boxplot(x='RoofStyle', y='SalePrice', data=housing)
plt.subplot(3, 3, 8)
sns.boxplot(x='LandSlope', y='SalePrice', data=housing)
plt.subplot(3, 3, 9)
sns.boxplot(x='Neighborhood', y='SalePrice', data=housing)

plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
sns.boxplot(x='ExterQual', y='SalePrice', data=housing)
plt.subplot(3, 3, 2)
sns.boxplot(x='Foundation', y='SalePrice', data=housing)
plt.subplot(3, 3, 3)
sns.boxplot(x='BsmtQual', y='SalePrice', data=housing)
plt.subplot(3, 3, 4)
sns.boxplot(x='Heating', y='SalePrice', data=housing)
plt.subplot(3, 3, 5)
sns.boxplot(x='CentralAir', y='SalePrice', data=housing)
plt.subplot(3, 3, 6)
sns.boxplot(x='Electrical', y='SalePrice', data=housing)
plt.subplot(3, 3, 7)
sns.boxplot(x='KitchenQual', y='SalePrice', data=housing)
plt.subplot(3, 3, 8)
sns.boxplot(x='GarageType', y='SalePrice', data=housing)
plt.subplot(3, 3, 9)
sns.boxplot(x='GarageQual', y='SalePrice', data=housing)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='SaleType', y='SalePrice', data=housing)
housing.drop(columns=['Utilities'], inplace=True)
test.drop(columns=['Utilities'], inplace=True)
cat_vars = list(housing.dtypes[housing.dtypes == 'object'].index)
housing[cat_vars].head(10)
housing[['LandSlope', 'ExterQual', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'GarageFinish', 'GarageQual', 'GarageCond', 'ExterCond', 'LotShape']].head()
housing['LandSlope'] = housing.LandSlope.map({'Sev': 0, 'Mod': 1, 'Gtl': 2})
housing['ExterQual'] = housing.ExterQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
housing['BsmtQual'] = housing.BsmtQual.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
housing['BsmtCond'] = housing.BsmtCond.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
housing['BsmtExposure'] = housing.BsmtExposure.map({'No Basement': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
housing['BsmtFinType1'] = housing.BsmtFinType1.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
housing['BsmtFinType2'] = housing.BsmtFinType2.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
housing['HeatingQC'] = housing.HeatingQC.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
housing['CentralAir'] = housing.CentralAir.map({'N': 0, 'Y': 1})
housing['KitchenQual'] = housing.KitchenQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
housing['GarageFinish'] = housing.GarageFinish.map({'No Garage': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
housing['GarageQual'] = housing.GarageQual.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
housing['GarageCond'] = housing.GarageCond.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
housing['ExterCond'] = housing.ExterCond.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
housing['LotShape'] = housing.LotShape.map({'IR1': 0, 'IR2': 1, 'IR3': 2, 'Reg': 3})
housing.head()
test['LandSlope'] = test.LandSlope.map({'Sev': 0, 'Mod': 1, 'Gtl': 2})
test['ExterQual'] = test.ExterQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
test['BsmtQual'] = test.BsmtQual.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
test['BsmtCond'] = test.BsmtCond.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
test['BsmtExposure'] = test.BsmtExposure.map({'No Basement': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
test['BsmtFinType1'] = test.BsmtFinType1.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
test['BsmtFinType2'] = test.BsmtFinType2.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
test['HeatingQC'] = test.HeatingQC.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
test['CentralAir'] = test.CentralAir.map({'N': 0, 'Y': 1})
test['KitchenQual'] = test.KitchenQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
test['GarageFinish'] = test.GarageFinish.map({'No Garage': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
test['GarageQual'] = test.GarageQual.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
test['GarageCond'] = test.GarageCond.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
test['ExterCond'] = test.ExterCond.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
test['LotShape'] = test.LotShape.map({'IR1': 0, 'IR2': 1, 'IR3': 2, 'Reg': 3})
cat_vars = list(housing.dtypes[housing.dtypes == 'object'].index)
cat_vars
housing = pd.get_dummies(data=housing, columns=cat_vars, drop_first=True)
housing.info()
test = pd.get_dummies(data=test, columns=cat_vars, drop_first=True)
test.info()
missing_cols = set(housing.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
(housing, test) = housing.align(test, axis=1)
test.info()
housing.describe()
X = housing.drop(['SalePrice'], axis=1)
X.head()
y = housing['SalePrice']
y.head()
from sklearn.preprocessing import scale
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns
cols = test.columns
test = pd.DataFrame(scale(test))
test.columns = cols
test.columns
np.random.seed(0)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
len(X_train.index)
len(X_test.index)
lm = LinearRegression()