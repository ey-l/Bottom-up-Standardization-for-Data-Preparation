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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.shape
_input0.shape
_input1.info()
_input1.describe([0.25, 0.5, 0.75, 0.99])
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input1.isnull().sum().sort_values(ascending=False).head(20)
_input1[_input1.isnull().sum(axis=1) > 1]
round(_input1.isnull().sum() * 100 / _input1.shape[0], 2).sort_values(ascending=False).head(20)
threshold = 10
drop_cols = round(_input1.isnull().sum() * 100 / _input1.shape[0], 2)[round(_input1.isnull().sum() * 100 / _input1.shape[0], 2) > threshold].index.tolist()
drop_cols
_input1 = _input1.drop(columns=drop_cols, inplace=False)
_input0 = _input0.drop(columns=drop_cols, inplace=False)
_input1.shape
_input1.head()
round(_input1.isnull().sum() * 100 / _input1.shape[0], 2)[round(_input1.isnull().sum() * 100 / _input1.shape[0], 2) > 0].sort_values(ascending=False)
round(_input0.isnull().sum() * 100 / _input0.shape[0], 2)[round(_input0.isnull().sum() * 100 / _input0.shape[0], 2) > 0].sort_values(ascending=False)
_input1['GarageYrBlt'] = 2021 - _input1['GarageYrBlt']
_input1['YearBuilt'] = 2021 - _input1['YearBuilt']
_input1['YearRemodAdd'] = 2021 - _input1['YearRemodAdd']
_input1['YrSold'] = 2021 - _input1['YrSold']
_input1[['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']].head()
_input0['GarageYrBlt'] = 2021 - _input0['GarageYrBlt']
_input0['YearBuilt'] = 2021 - _input0['YearBuilt']
_input0['YearRemodAdd'] = 2021 - _input0['YearRemodAdd']
_input0['YrSold'] = 2021 - _input0['YrSold']
_input0[['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']].head()
_input1['GarageFinish'] = _input1['GarageFinish'].fillna('No Garage', inplace=False)
_input1['GarageType'] = _input1['GarageType'].fillna('No Garage', inplace=False)
_input1['GarageQual'] = _input1['GarageQual'].fillna('No Garage', inplace=False)
_input1['GarageCond'] = _input1['GarageCond'].fillna('No Garage', inplace=False)
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(-1, inplace=False)
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('No Basement', inplace=False)
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna('No Basement', inplace=False)
_input1['BsmtQual'] = _input1['BsmtQual'].fillna('No Basement', inplace=False)
_input1['BsmtCond'] = _input1['BsmtCond'].fillna('No Basement', inplace=False)
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna('No Basement', inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None', inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(0, inplace=False)
_input1 = _input1.dropna(axis=0, inplace=False)
_input1.shape
_input0['GarageFinish'] = _input0['GarageFinish'].fillna('No Garage', inplace=False)
_input0['GarageType'] = _input0['GarageType'].fillna('No Garage', inplace=False)
_input0['GarageQual'] = _input0['GarageQual'].fillna('No Garage', inplace=False)
_input0['GarageCond'] = _input0['GarageCond'].fillna('No Garage', inplace=False)
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(-1, inplace=False)
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna('No Basement', inplace=False)
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna('No Basement', inplace=False)
_input0['BsmtQual'] = _input0['BsmtQual'].fillna('No Basement', inplace=False)
_input0['BsmtCond'] = _input0['BsmtCond'].fillna('No Basement', inplace=False)
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna('No Basement', inplace=False)
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('None', inplace=False)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(0, inplace=False)
_input0.shape
_input1.describe([0.25, 0.5, 0.75, 0.99])
num_col = list(_input1.dtypes[_input1.dtypes != 'object'].index)

def drop_outliers(x):
    list = []
    for col in num_col:
        Q1 = x[col].quantile(0.25)
        Q3 = x[col].quantile(0.99)
        IQR = Q3 - Q1
        x = x[(x[col] >= Q1 - 1.5 * IQR) & (x[col] <= Q3 + 1.5 * IQR)]
    return x
_input1 = drop_outliers(_input1)
_input1.shape
_input1.describe([0.25, 0.5, 0.75, 0.99])
_input1 = _input1.drop(columns=['PoolArea'], inplace=False)
_input0 = _input0.drop(columns=['PoolArea'], inplace=False)
plt.title('SalePrice')
sns.distplot(_input1['SalePrice'], bins=10)
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
plt.title('SalePrice')
sns.distplot(_input1['SalePrice'], bins=10)
num_vars = list(_input1.dtypes[_input1.dtypes != 'object'].index)
_input1[num_vars].head()
plt.figure(figsize=(10, 5))
sns.pairplot(_input1, x_vars=['MSSubClass', 'LotArea', 'MasVnrArea'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(_input1, x_vars=['OverallQual', 'OverallCond', 'OpenPorchSF'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(_input1, x_vars=['BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(_input1, x_vars=['1stFlrSF', '2ndFlrSF', 'GrLivArea'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(_input1, x_vars=['BsmtFullBath', 'FullBath', 'HalfBath'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(_input1, x_vars=['BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(_input1, x_vars=['GarageCars', 'GarageArea', 'WoodDeckSF'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
sns.pairplot(_input1, x_vars=['3SsnPorch', 'MiscVal', 'KitchenAbvGr'], y_vars='SalePrice', height=4, aspect=1, kind='scatter')
_input1 = _input1.drop(columns=['MSSubClass', '3SsnPorch', 'MiscVal'], inplace=False)
_input1 = _input1.drop(columns=['KitchenAbvGr', 'BsmtFullBath', 'BsmtHalfBath'], inplace=False)
_input1['GrLivArea'] = _input1['GrLivArea'].clip(0, 3000)
_input1['TotalBsmtSF'] = _input1['TotalBsmtSF'].clip(0, 3000)
_input1['1stFlrSF'] = _input1['1stFlrSF'].clip(0, 3000)
_input1['GarageArea'] = _input1['GarageArea'].clip(0, 1200)
_input1['BsmtFinSF1'] = _input1['BsmtFinSF1'].clip(0, 2500)
_input1['OpenPorchSF'] = _input1['OpenPorchSF'].clip(0, 400)
_input1['LotArea'] = _input1['LotArea'].clip(0, 60000)
_input1.shape
_input0 = _input0.drop(columns=['MSSubClass', '3SsnPorch', 'MiscVal'], inplace=False)
_input0 = _input0.drop(columns=['KitchenAbvGr', 'BsmtFullBath', 'BsmtHalfBath'], inplace=False)
_input0['GrLivArea'] = _input0['GrLivArea'].clip(0, 3000)
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].clip(0, 3000)
_input0['1stFlrSF'] = _input0['1stFlrSF'].clip(0, 3000)
_input0['GarageArea'] = _input0['GarageArea'].clip(0, 1200)
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].clip(0, 2500)
_input0['OpenPorchSF'] = _input0['OpenPorchSF'].clip(0, 400)
_input0['LotArea'] = _input0['LotArea'].clip(0, 60000)
_input0.shape
plt.figure(figsize=(30, 20))
sns.heatmap(_input1.corr(), annot=True, cmap='YlGnBu')
_input1 = _input1.drop(columns=['GarageArea', 'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'GarageYrBlt', 'YearRemodAdd'], inplace=False)
_input1.shape
_input0 = _input0.drop(columns=['GarageArea', 'TotRmsAbvGrd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'GarageYrBlt', 'YearRemodAdd'], inplace=False)
_input0.shape
numerical_columns = _input1.select_dtypes(include=['int64', 'float64'])
skewness_of_feats = numerical_columns.apply(lambda x: skew(x)).sort_values(ascending=False)
print(skewness_of_feats)
_input1['LowQualFinSF'] = np.log1p(_input1['LowQualFinSF'])
_input1['LotArea'] = np.log1p(_input1['LotArea'])
_input1['BsmtFinSF2'] = np.log1p(_input1['BsmtFinSF2'])
_input1['ScreenPorch'] = np.log1p(_input1['ScreenPorch'])
_input1['EnclosedPorch'] = np.log1p(_input1['EnclosedPorch'])
_input1['MasVnrArea'] = np.log1p(_input1['MasVnrArea'])
_input1['OpenPorchSF'] = np.log1p(_input1['OpenPorchSF'])
_input1['WoodDeckSF'] = np.log1p(_input1['WoodDeckSF'])
_input1['BsmtUnfSF'] = np.log1p(_input1['BsmtUnfSF'])
_input0['LowQualFinSF'] = np.log1p(_input0['LowQualFinSF'])
_input0['LotArea'] = np.log1p(_input0['LotArea'])
_input0['BsmtFinSF2'] = np.log1p(_input0['BsmtFinSF2'])
_input0['ScreenPorch'] = np.log1p(_input0['ScreenPorch'])
_input0['EnclosedPorch'] = np.log1p(_input0['EnclosedPorch'])
_input0['MasVnrArea'] = np.log1p(_input0['MasVnrArea'])
_input0['OpenPorchSF'] = np.log1p(_input0['OpenPorchSF'])
_input0['WoodDeckSF'] = np.log1p(_input0['WoodDeckSF'])
_input0['BsmtUnfSF'] = np.log1p(_input0['BsmtUnfSF'])
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
sns.boxplot(x='MSZoning', y='SalePrice', data=_input1)
plt.subplot(3, 3, 2)
sns.boxplot(x='BldgType', y='SalePrice', data=_input1)
plt.subplot(3, 3, 3)
sns.boxplot(x='Street', y='SalePrice', data=_input1)
plt.subplot(3, 3, 4)
sns.boxplot(x='LotShape', y='SalePrice', data=_input1)
plt.subplot(3, 3, 5)
sns.boxplot(x='HouseStyle', y='SalePrice', data=_input1)
plt.subplot(3, 3, 6)
sns.boxplot(x='Utilities', y='SalePrice', data=_input1)
plt.subplot(3, 3, 7)
sns.boxplot(x='RoofStyle', y='SalePrice', data=_input1)
plt.subplot(3, 3, 8)
sns.boxplot(x='LandSlope', y='SalePrice', data=_input1)
plt.subplot(3, 3, 9)
sns.boxplot(x='Neighborhood', y='SalePrice', data=_input1)
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
sns.boxplot(x='ExterQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 2)
sns.boxplot(x='Foundation', y='SalePrice', data=_input1)
plt.subplot(3, 3, 3)
sns.boxplot(x='BsmtQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 4)
sns.boxplot(x='Heating', y='SalePrice', data=_input1)
plt.subplot(3, 3, 5)
sns.boxplot(x='CentralAir', y='SalePrice', data=_input1)
plt.subplot(3, 3, 6)
sns.boxplot(x='Electrical', y='SalePrice', data=_input1)
plt.subplot(3, 3, 7)
sns.boxplot(x='KitchenQual', y='SalePrice', data=_input1)
plt.subplot(3, 3, 8)
sns.boxplot(x='GarageType', y='SalePrice', data=_input1)
plt.subplot(3, 3, 9)
sns.boxplot(x='GarageQual', y='SalePrice', data=_input1)
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='SaleType', y='SalePrice', data=_input1)
_input1 = _input1.drop(columns=['Utilities'], inplace=False)
_input0 = _input0.drop(columns=['Utilities'], inplace=False)
cat_vars = list(_input1.dtypes[_input1.dtypes == 'object'].index)
_input1[cat_vars].head(10)
_input1[['LandSlope', 'ExterQual', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'GarageFinish', 'GarageQual', 'GarageCond', 'ExterCond', 'LotShape']].head()
_input1['LandSlope'] = _input1.LandSlope.map({'Sev': 0, 'Mod': 1, 'Gtl': 2})
_input1['ExterQual'] = _input1.ExterQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['BsmtQual'] = _input1.BsmtQual.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['BsmtCond'] = _input1.BsmtCond.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['BsmtExposure'] = _input1.BsmtExposure.map({'No Basement': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
_input1['BsmtFinType1'] = _input1.BsmtFinType1.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
_input1['BsmtFinType2'] = _input1.BsmtFinType2.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
_input1['HeatingQC'] = _input1.HeatingQC.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['CentralAir'] = _input1.CentralAir.map({'N': 0, 'Y': 1})
_input1['KitchenQual'] = _input1.KitchenQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['GarageFinish'] = _input1.GarageFinish.map({'No Garage': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
_input1['GarageQual'] = _input1.GarageQual.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['GarageCond'] = _input1.GarageCond.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input1['ExterCond'] = _input1.ExterCond.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input1['LotShape'] = _input1.LotShape.map({'IR1': 0, 'IR2': 1, 'IR3': 2, 'Reg': 3})
_input1.head()
_input0['LandSlope'] = _input0.LandSlope.map({'Sev': 0, 'Mod': 1, 'Gtl': 2})
_input0['ExterQual'] = _input0.ExterQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input0['BsmtQual'] = _input0.BsmtQual.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input0['BsmtCond'] = _input0.BsmtCond.map({'No Basement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input0['BsmtExposure'] = _input0.BsmtExposure.map({'No Basement': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
_input0['BsmtFinType1'] = _input0.BsmtFinType1.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
_input0['BsmtFinType2'] = _input0.BsmtFinType2.map({'No Basement': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
_input0['HeatingQC'] = _input0.HeatingQC.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input0['CentralAir'] = _input0.CentralAir.map({'N': 0, 'Y': 1})
_input0['KitchenQual'] = _input0.KitchenQual.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input0['GarageFinish'] = _input0.GarageFinish.map({'No Garage': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
_input0['GarageQual'] = _input0.GarageQual.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input0['GarageCond'] = _input0.GarageCond.map({'No Garage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
_input0['ExterCond'] = _input0.ExterCond.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
_input0['LotShape'] = _input0.LotShape.map({'IR1': 0, 'IR2': 1, 'IR3': 2, 'Reg': 3})
cat_vars = list(_input1.dtypes[_input1.dtypes == 'object'].index)
cat_vars
_input1 = pd.get_dummies(data=_input1, columns=cat_vars, drop_first=True)
_input1.info()
_input0 = pd.get_dummies(data=_input0, columns=cat_vars, drop_first=True)
_input0.info()
missing_cols = set(_input1.columns) - set(_input0.columns)
for c in missing_cols:
    _input0[c] = 0
(_input1, _input0) = _input1.align(_input0, axis=1)
_input0.info()
_input1.describe()
X = _input1.drop(['SalePrice'], axis=1)
X.head()
y = _input1['SalePrice']
y.head()
from sklearn.preprocessing import scale
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns
cols = _input0.columns
_input0 = pd.DataFrame(scale(_input0))
_input0.columns = cols
_input0.columns
np.random.seed(0)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
len(X_train.index)
len(X_test.index)
lm = LinearRegression()