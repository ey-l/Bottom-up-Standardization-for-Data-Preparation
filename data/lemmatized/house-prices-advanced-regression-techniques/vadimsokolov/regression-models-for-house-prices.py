import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.columns
plt.figure(figsize=(12, 8))
sns.kdeplot(_input1['SalePrice'], color='darkturquoise', shade=True)
plt.grid()
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.figure(figsize=(16, 12))
sns.heatmap(_input1.corr(), annot=True)
plt.title('Correlation matrix')
corr_for_sale_price = _input1.corr()
corr_for_sale_price['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.boxplot(x=_input1['OverallQual'], y=_input1['SalePrice'])
plt.title('The boxplot for OverallQual')
plt.figure(figsize=(12, 8))
sns.boxplot(x=_input1['YearBuilt'], y=_input1['SalePrice'])
plt.title('The boxplot for YearBuilt')
plt.xticks(rotation=60)
plt.figure(figsize=(12, 8))
sns.scatterplot(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.grid()
plt.title('Scatter plot for GrLivArea')
plt.figure(figsize=(12, 8))
sns.boxplot(x=_input1['GarageCars'], y=_input1['SalePrice'])

def prepare_data(df):
    if df.dtype == 'object':
        df = df.fillna('N')
    if df.dtype == 'float32' or df.dtype == 'float64':
        df = df.fillna(0)
    return df
_input1 = _input1.apply(lambda x: prepare_data(x))
_input0 = _input0.apply(lambda x: prepare_data(x))

def garage_area_category(area):
    if area <= 250:
        return 1
    elif 250 < area <= 500:
        return 2
    elif 500 < area <= 1000:
        return 3
    return 4
_input1['GarageArea_category'] = _input1['GarageArea'].apply(garage_area_category)
_input0['GarageArea_category'] = _input0['GarageArea'].apply(garage_area_category)

def grlivarea_category(area):
    if area <= 1000:
        return 1
    elif 1000 < area <= 2000:
        return 2
    elif 2000 < area <= 3000:
        return 3
    return 4
_input1['GrLivArea_category'] = _input1['GrLivArea'].apply(grlivarea_category)
_input0['GrLivArea_category'] = _input0['GrLivArea'].apply(grlivarea_category)

def flrSF_and_bsmt_category(square):
    if square <= 500:
        return 1
    elif 500 < square <= 1000:
        return 2
    elif 1000 < square <= 1500:
        return 3
    elif 1500 < square <= 2000:
        return 4
    return 5
_input1['1stFlrSF_category'] = _input1['1stFlrSF'].apply(flrSF_and_bsmt_category)
_input1['2ndFlrSF_category'] = _input1['2ndFlrSF'].apply(flrSF_and_bsmt_category)
_input0['1stFlrSF_category'] = _input0['1stFlrSF'].apply(flrSF_and_bsmt_category)
_input0['2ndFlrSF_category'] = _input0['2ndFlrSF'].apply(flrSF_and_bsmt_category)
_input1['BsmtUnfSF_category'] = _input1['BsmtUnfSF'].apply(flrSF_and_bsmt_category)
_input0['BsmtUnfSF_category'] = _input0['BsmtUnfSF'].apply(flrSF_and_bsmt_category)

def lot_frontage_category(frontage):
    if frontage <= 50:
        return 1
    elif 50 < frontage <= 100:
        return 2
    elif 100 < frontage <= 150:
        return 3
    return 4
_input1['LotFrontage_category'] = _input1['LotFrontage'].apply(lot_frontage_category)
_input0['LotFrontage_category'] = _input0['LotFrontage'].apply(lot_frontage_category)

def lot_area_category(area):
    if area <= 5000:
        return 1
    elif 5000 < area <= 10000:
        return 2
    elif 10000 < area <= 15000:
        return 3
    elif 15000 < area <= 20000:
        return 4
    elif 20000 < area <= 25000:
        return 5
    return 6
_input1['LotArea_category'] = _input1['LotArea'].apply(lot_area_category)
_input0['LotArea_category'] = _input0['LotArea'].apply(lot_area_category)

def year_category(year):
    if year <= 1910:
        return 1
    elif 1910 < year <= 1950:
        return 2
    elif 1950 < year <= 1980:
        return 3
    elif 1980 < year <= 2000:
        return 4
    return 5
_input1['YearBuilt_category'] = _input1['YearBuilt'].apply(year_category)
_input0['YearBuilt_category'] = _input0['YearBuilt'].apply(year_category)
_input1['YearRemodAdd_category'] = _input1['YearRemodAdd'].apply(year_category)
_input0['YearRemodAdd_category'] = _input0['YearRemodAdd'].apply(year_category)
_input1['GarageYrBlt_category'] = _input1['GarageYrBlt'].apply(year_category)
_input0['GarageYrBlt_category'] = _input0['GarageYrBlt'].apply(year_category)

def vnr_area_category(area):
    if area <= 250:
        return 1
    elif 250 < area <= 500:
        return 2
    elif 500 < area <= 750:
        return 3
    return 4
_input1['MasVnrArea_category'] = _input1['MasVnrArea'].apply(vnr_area_category)
_input0['MasVnrArea_category'] = _input0['MasVnrArea'].apply(vnr_area_category)
_input1['AllSF'] = _input1['GrLivArea'] + _input1['TotalBsmtSF']
_input0['AllSF'] = _input0['GrLivArea'] + _input0['TotalBsmtSF']

def allsf_category(area):
    if area < 1000:
        return 1
    elif 1000 < area <= 2000:
        return 2
    elif 2000 < area <= 3000:
        return 3
    elif 3000 < area <= 4000:
        return 4
    elif 4000 < area <= 5000:
        return 5
    elif 5000 < area <= 6000:
        return 6
    return 7
_input1['AllSF_category'] = _input1['AllSF'].apply(allsf_category)
_input0['AllSF_category'] = _input0['AllSF'].apply(allsf_category)
_input1 = _input1.drop(['AllSF', 'MasVnrArea', 'GarageYrBlt', 'YearRemodAdd', 'YearBuilt', 'LotArea', 'LotFrontage', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF', 'Neighborhood', 'BldgType', 'Exterior1st', 'Exterior2nd', 'MiscFeature', 'MiscVal'], axis=1)
_input0 = _input0.drop(['AllSF', 'MasVnrArea', 'GarageYrBlt', 'YearRemodAdd', 'YearBuilt', 'LotArea', 'LotFrontage', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF', 'Neighborhood', 'BldgType', 'Exterior1st', 'Exterior2nd', 'MiscFeature', 'MiscVal'], axis=1)

def object_to_int(df):
    if df.dtype == 'object':
        df = LabelEncoder().fit_transform(df)
    return df
_input1 = _input1.apply(lambda x: object_to_int(x))
_input0 = _input0.apply(lambda x: object_to_int(x))
dummy_col = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'GarageArea_category', 'GrLivArea_category', '1stFlrSF_category', '2ndFlrSF_category', 'BsmtUnfSF_category', 'LotFrontage_category', 'LotArea_category', 'YearBuilt_category', 'YearRemodAdd_category', 'GarageYrBlt_category', 'MasVnrArea_category', 'AllSF_category']
X = _input1.drop(['SalePrice'], axis=1)
y = _input1['SalePrice']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.25, random_state=12345)
std_col = ['MSSubClass', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
scaler = StandardScaler()