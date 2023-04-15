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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.columns
plt.figure(figsize=(12, 8))
sns.kdeplot(train['SalePrice'], color='darkturquoise', shade=True)
plt.grid()
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.figure(figsize=(16, 12))
sns.heatmap(train.corr(), annot=True)
plt.title('Correlation matrix')
corr_for_sale_price = train.corr()
corr_for_sale_price['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.boxplot(x=train['OverallQual'], y=train['SalePrice'])
plt.title('The boxplot for OverallQual')
plt.figure(figsize=(12, 8))
sns.boxplot(x=train['YearBuilt'], y=train['SalePrice'])
plt.title('The boxplot for YearBuilt')
plt.xticks(rotation=60)
plt.figure(figsize=(12, 8))
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])
plt.grid()
plt.title('Scatter plot for GrLivArea')
plt.figure(figsize=(12, 8))
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])

def prepare_data(df):
    if df.dtype == 'object':
        df = df.fillna('N')
    if df.dtype == 'float32' or df.dtype == 'float64':
        df = df.fillna(0)
    return df
train = train.apply(lambda x: prepare_data(x))
test = test.apply(lambda x: prepare_data(x))

def garage_area_category(area):
    if area <= 250:
        return 1
    elif 250 < area <= 500:
        return 2
    elif 500 < area <= 1000:
        return 3
    return 4
train['GarageArea_category'] = train['GarageArea'].apply(garage_area_category)
test['GarageArea_category'] = test['GarageArea'].apply(garage_area_category)

def grlivarea_category(area):
    if area <= 1000:
        return 1
    elif 1000 < area <= 2000:
        return 2
    elif 2000 < area <= 3000:
        return 3
    return 4
train['GrLivArea_category'] = train['GrLivArea'].apply(grlivarea_category)
test['GrLivArea_category'] = test['GrLivArea'].apply(grlivarea_category)

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
train['1stFlrSF_category'] = train['1stFlrSF'].apply(flrSF_and_bsmt_category)
train['2ndFlrSF_category'] = train['2ndFlrSF'].apply(flrSF_and_bsmt_category)
test['1stFlrSF_category'] = test['1stFlrSF'].apply(flrSF_and_bsmt_category)
test['2ndFlrSF_category'] = test['2ndFlrSF'].apply(flrSF_and_bsmt_category)
train['BsmtUnfSF_category'] = train['BsmtUnfSF'].apply(flrSF_and_bsmt_category)
test['BsmtUnfSF_category'] = test['BsmtUnfSF'].apply(flrSF_and_bsmt_category)

def lot_frontage_category(frontage):
    if frontage <= 50:
        return 1
    elif 50 < frontage <= 100:
        return 2
    elif 100 < frontage <= 150:
        return 3
    return 4
train['LotFrontage_category'] = train['LotFrontage'].apply(lot_frontage_category)
test['LotFrontage_category'] = test['LotFrontage'].apply(lot_frontage_category)

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
train['LotArea_category'] = train['LotArea'].apply(lot_area_category)
test['LotArea_category'] = test['LotArea'].apply(lot_area_category)

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
train['YearBuilt_category'] = train['YearBuilt'].apply(year_category)
test['YearBuilt_category'] = test['YearBuilt'].apply(year_category)
train['YearRemodAdd_category'] = train['YearRemodAdd'].apply(year_category)
test['YearRemodAdd_category'] = test['YearRemodAdd'].apply(year_category)
train['GarageYrBlt_category'] = train['GarageYrBlt'].apply(year_category)
test['GarageYrBlt_category'] = test['GarageYrBlt'].apply(year_category)

def vnr_area_category(area):
    if area <= 250:
        return 1
    elif 250 < area <= 500:
        return 2
    elif 500 < area <= 750:
        return 3
    return 4
train['MasVnrArea_category'] = train['MasVnrArea'].apply(vnr_area_category)
test['MasVnrArea_category'] = test['MasVnrArea'].apply(vnr_area_category)
train['AllSF'] = train['GrLivArea'] + train['TotalBsmtSF']
test['AllSF'] = test['GrLivArea'] + test['TotalBsmtSF']

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
train['AllSF_category'] = train['AllSF'].apply(allsf_category)
test['AllSF_category'] = test['AllSF'].apply(allsf_category)
train = train.drop(['AllSF', 'MasVnrArea', 'GarageYrBlt', 'YearRemodAdd', 'YearBuilt', 'LotArea', 'LotFrontage', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF', 'Neighborhood', 'BldgType', 'Exterior1st', 'Exterior2nd', 'MiscFeature', 'MiscVal'], axis=1)
test = test.drop(['AllSF', 'MasVnrArea', 'GarageYrBlt', 'YearRemodAdd', 'YearBuilt', 'LotArea', 'LotFrontage', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF', 'Neighborhood', 'BldgType', 'Exterior1st', 'Exterior2nd', 'MiscFeature', 'MiscVal'], axis=1)

def object_to_int(df):
    if df.dtype == 'object':
        df = LabelEncoder().fit_transform(df)
    return df
train = train.apply(lambda x: object_to_int(x))
test = test.apply(lambda x: object_to_int(x))
dummy_col = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'GarageArea_category', 'GrLivArea_category', '1stFlrSF_category', '2ndFlrSF_category', 'BsmtUnfSF_category', 'LotFrontage_category', 'LotArea_category', 'YearBuilt_category', 'YearRemodAdd_category', 'GarageYrBlt_category', 'MasVnrArea_category', 'AllSF_category']
X = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.25, random_state=12345)
std_col = ['MSSubClass', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
scaler = StandardScaler()