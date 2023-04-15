import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgbm

def read_file(train_filePath, test_filePath):
    train = pd.read_csv(train_filePath)
    test = pd.read_csv(test_filePath)
    Id = test['Id']
    y = train['SalePrice']
    train.drop('SalePrice', axis=1, inplace=True)
    allHousing_data = pd.concat([train, test])
    return (Id, y, allHousing_data)
train_filePath = '_data/input/house-prices-advanced-regression-techniques/train.csv'
test_filePath = '_data/input/house-prices-advanced-regression-techniques/test.csv'
(Id, y, allHousing_data) = read_file(train_filePath, test_filePath)
print(allHousing_data.shape)
print(allHousing_data.head())
print(allHousing_data.isnull().sum())
print(allHousing_data.describe())

def data_cleaning(allHousing_data):
    missingValue_FillingWithMode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'KitchenQual', 'Functional', 'SaleType', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars']
    for i in missingValue_FillingWithMode:
        allHousing_data[i].fillna(allHousing_data[i].mode()[0], inplace=True)
    missingValue_FillingWithNone = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    allHousing_data[missingValue_FillingWithNone] = allHousing_data[missingValue_FillingWithNone].fillna('None')
    missingValue_FillingWithMedian = ['LotFrontage', 'MasVnrArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1', 'GarageArea']
    for i in missingValue_FillingWithMedian:
        allHousing_data[i].fillna(allHousing_data[i].median(), inplace=True)
    allHousing_data['GarageYrBlt'] = allHousing_data['GarageYrBlt'].fillna(allHousing_data['YearRemodAdd'])
    return allHousing_data
allHousing_data = data_cleaning(allHousing_data)

def feature_eng(allHousing_data):
    allHousing_data['House_age'] = allHousing_data['YrSold'] - allHousing_data['YearBuilt']
    allHousing_data['Rmod_age'] = allHousing_data['YrSold'] - allHousing_data['YearRemodAdd']
    skewness = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'House_age', 'Rmod_age', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'TotalBsmtSF', 'LowQualFinSF']
    for i in skewness:
        if allHousing_data[i].skew(axis=0) >= 1 or allHousing_data[i].skew(axis=0) <= -1:
            allHousing_data[i] = np.log10(allHousing_data[i] + 1)
    convert_val = list(allHousing_data.select_dtypes(include='object').columns.values)
    ordinal_encoder = OrdinalEncoder()
    allHousing_data[convert_val] = ordinal_encoder.fit_transform(allHousing_data[convert_val])
    return allHousing_data
allHousing_data = feature_eng(allHousing_data)

def feature_selection(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_regression, k=60)