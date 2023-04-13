import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.sample(5)
print('The DataFrame has ' + str(_input1.shape[0]) + ' samples and ' + str(_input1.shape[1]) + ' columns')
print('Duplicate entries in the dataset: ' + str(_input1.duplicated().sum()))
_input1.info()
import missingno as msno
msno.matrix(_input1, labels=True)
_input1.isna().mean().sort_values().plot(kind='bar', figsize=(15, 4))
plt.title('Percentage of missing values per feature', fontweight='bold')
_input1['PoolQC'].value_counts()
NA_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']
_input1[NA_columns] = _input1[NA_columns].fillna(value='NA')
mcv_columns = ['MasVnrArea', 'Electrical']
for col in mcv_columns:
    _input1[col] = _input1[col].fillna(value=_input1[col].value_counts().index[0])
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['YearBuilt'], inplace=False)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean(), inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(value='None', inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mean(), inplace=False)
_input1[_input1.isna().any(axis=1)]
_input1.plot(lw=0, marker='.', subplots=True, layout=(-1, 4), figsize=(15, 30), markersize=1)
plt.tight_layout()
_input1['MS SubClass'] = _input1['MSSubClass'].astype(str)
categorical = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
import seaborn as sns
(fig, axes) = plt.subplots(nrows=12, ncols=4, figsize=(18, 45), sharey=False)
for i in range(len(categorical)):
    col = int(i / 4)
    row = i % 4
    sns.violinplot(x=_input1[categorical[i]], y=_input1['SalePrice'], ax=axes[col, row])
    axes[col, row].set_xticklabels(axes[col, row].get_xticklabels(), rotation=90)
plt.tight_layout()
ordinal = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
cols_replace = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for col in cols_replace:
    _input1[col] = _input1[col].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}, inplace=False)
dummy_ordinal = ['LotShape', 'Utilities', 'LandSlope', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence']
encoded_df = pd.get_dummies(_input1, columns=nominal + dummy_ordinal)
print('The encoded DataFrame has ' + str(encoded_df.shape[0]) + ' samples and ' + str(encoded_df.shape[1]) + ' columns')
continuous = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
encoded_df[continuous].hist(layout=(-1, 4), bins=30, grid=False, figsize=(15, 10))
plt.suptitle('Distribution of continous variables', size=15)
plt.tight_layout()
continuous_to_encode = ['LotFrontage', 'LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']
encoded_df[continuous_to_encode] = np.log1p(encoded_df[continuous_to_encode])
encoded_df[continuous_to_encode].hist(layout=(-1, 6), bins=30, grid=False, figsize=(15, 3))
plt.suptitle('Skewed variables that have been converted with logaritmic transormation', size=15)
plt.tight_layout()
(fig, axes) = plt.subplots(nrows=5, ncols=4, figsize=(12, 16), sharey=True)
plt.suptitle('Relationship of continuous variables with the house price', size=15)
for i in range(len(continuous)):
    col = int(i / 4)
    row = i % 4
    sns.scatterplot(x=encoded_df[continuous[i]], y=encoded_df['SalePrice'], ax=axes[col, row])
plt.tight_layout()
continuous_poly_transform = ['GrLivArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea']

def feature_transformation(df):
    for c in continuous_poly_transform:
        for d in [0.5, 2, 3]:
            name = '{}**{}'.format(c, d)
            _input1[name] = _input1[c] ** d
    return _input1
encoded_df = feature_transformation(encoded_df)
encoded_df.head()
train = encoded_df.sample(frac=0.7, random_state=0)
test = encoded_df.drop(train.index)
x_tr = train.drop(['Id', 'SalePrice'], axis=1)
x_val = test.drop(['Id', 'SalePrice'], axis=1)
y_tr = train.SalePrice.values
y_val = test.SalePrice.values
print('Training sets:', x_tr.shape, y_tr.shape)
print('Validation sets:', x_val.shape, y_val.shape)

def RMSE(y, y_pred):
    mse = np.mean(np.square(y - y_pred))
    return np.sqrt(mse)
rmse_baseline = RMSE(y_val, np.median(y_tr))
print('RMSE baseline: {:.3f}$'.format(rmse_baseline))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_tr_rescaled = scaler.fit_transform(x_tr)
x_val_rescaled = scaler.transform(x_val)
y_tr_rescaled = np.log10(y_tr)
y_val_rescaled = np.log10(y_val)
from sklearn.linear_model import Ridge
gs_results = []
for alpha in np.logspace(-4, 8, num=50):
    ridge = Ridge(alpha=alpha)