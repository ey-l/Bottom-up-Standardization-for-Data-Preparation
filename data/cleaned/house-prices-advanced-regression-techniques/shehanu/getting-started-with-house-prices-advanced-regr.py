import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_dataset.head()
train_dataset.shape
train_dataset.info()
NaN_1 = train_dataset.isnull().sum()
print(NaN_1[NaN_1 > 0])
train_dataset = train_dataset.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)
selcted_col = np.array(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])
for col in selcted_col:
    train_dataset[col] = train_dataset[col].fillna(train_dataset[col].mean())
selcted_col = np.array(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'Electrical'])
for col in selcted_col:
    train_dataset[col] = train_dataset[col].fillna(train_dataset[col].mode()[0])
np.array(train_dataset.isnull().sum())
train_dataset.head()
test_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_dataset.head()
test_dataset.shape
test_dataset.info()
NaN_ = test_dataset.isnull().sum()
print(NaN_[NaN_ > 0])
test_dataset = test_dataset.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)
selcted_col3 = np.array(['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'SaleType'])
for col3 in selcted_col3:
    test_dataset[col3] = test_dataset[col3].fillna(test_dataset[col3].mode()[0])
selcted_col4 = np.array(['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageYrBlt', 'GarageArea', 'GarageCars'])
for column in selcted_col4:
    test_dataset[column] = test_dataset[column].fillna(test_dataset[column].mean())
np.array(test_dataset.isnull().sum())
test_dataset.info()
test_dataset.head()
full_train = pd.concat([train_dataset, test_dataset], axis=0)
full_train.shape
full_train.head()
categorical_cols = full_train.select_dtypes('object').columns.to_list()
categorical_cols
train_full = full_train
i = 0
for fields in categorical_cols:
    df1 = pd.get_dummies(full_train[fields], drop_first=True)
    full_train.drop([fields], axis=1, inplace=True)
    if i == 0:
        train_full = df1.copy()
    else:
        train_full = pd.concat([train_full, df1], axis=1)
    i = i + 1
full_train = pd.concat([full_train, train_full], axis=1)
full_train = full_train.loc[:, ~full_train.columns.duplicated()]
full_train.head()
new_train_dataset = full_train.iloc[:1459, :]
new_test_dataset = full_train.iloc[1460:, :]
(new_train_dataset.shape, new_test_dataset.shape)
import xgboost
model = xgboost.XGBRegressor()
y_train = new_train_dataset['SalePrice']
X_train = new_train_dataset.drop(['SalePrice'], axis=1)