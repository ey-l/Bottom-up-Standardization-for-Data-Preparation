import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1.info()
_input1.head()
_input1.corr().sort_values(by='SalePrice')['SalePrice']
X_train_num = _input1[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'SalePrice']]
X_train_num.info()
X_test_num = _input0[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]
X_test_num.info()
X_train_cat = _input1.select_dtypes(include=['object'])
X_train_cat.info()
X_train_cat.head()

def replaceNaN(df):
    df = df.fillna({'Alley': 'No alley access', 'BsmtQual': 'No Basement', 'BsmtCond': 'No Basement', 'BsmtExposure': 'No Basement', 'BsmtFinType1': 'No Basement', 'BsmtFinType2': 'No Basement', 'FireplaceQu': 'No Fireplace', 'GarageType': 'No Garage', 'GarageFinish': 'No Garage', 'GarageQual': 'No Garage', 'GarageCond': 'No Garage', 'PoolQC': 'No Pool', 'Fence': 'No Fence', 'MiscFeature': 'None'}, inplace=False)
X_train_cat.head()
replaceNaN(X_train_cat)
X_train_cat.info()
X_test_cat = _input0.select_dtypes(include=['object'])
X_test_cat.info()
replaceNaN(X_test_cat)
X_test_cat.info()
train_id = X_train_cat.index
test_id = X_test_cat.index
data = pd.concat([X_train_cat, X_test_cat])
data.head()
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
colsOrdinal = ['LotShape', 'Utilities', 'LandSlope', 'HouseStyle', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']
ordinal = OrdinalEncoder()
data[colsOrdinal] = pd.DataFrame(ordinal.fit_transform(data[colsOrdinal]), columns=colsOrdinal)
data.head()
cols = list(set(data.columns) - set(colsOrdinal))
oneHot = OneHotEncoder()
oneHotArr = oneHot.fit_transform(data[cols]).toarray()
labels = np.array(oneHot.get_feature_names_out()).ravel()
OneHotData = pd.DataFrame(oneHotArr, columns=labels, index=data.index)
OneHotData.head()
data = pd.concat([data[colsOrdinal], OneHotData], axis=1)
data.head()
X_train_cat = data.loc[train_id]
X_train_cat
X_test_cat = data.loc[test_id]
X_test_cat
X_train_cat.index = X_train_num.index
X_train_prepared = pd.concat([X_train_cat, X_train_num], axis=1)
X_train_prepared.head()
X_test_cat.index = X_test_num.index
X_test_prepared = pd.concat([X_test_cat, X_test_num], axis=1)
X_test_prepared.head()
train_labels = pd.DataFrame(X_train_prepared['SalePrice'], columns=['SalePrice'])
print(train_labels.head())
X_train_prepared = X_train_prepared.drop(['SalePrice'], axis=1)

def normalize(train, test):
    train_normalized = train.copy()
    test_normalized = test.copy()
    for column in train_normalized.columns:
        mini = min(train_normalized[column].min(), test_normalized[column].min())
        maxi = max(train_normalized[column].max(), test_normalized[column].max())
        train_normalized[column] = (train_normalized[column] - mini) / (maxi - mini)
        test_normalized[column] = (test_normalized[column] - mini) / (maxi - mini)
    return (train_normalized, test_normalized)
cols = X_train_prepared.columns
(train, test) = normalize(X_train_prepared, X_test_prepared)
X_train_prepared = pd.DataFrame(train, columns=cols)
X_test_prepared = pd.DataFrame(test, columns=cols)
print(X_train_prepared.head())
print(X_test_prepared.head())
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')