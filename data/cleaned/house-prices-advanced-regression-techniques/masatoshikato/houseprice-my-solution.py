import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import scipy
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
PATH = '_data/input/house-prices-advanced-regression-techniques/'
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
sample = pd.read_csv(PATH + 'sample_submission.csv')
train['SalePrice'] = np.log1p(train['SalePrice'])
plt.figure(figsize=(10, 4))
sns.distplot(train['SalePrice'], fit=norm, fit_kws={'color': 'tomato', 'label': 'norm'})
plt.ylabel('')
plt.legend()

train['LotFrontage'] = train['LotFrontage'].fillna(0.0)
train['Alley'] = train['Alley'].fillna('NaN')
train['BsmtQual'] = train['BsmtQual'].fillna('NaN')
train['BsmtCond'] = train['BsmtCond'].fillna('NaN')
train['BsmtExposure'] = train['BsmtExposure'].fillna('NaN')
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NaN')
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('NaN')
train['FireplaceQu'] = train['FireplaceQu'].fillna('NaN')
train['GarageType'] = train['GarageType'].fillna('NaN')
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)
train['GarageFinish'] = train['GarageFinish'].fillna('NaN')
train['GarageQual'] = train['GarageQual'].fillna('NaN')
train['GarageCond'] = train['GarageCond'].fillna('NaN')
train['PoolQC'] = train['PoolQC'].fillna('NaN')
train['Fence'] = train['Fence'].fillna('NaN')
train['MiscFeature'] = train['MiscFeature'].fillna('NaN')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0.0)
test['LotFrontage'] = test['LotFrontage'].fillna(0.0)
test['Alley'] = test['Alley'].fillna('NaN')
test['BsmtQual'] = test['BsmtQual'].fillna('NaN')
test['BsmtCond'] = test['BsmtCond'].fillna('NaN')
test['BsmtExposure'] = test['BsmtExposure'].fillna('NaN')
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('NaN')
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('NaN')
test['FireplaceQu'] = test['FireplaceQu'].fillna('NaN')
test['GarageType'] = test['GarageType'].fillna('NaN')
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)
test['GarageFinish'] = test['GarageFinish'].fillna('NaN')
test['GarageQual'] = test['GarageQual'].fillna('NaN')
test['GarageCond'] = test['GarageCond'].fillna('NaN')
test['PoolQC'] = test['PoolQC'].fillna('NaN')
test['Fence'] = test['Fence'].fillna('NaN')
test['MiscFeature'] = test['MiscFeature'].fillna('NaN')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0.0)
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0.0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0.0)
train_detchd = train[train['GarageType'] == 'Detchd']
test_detchd = test[test['GarageType'] == 'Detchd']
train['GarageArea'] = train['GarageArea'].fillna(train_detchd['GarageArea'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test_detchd['GarageArea'].mean())
np.abs(train.corr()['SalePrice']).sort_values(ascending=True).head(15)
test.isnull().sum()[test.isnull().sum() > 0]
train.drop(['Id', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType'], axis=1, inplace=True)
test.drop(['Id', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType'], axis=1, inplace=True)
train.dtypes[train.dtypes == 'object'].index
oe = OrdinalEncoder()
encoded = oe.fit_transform(train[['Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition']].values)
train[['Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition']] = encoded
encoded = oe.fit_transform(test[['Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition']].values)
test[['Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition']] = encoded

def check(df, column):
    col = np.abs(test.corr()[column])
    print(col.sort_values(ascending=False).head(13))

def missing_value_classify(df, column, column2, column3, column4, column5, column6, column7):
    target = df[[column, column2, column3, column4, column5, column6, column7]]
    notnull = target[target[column].notnull()].values
    null = target[target[column].isnull()].values
    X = notnull[:, 1:]
    y = notnull[:, 0]
    rf = RandomForestClassifier(random_state=0, n_estimators=1000, n_jobs=-1)