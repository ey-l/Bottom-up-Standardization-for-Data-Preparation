import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='whitegrid')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.describe()
null_col = _input1.isnull().sum() / len(_input1) * 100
null_col = null_col.sort_values(ascending=False)
null_col
corr = _input1.corr()
corr = corr.sort_values(['SalePrice'], ascending=False, inplace=False)
corr.SalePrice
train_data = _input1.drop(['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Id', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass', 'EnclosedPorch', 'GarageType', 'KitchenAbvGr', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'SalePrice'], axis=1)
test_data = _input0.drop(['BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Id', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass', 'EnclosedPorch', 'GarageType', 'KitchenAbvGr', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
categorical_features = train_data.select_dtypes(include=['object']).columns
categorical_features
numerical_features = train_data.select_dtypes(exclude=['object']).columns
numerical_features
sns.distplot(_input1['SalePrice'], color='r', kde=False)
plt.title('Distribution of Sale Price')
plt.ylabel('Number of Occurences')
plt.xlabel('Sale Price')
sns.barplot(x='SaleCondition', y='SalePrice', data=_input1)
plt.scatter(x='TotalBsmtSF', y='SalePrice', data=_input1)
plt.xlabel('Total Basement in Square Feet')
sns.catplot(x='Street', y='SalePrice', data=_input1)

def mapping(data):
    GarageCondM = {'TA': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}
    data['GarageCond'] = data['GarageCond'].map(GarageCondM)
    GarageQualM = {'TA': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}
    data['GarageQual'] = data['GarageQual'].map(GarageQualM)
    GarageFinishM = {'RFn': 0, 'Unf': 1}
    data['GarageFinish'] = data['GarageFinish'].map(GarageFinishM)
    BsmtFinType2M = {'Unf': 0, 'BLQ': 1, 'ALQ': 2, 'Rec': 3, 'LwQ': 4, 'GLQ': 5}
    data['BsmtFinType2'] = data['BsmtFinType2'].map(BsmtFinType2M)
    BsmtExposureM = {'No': 0, 'Gd': 1, 'Mn': 2, 'Av': 3}
    data['BsmtExposure'] = data['BsmtExposure'].map(BsmtExposureM)
    BsmtCondM = {'TA': 0, 'Fa': 1, 'Gd': 2}
    data['BsmtCond'] = data['BsmtCond'].map(BsmtCondM)
    BsmtQualM = {'TA': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3}
    data['BsmtQual'] = data['BsmtQual'].map(BsmtQualM)
    BsmtFinType1M = {'GLQ': 0, 'ALQ': 1, 'Unf': 2, 'Rec': 3, 'BLQ': 4, 'LwQ': 5}
    data['BsmtFinType1'] = data['BsmtFinType1'].map(BsmtFinType1M)
    MasVnrTypeM = {'BrkFace': 0, 'None': 1, 'Stone': 2, 'BrkCmn': 3, 'BLQ': 4, 'LwQ': 5}
    data['MasVnrType'] = data['MasVnrType'].map(MasVnrTypeM)
    ElectricalM = {'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'BrkCmn': 3, 'BLQ': 4, 'LwQ': 5}
    data['Electrical'] = data['Electrical'].map(ElectricalM)
    return data
train_data = mapping(train_data)
test_data = mapping(test_data)
train_data = train_data.fillna(train_data.mean())
train_data = pd.DataFrame(train_data)
test_data = test_data.fillna(test_data.mean())
test_data = pd.DataFrame(test_data)
train_data.head()
test_data.head()
train_data['train'] = 1
test_data['test'] = 0
combined = pd.concat([train_data, test_data])
combined = pd.get_dummies(combined, prefix_sep='_', columns=list(categorical_features))
combined.head()
train_data = combined[combined['train'] == 1]
test_data = combined[combined['test'] == 0]
train_data = train_data.drop(['test', 'train'], axis=1, inplace=False)
test_data = test_data.drop(['train', 'test'], axis=1, inplace=False)
train_data.head()
test_data.head()
categorical_features = train_data.select_dtypes(include=['object']).columns
categorical_features
categorical_features = test_data.select_dtypes(include=['object']).columns
categorical_features

def null(data):
    null_col = data.isnull().sum() / len(data) * 100
    null_col = null_col.sort_values(ascending=False)
    return null_col
null(test_data)
null(train_data)
X_train = train_data.copy()
Y_train = _input1['SalePrice'].values
X_test = test_data.copy()
(X_train.shape, Y_train.shape, X_test.shape)
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
gbm = xgb.XGBRegressor(colsample_bytree=0.1, gamma=0.0, learning_rate=0.01, max_depth=3, min_child_weight=0, n_estimators=10000, reg_alpha=0.0006, reg_lambda=0.6, subsample=0.7, seed=30, silent=1)