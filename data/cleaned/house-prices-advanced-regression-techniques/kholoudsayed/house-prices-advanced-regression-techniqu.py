import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math, time, random, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
train.shape
train.info()
test.head()
test.shape
test.info()
sample_submission.head()
sample_submission.info()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train.isnull().sum()
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])
train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])
train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train.drop(['Alley'], axis=1, inplace=True)
train.drop(['GarageYrBlt'], axis=1, inplace=True)
train.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
train.drop(['Id'], axis=1, inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train.shape
train.info()
train.head()
fig = train.hist(figsize=(9, 9))
sns.distplot(train['SalePrice'], kde=False, bins=8)
sns.lineplot(x='SaleCondition', y='SalePrice', data=train)
column_name_cat = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2', 'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

def Hot_Encoded_Cat(Nams_list, train_Dataset):
    for name in Nams_list:
        frames = pd.get_dummies(train_Dataset[name], drop_first=True)
        train_Dataset.drop([name], axis=1, inplace=True)
        train_Dataset = pd.concat([train_Dataset, frames], axis=1)
    return train_Dataset
sns.heatmap(test.isnull(), yticklabels=False, cbar=False)
test.isnull().sum()
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['FireplaceQu'] = test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])
test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])
test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])
test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])
test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())
test.drop(['Alley'], axis=1, inplace=True)
test.drop(['GarageYrBlt'], axis=1, inplace=True)
test.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
sns.heatmap(test.isnull(), yticklabels=False, cbar=False)
ConCat_DF = pd.concat([train, test], axis=0)
ConCat_DF.head()
ConCat_DF.shape
Final_Result = Hot_Encoded_Cat(column_name_cat, ConCat_DF)
Final_Result.shape
Final_Result = Final_Result.loc[:, ~Final_Result.columns.duplicated()]
Final_Result
Final_Result.shape
DataFrame_Train = Final_Result.iloc[:1422, :]
DataFrame_Test = Final_Result.iloc[1422:, :]
DataFrame_Test.drop(['SalePrice'], axis=1, inplace=True)
X_train = DataFrame_Train.drop(['SalePrice'], axis=1)
y_train = DataFrame_Train['SalePrice']
regressor = LinearRegression()