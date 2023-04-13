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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head()
_input1.shape
_input1.info()
_input0.head()
_input0.shape
_input0.info()
_input2.head()
_input2.info()
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
_input1.isnull().sum()
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(_input1['FireplaceQu'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1 = _input1.drop(['Alley'], axis=1, inplace=False)
_input1 = _input1.drop(['GarageYrBlt'], axis=1, inplace=False)
_input1 = _input1.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input1 = _input1.dropna(inplace=False)
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
_input1.shape
_input1.info()
_input1.head()
fig = _input1.hist(figsize=(9, 9))
sns.distplot(_input1['SalePrice'], kde=False, bins=8)
sns.lineplot(x='SaleCondition', y='SalePrice', data=_input1)
column_name_cat = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2', 'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

def Hot_Encoded_Cat(Nams_list, train_Dataset):
    for name in Nams_list:
        frames = pd.get_dummies(train_Dataset[name], drop_first=True)
        train_Dataset = train_Dataset.drop([name], axis=1, inplace=False)
        train_Dataset = pd.concat([train_Dataset, frames], axis=1)
    return train_Dataset
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False)
_input0.isnull().sum()
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean())
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna(_input0['FireplaceQu'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean())
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(_input0['BsmtFinSF2'].mean())
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].mean())
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].mean())
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mode()[0])
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].mode()[0])
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].mean())
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].mean())
_input0 = _input0.drop(['Alley'], axis=1, inplace=False)
_input0 = _input0.drop(['GarageYrBlt'], axis=1, inplace=False)
_input0 = _input0.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=False)
ConCat_DF = pd.concat([_input1, _input0], axis=0)
ConCat_DF.head()
ConCat_DF.shape
Final_Result = Hot_Encoded_Cat(column_name_cat, ConCat_DF)
Final_Result.shape
Final_Result = Final_Result.loc[:, ~Final_Result.columns.duplicated()]
Final_Result
Final_Result.shape
DataFrame_Train = Final_Result.iloc[:1422, :]
DataFrame_Test = Final_Result.iloc[1422:, :]
DataFrame_Test = DataFrame_Test.drop(['SalePrice'], axis=1, inplace=False)
X_train = DataFrame_Train.drop(['SalePrice'], axis=1)
y_train = DataFrame_Train['SalePrice']
regressor = LinearRegression()