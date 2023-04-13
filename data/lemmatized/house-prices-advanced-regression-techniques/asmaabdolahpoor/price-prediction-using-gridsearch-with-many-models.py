import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
np.set_printoptions(threshold=sys.maxsize)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
print(_input0.shape)
print(_input1.shape)
_input1.head()
_input1.isnull().sum().nlargest(20)
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)
_input0.isnull().sum().nlargest(20)
_input0 = _input0.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)
top_corr_features = _input1.corr()['SalePrice'].sort_values(ascending=False)
top_corr_features
Correlation_Matrix = _input1.corr()
Correlation_Matrix.style.background_gradient(cmap='coolwarm')
_input1.isnull().sum().nlargest(15)
low_corr_cols = ['MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold']
_input1 = _input1.drop(low_corr_cols, axis=1)
_input0 = _input0.drop(low_corr_cols, axis=1)
categorical_feature = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterCond', 'BsmtCond', 'Heating', 'Functional', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
ordinal_feature = ['BsmtQual', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual']
numerical_feature = ['LotFrontage', 'LotArea', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'FullBath', 'TotRmsAbvGrd', 'ScreenPorch', 'PoolArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'TotalBsmtSF', 'GarageCars', 'MasVnrArea']
all_features = categorical_feature + ordinal_feature + numerical_feature

def Impute_Data(df, type_df):
    if type_df == 'numerical':
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')