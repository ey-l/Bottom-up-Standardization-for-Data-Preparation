import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import uniform
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df = _input1.append(_input0, ignore_index=True)
(_input1.shape, _input0.shape, _input1.columns.values)
df.SalePrice = np.log(df.SalePrice)
quan = list(_input0.loc[:, _input0.dtypes != 'object'].drop('Id', axis=1).columns.values)
qual = list(_input0.loc[:, _input0.dtypes == 'object'].columns.values)
hasNAN = df[quan].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)
print('**' * 40)
hasNAN = df[qual].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)
df.LotFrontage = df.LotFrontage.fillna(df.LotFrontage.median(), inplace=False)
df.GarageYrBlt = df.GarageYrBlt.fillna(0, inplace=False)
df.MasVnrArea = df.MasVnrArea.fillna(0, inplace=False)
df.BsmtHalfBath = df.BsmtHalfBath.fillna(0, inplace=False)
df.BsmtFullBath = df.BsmtFullBath.fillna(0, inplace=False)
df.GarageArea = df.GarageArea.fillna(0, inplace=False)
df.GarageCars = df.GarageCars.fillna(0, inplace=False)
df.TotalBsmtSF = df.TotalBsmtSF.fillna(0, inplace=False)
df.BsmtUnfSF = df.BsmtUnfSF.fillna(0, inplace=False)
df.BsmtFinSF2 = df.BsmtFinSF2.fillna(0, inplace=False)
df.BsmtFinSF1 = df.BsmtFinSF1.fillna(0, inplace=False)
df.PoolQC = df.PoolQC.fillna('NA', inplace=False)
df.MiscFeature = df.MiscFeature.fillna('NA', inplace=False)
df.Alley = df.Alley.fillna('NA', inplace=False)
df.Fence = df.Fence.fillna('NA', inplace=False)
df.FireplaceQu = df.FireplaceQu.fillna('NA', inplace=False)
df.GarageCond = df.GarageCond.fillna('NA', inplace=False)
df.GarageQual = df.GarageQual.fillna('NA', inplace=False)
df.GarageFinish = df.GarageFinish.fillna('NA', inplace=False)
df.GarageType = df.GarageType.fillna('NA', inplace=False)
df.BsmtExposure = df.BsmtExposure.fillna('NA', inplace=False)
df.BsmtCond = df.BsmtCond.fillna('NA', inplace=False)
df.BsmtQual = df.BsmtQual.fillna('NA', inplace=False)
df.BsmtFinType2 = df.BsmtFinType2.fillna('NA', inplace=False)
df.BsmtFinType1 = df.BsmtFinType1.fillna('NA', inplace=False)
df.MasVnrType = df.MasVnrType.fillna('None', inplace=False)
df.Exterior2nd = df.Exterior2nd.fillna('None', inplace=False)
df.Functional = df.Functional.fillna(df.Functional.mode()[0], inplace=False)
df.Utilities = df.Utilities.fillna(df.Utilities.mode()[0], inplace=False)
df.Exterior1st = df.Exterior1st.fillna(df.Exterior1st.mode()[0], inplace=False)
df.SaleType = df.SaleType.fillna(df.SaleType.mode()[0], inplace=False)
df.KitchenQual = df.KitchenQual.fillna(df.KitchenQual.mode()[0], inplace=False)
df.Electrical = df.Electrical.fillna(df.Electrical.mode()[0], inplace=False)
for i in df.Neighborhood.unique():
    if df.MSZoning[df.Neighborhood == i].isnull().sum() > 0:
        df.loc[df.Neighborhood == i, 'MSZoning'] = df.loc[df.Neighborhood == i, 'MSZoning'].fillna(df.loc[df.Neighborhood == i, 'MSZoning'].mode()[0])
df.Alley = df.Alley.map({'NA': 0, 'Grvl': 1, 'Pave': 2})
df.BsmtCond = df.BsmtCond.map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.BsmtExposure = df.BsmtExposure.map({'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
df['BsmtFinType1'] = df['BsmtFinType1'].map({'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
df['BsmtFinType2'] = df['BsmtFinType2'].map({'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
df.BsmtQual = df.BsmtQual.map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.ExterCond = df.ExterCond.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.ExterQual = df.ExterQual.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.FireplaceQu = df.FireplaceQu.map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.Functional = df.Functional.map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})
df.GarageCond = df.GarageCond.map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.GarageQual = df.GarageQual.map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.HeatingQC = df.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.KitchenQual = df.KitchenQual.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df.LandSlope = df.LandSlope.map({'Sev': 1, 'Mod': 2, 'Gtl': 3})
df.PavedDrive = df.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})
df.PoolQC = df.PoolQC.map({'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
df.Street = df.Street.map({'Grvl': 1, 'Pave': 2})
df.Utilities = df.Utilities.map({'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4})
newquan = ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'ExterCond', 'ExterQual', 'FireplaceQu', 'Functional', 'GarageCond', 'GarageQual', 'HeatingQC', 'KitchenQual', 'LandSlope', 'PavedDrive', 'PoolQC', 'Street', 'Utilities']
quan = quan + newquan
for i in newquan:
    qual.remove(i)
df.MSSubClass = df.MSSubClass.map({20: 'class1', 30: 'class2', 40: 'class3', 45: 'class4', 50: 'class5', 60: 'class6', 70: 'class7', 75: 'class8', 80: 'class9', 85: 'class10', 90: 'class11', 120: 'class12', 150: 'class13', 160: 'class14', 180: 'class15', 190: 'class16'})
df = df.drop('MoSold', axis=1)
quan.remove('MoSold')
quan.remove('MSSubClass')
qual.append('MSSubClass')
df['Age'] = df.YrSold - df.YearBuilt
df['AgeRemod'] = df.YrSold - df.YearRemodAdd
df['AgeGarage'] = df.YrSold - df.GarageYrBlt
max_AgeGarage = np.max(df.AgeGarage[df.AgeGarage < 1000])
df['AgeGarage'] = df['AgeGarage'].map(lambda x: max_AgeGarage if x > 1000 else x)
df.Age = df.Age.map(lambda x: 0 if x < 0 else x)
df.AgeRemod = df.AgeRemod.map(lambda x: 0 if x < 0 else x)
df.AgeGarage = df.AgeGarage.map(lambda x: 0 if x < 0 else x)
df = df.drop(['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1)
for i in ['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    quan.remove(i)
quan = quan + ['Age', 'AgeRemod', 'AgeGarage']
index_drop = df.LotFrontage[df.LotFrontage > 300].index
index_drop = np.append(index_drop, df.LotArea[df.LotArea > 100000].index)
index_drop = np.append(index_drop, df.BsmtFinSF1[df.BsmtFinSF1 > 4000].index)
index_drop = np.append(index_drop, df.TotalBsmtSF[df.TotalBsmtSF > 6000].index)
index_drop = np.append(index_drop, df['1stFlrSF'][df['1stFlrSF'] > 4000].index)
index_drop = np.append(index_drop, df.GrLivArea[(df.GrLivArea > 4000) & (df.SalePrice < 13)].index)
index_drop = np.unique(index_drop)
index_drop = index_drop[index_drop < 1460]
df = df.drop(index_drop).reset_index(drop=True)
print('{} examples in the training set are dropped.'.format(len(index_drop)))
for i in quan:
    print(i + ': {}'.format(round(skew(df[i]), 2)))
skewed_features = np.array(quan)[np.abs(skew(df[quan])) > 0.5]
df[skewed_features] = np.log1p(df[skewed_features])
dummy_drop = []
for i in qual:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]
df = pd.get_dummies(df, columns=qual)
df = df.drop(dummy_drop, axis=1)
X_train = df[:-1459].drop(['SalePrice', 'Id'], axis=1)
y_train = df[:-1459]['SalePrice']
X_test = df[-1459:].drop(['SalePrice', 'Id'], axis=1)
scaler = RobustScaler()
X_train[quan] = scaler.fit_transform(X_train[quan])
X_test[quan] = scaler.transform(X_test[quan])
(X_train.shape, X_test.shape)
xgb = XGBRegressor()