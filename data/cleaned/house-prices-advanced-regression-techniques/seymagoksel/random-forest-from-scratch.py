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
from ycimpute.imputer import knnimput, MICE
from sklearn.impute import SimpleImputer
import math
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_salePrice = train['SalePrice']
all_data = pd.concat([train, test])
missing_categorical_cols = ['BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType2', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC']
for col in missing_categorical_cols:
    col = all_data[col]
    for index in range(col.size):
        if pd.isnull(col.iloc[index]):
            col.iloc[index] = 'NA'
        else:
            continue
garageYrBlt = all_data['GarageYrBlt']
for index in range(garageYrBlt.size):
    if np.isnan(garageYrBlt.iloc[index]):
        garageYrBlt.iloc[index] = 0.0
    else:
        continue
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['BsmtFinType1'] = mean_f.fit_transform(all_data[['BsmtFinType1']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['Electrical'] = mean_f.fit_transform(all_data[['Electrical']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['MSZoning'] = mean_f.fit_transform(all_data[['MSZoning']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['Functional'] = mean_f.fit_transform(all_data[['Functional']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['Utilities'] = mean_f.fit_transform(all_data[['Utilities']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['BsmtHalfBath'] = mean_f.fit_transform(all_data[['BsmtHalfBath']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['BsmtFullBath'] = mean_f.fit_transform(all_data[['BsmtFullBath']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['KitchenQual'] = mean_f.fit_transform(all_data[['KitchenQual']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['Exterior2nd'] = mean_f.fit_transform(all_data[['Exterior2nd']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['Exterior1st'] = mean_f.fit_transform(all_data[['Exterior1st']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['BsmtFinSF1'] = mean_f.fit_transform(all_data[['BsmtFinSF1']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['SaleType'] = mean_f.fit_transform(all_data[['SaleType']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['TotalBsmtSF'] = mean_f.fit_transform(all_data[['TotalBsmtSF']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['BsmtUnfSF'] = mean_f.fit_transform(all_data[['BsmtUnfSF']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['BsmtFinSF2'] = mean_f.fit_transform(all_data[['BsmtFinSF2']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['GarageCars'] = mean_f.fit_transform(all_data[['GarageCars']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['GarageArea'] = mean_f.fit_transform(all_data[['GarageArea']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['MasVnrType'] = mean_f.fit_transform(all_data[['MasVnrType']])
mean_f = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
all_data['MasVnrArea'] = mean_f.fit_transform(all_data[['MasVnrArea']])
masVnrType = all_data['MasVnrType']
for index in range(masVnrType.size):
    if masVnrType.iloc[index] == 'None':
        masVnrType.iloc[index] = 'NA'
    else:
        continue
all_data['MasVnrType'] = masVnrType
centralAir = all_data['CentralAir']
for index in range(centralAir.size):
    if centralAir.iloc[index] == 'N':
        centralAir.iloc[index] = 0
    else:
        centralAir.iloc[index] = 1
all_data['CentralAir'] = centralAir
alley = all_data['Alley']
HasAlley = []
AlleyGrvl = []
AlleyPave = []
for index in range(alley.size):
    if alley.iloc[index] == 'NA':
        HasAlley.append(0)
    else:
        HasAlley.append(1)
    if alley.iloc[index] == 'Grvl':
        AlleyGrvl.append(1)
    else:
        AlleyGrvl.append(0)
    if alley.iloc[index] == 'Pave':
        AlleyPave.append(1)
    else:
        AlleyPave.append(0)
all_data['HasAlley'] = HasAlley
all_data['AlleyGrvl'] = AlleyGrvl
all_data['AlleyPave'] = AlleyPave
all_data = all_data.drop('Alley', axis='columns')
BsmtQual_ = all_data['BsmtQual']
BsmtQual = []
HasBsmt = []
for index in range(BsmtQual_.size):
    if BsmtQual_.iloc[index] == 'NA':
        BsmtQual.append(0)
        HasBsmt.append(0)
    else:
        HasBsmt.append(1)
    if BsmtQual_.iloc[index] == 'Ex':
        BsmtQual.append(105)
    if BsmtQual_.iloc[index] == 'Gd':
        BsmtQual.append(95)
    if BsmtQual_.iloc[index] == 'TA':
        BsmtQual.append(85)
    if BsmtQual_.iloc[index] == 'Fa':
        BsmtQual.append(75)
    if BsmtQual_.iloc[index] == 'Po':
        BsmtQual.append(65)
all_data['HasBsmt'] = HasBsmt
all_data['BsmtQual'] = BsmtQual
BsmtCond_ = all_data['BsmtCond']
BsmtCond = []
for index in range(BsmtCond_.size):
    if BsmtCond_.iloc[index] == 'NA':
        BsmtCond.append(0)
    if BsmtCond_.iloc[index] == 'Ex':
        BsmtCond.append(5)
    if BsmtCond_.iloc[index] == 'Gd':
        BsmtCond.append(4)
    if BsmtCond_.iloc[index] == 'TA':
        BsmtCond.append(3)
    if BsmtCond_.iloc[index] == 'Fa':
        BsmtCond.append(2)
    if BsmtCond_.iloc[index] == 'Po':
        BsmtCond.append(1)
all_data['BsmtCond'] = BsmtCond
GarageQual_ = all_data['GarageQual']
HasGarage = []
GarageQual = []
for index in range(GarageQual_.size):
    if GarageQual_.iloc[index] == 'NA':
        HasGarage.append(0)
        GarageQual.append(0)
    else:
        HasGarage.append(1)
    if GarageQual_.iloc[index] == 'Ex':
        GarageQual.append(5)
    if GarageQual_.iloc[index] == 'Gd':
        GarageQual.append(4)
    if GarageQual_.iloc[index] == 'TA':
        GarageQual.append(3)
    if GarageQual_.iloc[index] == 'Fa':
        GarageQual.append(2)
    if GarageQual_.iloc[index] == 'Po':
        GarageQual.append(1)
all_data['GarageQual'] = GarageQual
all_data['HasGarage'] = HasGarage
GarageCond_ = all_data['GarageCond']
GarageCond = []
for index in range(GarageCond_.size):
    if GarageCond_.iloc[index] == 'NA':
        GarageCond.append(0)
    if GarageCond_.iloc[index] == 'Ex':
        GarageCond.append(5)
    if GarageCond_.iloc[index] == 'Gd':
        GarageCond.append(4)
    if GarageCond_.iloc[index] == 'TA':
        GarageCond.append(3)
    if GarageCond_.iloc[index] == 'Fa':
        GarageCond.append(2)
    if GarageCond_.iloc[index] == 'Po':
        GarageCond.append(1)
all_data['GarageCond'] = GarageCond
PoolQC_ = all_data['PoolQC']
HasPool = []
PoolQC = []
for index in range(PoolQC_.size):
    if PoolQC_.iloc[index] == 'NA':
        HasPool.append(0)
        PoolQC.append(0)
    else:
        HasPool.append(1)
    if PoolQC_.iloc[index] == 'Ex':
        PoolQC.append(4)
    if PoolQC_.iloc[index] == 'Gd':
        PoolQC.append(3)
    if PoolQC_.iloc[index] == 'TA':
        PoolQC.append(2)
    if PoolQC_.iloc[index] == 'Fa':
        PoolQC.append(1)
all_data['PoolQC'] = PoolQC
all_data['HasPool'] = HasPool
FireplaceQu_ = all_data['FireplaceQu']
HasFireplace = []
FireplaceQu = []
for index in range(FireplaceQu_.size):
    if FireplaceQu_.iloc[index] == 'NA':
        HasFireplace.append(0)
        FireplaceQu.append(0)
    else:
        HasFireplace.append(1)
    if FireplaceQu_.iloc[index] == 'Ex':
        FireplaceQu.append(5)
    if FireplaceQu_.iloc[index] == 'Gd':
        FireplaceQu.append(4)
    if FireplaceQu_.iloc[index] == 'TA':
        FireplaceQu.append(3)
    if FireplaceQu_.iloc[index] == 'Fa':
        FireplaceQu.append(2)
    if FireplaceQu_.iloc[index] == 'Po':
        FireplaceQu.append(1)
all_data['FireplaceQu'] = FireplaceQu
all_data['HasFireplace'] = HasFireplace
Fence_ = all_data['Fence']
HasFence = []
Fence = []
for index in range(Fence_.size):
    if Fence_.iloc[index] == 'NA':
        HasFence.append(0)
        Fence.append(0)
    else:
        HasFence.append(1)
    if Fence_.iloc[index] == 'GdPrv':
        Fence.append(4)
    if Fence_.iloc[index] == 'MnPrv':
        Fence.append(3)
    if Fence_.iloc[index] == 'GdWo':
        Fence.append(2)
    if Fence_.iloc[index] == 'MnWw':
        Fence.append(1)
ExterQual_ = all_data['ExterQual']
ExterQual = []
for index in range(ExterQual_.size):
    if ExterQual_.iloc[index] == 'NA':
        ExterQual.append(0)
    if ExterQual_.iloc[index] == 'Ex':
        ExterQual.append(5)
    if ExterQual_.iloc[index] == 'Gd':
        ExterQual.append(4)
    if ExterQual_.iloc[index] == 'TA':
        ExterQual.append(3)
    if ExterQual_.iloc[index] == 'Fa':
        ExterQual.append(1)
    if ExterQual_.iloc[index] == 'Po':
        ExterQual.append(0)
all_data['ExterQual'] = ExterQual
ExterCond_ = all_data['ExterCond']
ExterCond = []
for index in range(ExterCond_.size):
    if ExterCond_.iloc[index] == 'NA':
        ExterCond.append(0)
    if ExterCond_.iloc[index] == 'Ex':
        ExterCond.append(5)
    if ExterCond_.iloc[index] == 'Gd':
        ExterCond.append(4)
    if ExterCond_.iloc[index] == 'TA':
        ExterCond.append(3)
    if ExterCond_.iloc[index] == 'Fa':
        ExterCond.append(2)
    if ExterCond_.iloc[index] == 'Po':
        ExterCond.append(1)
all_data['ExterCond'] = ExterCond
HeatingQC_ = all_data['HeatingQC']
HeatingQC = []
for index in range(HeatingQC_.size):
    if HeatingQC_.iloc[index] == 'NA':
        HeatingQC.append(0)
    if HeatingQC_.iloc[index] == 'Ex':
        HeatingQC.append(5)
    if HeatingQC_.iloc[index] == 'Gd':
        HeatingQC.append(4)
    if HeatingQC_.iloc[index] == 'TA':
        HeatingQC.append(3)
    if HeatingQC_.iloc[index] == 'Fa':
        HeatingQC.append(2)
    if HeatingQC_.iloc[index] == 'Po':
        HeatingQC.append(1)
all_data['ExterCond'] = ExterCond
KitchenQual_ = all_data['KitchenQual']
KitchenQual = []
for index in range(KitchenQual_.size):
    if KitchenQual_.iloc[index] == 'NA':
        KitchenQual.append(0)
    if KitchenQual_.iloc[index] == 'Ex':
        KitchenQual.append(5)
    if KitchenQual_.iloc[index] == 'Gd':
        KitchenQual.append(4)
    if KitchenQual_.iloc[index] == 'TA':
        KitchenQual.append(3)
    if KitchenQual_.iloc[index] == 'Fa':
        KitchenQual.append(2)
    if KitchenQual_.iloc[index] == 'Po':
        KitchenQual.append(1)
all_data['KitchenQual'] = KitchenQual
all_data_copy = all_data.copy()
all_data_copy = all_data.drop('Id', axis='columns')
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorial_cols = []
for col in all_data_copy.columns:
    if all_data_copy[col].dtype not in numeric_dtypes:
        categorial_cols.append(col)
    else:
        continue
all_data_dummy = pd.get_dummies(all_data_copy[categorial_cols])
all_data_copy_ = all_data.drop(categorial_cols, axis='columns')
all_data_dummy = pd.concat([all_data_copy_, all_data_dummy], axis=1)
var_names = list(all_data_dummy)
array_all_data = np.array(all_data_dummy)
all_data_knn_dummy = knnimput.KNN(k=4).complete(array_all_data)
all_data_knn_dummy = pd.DataFrame(all_data_knn_dummy, columns=var_names)
all_data_knn = all_data.copy()
all_data_knn['LotFrontage'] = all_data_knn_dummy['LotFrontage']
salePrice = []
for i in range(1459):
    salePrice.append(0)
X_knn = all_data_knn_dummy
y = list(train_salePrice) + salePrice
y = pd.Series(y)
clf = ExtraTreesClassifier(n_estimators=10)