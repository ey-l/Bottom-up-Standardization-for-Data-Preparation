import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']), ignore_index=True)
x = all_data.loc[np.logical_not(all_data['LotFrontage'].isnull()), 'LotArea']
y = all_data.loc[np.logical_not(all_data['LotFrontage'].isnull()), 'LotFrontage']
t = (x <= 25000) & (y <= 150)
p = np.polyfit(x[t], y[t], 1)
all_data.loc[all_data['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, all_data.loc[all_data['LotFrontage'].isnull(), 'LotArea'])
all_data.loc[all_data.Alley.isnull(), 'Alley'] = 'NoAlley'
all_data.loc[all_data.MasVnrType.isnull(), 'MasVnrType'] = 'None'
all_data.loc[all_data.MasVnrType == 'None', 'MasVnrArea'] = 0
all_data.loc[all_data.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'
all_data.loc[all_data.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'
all_data.loc[all_data.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'
all_data.loc[all_data.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'
all_data.loc[all_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
all_data.loc[all_data.BsmtFinType1 == 'NoBsmt', 'BsmtFinSF1'] = 0
all_data.loc[all_data.BsmtFinType2 == 'NoBsmt', 'BsmtFinSF2'] = 0
all_data.loc[all_data.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = all_data.BsmtFinSF1.median()
all_data.loc[all_data.BsmtQual == 'NoBsmt', 'BsmtUnfSF'] = 0
all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = all_data.BsmtUnfSF.median()
all_data.loc[all_data.BsmtQual == 'NoBsmt', 'TotalBsmtSF'] = 0
all_data.loc[all_data.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'
all_data.loc[all_data.GarageType.isnull(), 'GarageType'] = 'NoGarage'
all_data.loc[all_data.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'
all_data.loc[all_data.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'
all_data.loc[all_data.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
all_data.loc[all_data.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
all_data.loc[all_data.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0
all_data.loc[all_data.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
all_data.loc[all_data.MSZoning.isnull(), 'MSZoning'] = 'RL'
all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = 'AllPub'
all_data.loc[all_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
all_data.loc[all_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
all_data.loc[all_data.Functional.isnull(), 'Functional'] = 'Typ'
all_data.loc[all_data.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
all_data.loc[all_data.SaleCondition.isnull(), 'SaleType'] = 'WD'
all_data.loc[all_data['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'
all_data.loc[all_data['Fence'].isnull(), 'Fence'] = 'NoFence'
all_data.loc[all_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
all_data.loc[all_data['GarageArea'].isnull(), 'GarageArea'] = all_data.loc[all_data['GarageType'] == 'Detchd', 'GarageArea'].mean()
all_data.loc[all_data['GarageCars'].isnull(), 'GarageCars'] = all_data.loc[all_data['GarageType'] == 'Detchd', 'GarageCars'].median()
all_data = all_data.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0}, 'Street': {'Pave': 1, 'Grvl': 0}, 'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoFireplace': 0}, 'Fence': {'GdPrv': 2, 'GdWo': 2, 'MnPrv': 1, 'MnWw': 1, 'NoFence': 0}, 'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoBsmt': 0}, 'BsmtExposure': {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'NoBsmt': 0}, 'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoBsmt': 0}, 'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoGarage': 0}, 'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoGarage': 0}, 'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, 'Functional': {'Typ': 0, 'Min1': 1, 'Min2': 1, 'Mod': 2, 'Maj1': 3, 'Maj2': 4, 'Sev': 5, 'Sal': 6}})
newer_dwelling = all_data.MSSubClass.replace({20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
newer_dwelling.name = 'newer_dwelling'
all_data = all_data.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30', 40: 'SubClass_40', 45: 'SubClass_45', 50: 'SubClass_50', 60: 'SubClass_60', 70: 'SubClass_70', 75: 'SubClass_75', 80: 'SubClass_80', 85: 'SubClass_85', 90: 'SubClass_90', 120: 'SubClass_120', 150: 'SubClass_150', 160: 'SubClass_160', 180: 'SubClass_180', 190: 'SubClass_190'}})
overall_poor_qu = all_data.OverallQual.copy()
overall_poor_qu = 5 - overall_poor_qu
overall_poor_qu[overall_poor_qu < 0] = 0
overall_poor_qu.name = 'overall_poor_qu'
overall_good_qu = all_data.OverallQual.copy()
overall_good_qu = overall_good_qu - 5
overall_good_qu[overall_good_qu < 0] = 0
overall_good_qu.name = 'overall_good_qu'
overall_poor_cond = all_data.OverallCond.copy()
overall_poor_cond = 5 - overall_poor_cond
overall_poor_cond[overall_poor_cond < 0] = 0
overall_poor_cond.name = 'overall_poor_cond'
overall_good_cond = all_data.OverallCond.copy()
overall_good_cond = overall_good_cond - 5
overall_good_cond[overall_good_cond < 0] = 0
overall_good_cond.name = 'overall_good_cond'
exter_poor_qu = all_data.ExterQual.copy()
exter_poor_qu[exter_poor_qu < 3] = 1
exter_poor_qu[exter_poor_qu >= 3] = 0
exter_poor_qu.name = 'exter_poor_qu'
exter_good_qu = all_data.ExterQual.copy()
exter_good_qu[exter_good_qu <= 3] = 0
exter_good_qu[exter_good_qu > 3] = 1
exter_good_qu.name = 'exter_good_qu'
exter_poor_cond = all_data.ExterCond.copy()
exter_poor_cond[exter_poor_cond < 3] = 1
exter_poor_cond[exter_poor_cond >= 3] = 0
exter_poor_cond.name = 'exter_poor_cond'
exter_good_cond = all_data.ExterCond.copy()
exter_good_cond[exter_good_cond <= 3] = 0
exter_good_cond[exter_good_cond > 3] = 1
exter_good_cond.name = 'exter_good_cond'
bsmt_poor_cond = all_data.BsmtCond.copy()
bsmt_poor_cond[bsmt_poor_cond < 3] = 1
bsmt_poor_cond[bsmt_poor_cond >= 3] = 0
bsmt_poor_cond.name = 'bsmt_poor_cond'
bsmt_good_cond = all_data.BsmtCond.copy()
bsmt_good_cond[bsmt_good_cond <= 3] = 0
bsmt_good_cond[bsmt_good_cond > 3] = 1
bsmt_good_cond.name = 'bsmt_good_cond'
garage_poor_qu = all_data.GarageQual.copy()
garage_poor_qu[garage_poor_qu < 3] = 1
garage_poor_qu[garage_poor_qu >= 3] = 0
garage_poor_qu.name = 'garage_poor_qu'
garage_good_qu = all_data.GarageQual.copy()
garage_good_qu[garage_good_qu <= 3] = 0
garage_good_qu[garage_good_qu > 3] = 1
garage_good_qu.name = 'garage_good_qu'
garage_poor_cond = all_data.GarageCond.copy()
garage_poor_cond[garage_poor_cond < 3] = 1
garage_poor_cond[garage_poor_cond >= 3] = 0
garage_poor_cond.name = 'garage_poor_cond'
garage_good_cond = all_data.GarageCond.copy()
garage_good_cond[garage_good_cond <= 3] = 0
garage_good_cond[garage_good_cond > 3] = 1
garage_good_cond.name = 'garage_good_cond'
kitchen_poor_qu = all_data.KitchenQual.copy()
kitchen_poor_qu[kitchen_poor_qu < 3] = 1
kitchen_poor_qu[kitchen_poor_qu >= 3] = 0
kitchen_poor_qu.name = 'kitchen_poor_qu'
kitchen_good_qu = all_data.KitchenQual.copy()
kitchen_good_qu[kitchen_good_qu <= 3] = 0
kitchen_good_qu[kitchen_good_qu > 3] = 1
kitchen_good_qu.name = 'kitchen_good_qu'
qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu, exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu, garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)
bad_heating = all_data.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})
bad_heating.name = 'bad_heating'
MasVnrType_Any = all_data.MasVnrType.replace({'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0})
MasVnrType_Any.name = 'MasVnrType_Any'
SaleCondition_PriceDown = all_data.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})
SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'
Neighborhood_Good = pd.DataFrame(np.zeros((all_data.shape[0], 1)), columns=['Neighborhood_Good'])
Neighborhood_Good[all_data.Neighborhood == 'NridgHt'] = 1
Neighborhood_Good[all_data.Neighborhood == 'Crawfor'] = 1
Neighborhood_Good[all_data.Neighborhood == 'StoneBr'] = 1
Neighborhood_Good[all_data.Neighborhood == 'Somerst'] = 1
Neighborhood_Good[all_data.Neighborhood == 'NoRidge'] = 1
from sklearn.svm import SVC
svm = SVC(C=100)
pc = pd.Series(np.zeros(train.shape[0]))
pc[:] = 'pc1'
pc[train.SalePrice >= 150000] = 'pc2'
pc[train.SalePrice >= 220000] = 'pc3'
columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
X_t = pd.get_dummies(train.loc[:, columns_for_pc], sparse=True)