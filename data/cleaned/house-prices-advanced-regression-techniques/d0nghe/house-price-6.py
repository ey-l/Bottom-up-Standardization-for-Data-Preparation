import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sns.set_style('white')
sns.set_color_codes(palette='deep')
(f, ax) = plt.subplots(figsize=(8, 7))
sns.distplot(train['SalePrice'], color='b')
ax.xaxis.grid(False)
ax.set(ylabel='Frequency')
ax.set(xlabel='SalePrice')
ax.set(title='SalePrice distribution')
sns.despine(trim=True, left=True)

train
test.shape
data = pd.concat([train, test])
corr = data.corr()
colormap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(20, 20))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, fmt='.2f', linewidths=0.3, cmap=colormap, linecolor='white')

data.info()
data.index = range(2919)
data.drop(['Id', 'Utilities'], axis=1, inplace=True)
data['MoSold'] = data['MoSold'].astype(str)
data['YrSold'] = data['YrSold'].astype(str)
l2 = ['LotFrontage', 'MasVnrArea']
l3 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']
for item in l2:
    data[item] = data[item].fillna(data[item].mean())
for item in l3:
    data[item] = data[item].fillna(0)
data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
data['MiscFeature'] = data['MiscFeature'].fillna('None')
data['Alley'] = data['Alley'].fillna('None')
l1 = data['MSZoning'].unique().tolist()
for i in range(len(l1) - 1):
    j = np.random.randint(0, 4)
    data['MSZoning'] = data['MSZoning'].fillna(l1[j])
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna('None')
data['Exterior2nd'] = data['Exterior2nd'].fillna('None')
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['BsmtQual'] = data['BsmtQual'].fillna('None')
data['BsmtCond'] = data['BsmtCond'].fillna('None')
data['BsmtExposure'] = data['BsmtExposure'].fillna('None')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')
data['Fence'] = data['Fence'].fillna('None')
data['Functional'] = data['Functional'].fillna('Typical')
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')
data['GarageType'] = data['GarageType'].fillna('None')
data['GarageFinish'] = data['GarageFinish'].fillna('None')
data['GarageQual'] = data['GarageQual'].fillna('None')
data['GarageCond'] = data['GarageCond'].fillna('None')
data['PoolQC'] = data['PoolQC'].fillna('None')
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'HouseStyle', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'Neighborhood', 'SaleCondition', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'Condition1', 'Condition2', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'GarageType', 'SaleType', 'BldgType')
for c in cols:
    lb = LabelEncoder()