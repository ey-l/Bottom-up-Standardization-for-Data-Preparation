import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor, Ridge, Lasso
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Numlist1 = ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond']
Numlist2 = ['BsmtExposure']
Numlist3 = ['BsmtFinType1', 'BsmtFinType2']
Numlist4 = ['PoolQC']
Numlist5 = ['Fence']
Numlist6 = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']
Numlist7 = ['LotShape']
Numlist8 = ['LandSlope']
Numlist9 = ['Functional']
Numlist10 = ['GarageFinish']

def numeric_map1(x):
    return x.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, np.nan: 0})

def numeric_map2(y):
    return y.map({'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4, np.nan: 0})

def numeric_map3(z):
    return z.map({'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6, np.nan: 0})

def numeric_map4(a):
    return a.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, np.nan: 0})

def numeric_map5(b):
    return b.map({'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4, np.nan: 0})

def numeric_map6(c):
    return c.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

def numeric_map7(d):
    return d.map({'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4})

def numeric_map8(e):
    return e.map({'Sev': 1, 'Mod': 2, 'Gtl': 3})

def numeric_map9(f):
    return f.map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})

def numeric_map10(g):
    return g.map({'Unf': 1, 'RFn': 2, 'Fin': 3, np.nan: 0})
_input1[Numlist1] = _input1[Numlist1].apply(numeric_map1)
_input1[Numlist2] = _input1[Numlist2].apply(numeric_map2)
_input1[Numlist3] = _input1[Numlist3].apply(numeric_map3)
_input1[Numlist4] = _input1[Numlist4].apply(numeric_map4)
_input1[Numlist5] = _input1[Numlist5].apply(numeric_map5)
_input1[Numlist6] = _input1[Numlist6].apply(numeric_map6)
_input1[Numlist7] = _input1[Numlist7].apply(numeric_map7)
_input1[Numlist8] = _input1[Numlist8].apply(numeric_map8)
_input1[Numlist9] = _input1[Numlist9].apply(numeric_map9)
_input1[Numlist10] = _input1[Numlist10].apply(numeric_map10)
_input0[Numlist1] = _input0[Numlist1].apply(numeric_map1)
_input0[Numlist2] = _input0[Numlist2].apply(numeric_map2)
_input0[Numlist3] = _input0[Numlist3].apply(numeric_map3)
_input0[Numlist4] = _input0[Numlist4].apply(numeric_map4)
_input0[Numlist5] = _input0[Numlist5].apply(numeric_map5)
_input0[Numlist6] = _input0[Numlist6].apply(numeric_map6)
_input0[Numlist7] = _input0[Numlist7].apply(numeric_map7)
_input0[Numlist8] = _input0[Numlist8].apply(numeric_map8)
_input0[Numlist9] = _input0[Numlist9].apply(numeric_map9)
_input0[Numlist10] = _input0[Numlist10].apply(numeric_map10)
train_num = _input1.select_dtypes(exclude=['object'])
train_cat = _input1.select_dtypes('object')
test_num = _input0.select_dtypes(exclude=['object'])
test_cat = _input0.select_dtypes('object')
for i in train_num.columns:
    plt.figure(figsize=(15, 8))
    sns.distplot(train_num[i])
train_num = train_num.drop('MSSubClass', inplace=False, axis=1)
test_num = test_num.drop('MSSubClass', inplace=False, axis=1)
train_num = train_num.drop('YrSold', inplace=False, axis=1)
test_num = test_num.drop('YrSold', inplace=False, axis=1)
train_num = train_num.drop('Id', inplace=False, axis=1)
test_num = test_num.drop('Id', inplace=False, axis=1)
skewed_list = []
for i in train_num.columns:
    if abs(train_num[i].skew()) > 0.5:
        skewed_list.append(i)
skewed_list
skewed_list = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']
corr = train_num.corr()
plt.figure(figsize=(25, 25))
sns.heatmap(corr, annot=True)
train_num = train_num.drop(columns=['PoolArea', 'GarageCond', 'GarageFinish', 'Functional', 'GarageYrBlt', 'TotRmsAbvGrd', 'BsmtFinType2', '1stFlrSF'], inplace=False, axis=1)
test_num = test_num.drop(columns=['PoolArea', 'GarageCond', 'GarageFinish', 'Functional', 'GarageYrBlt', 'TotRmsAbvGrd', 'BsmtFinType2', '1stFlrSF'], inplace=False, axis=1)
corr2 = train_num.corr()
corr2['SalePrice'].sort_values(ascending=False).abs()
fe_list = ['BsmtUnfSF', 'BsmtCond', 'BedroomAbvGr', 'PoolQC', 'ScreenPorch', 'MoSold', '3SsnPorch', 'ExterCond', 'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal ', 'LowQualFinSF', 'LandSlope', 'OverallCond ', 'EnclosedPorch', 'KitchenAbvGr', 'Fence']
train_num.isnull().sum()
train_num = train_num.fillna(train_num.median())
test_num = test_num.fillna(test_num.median())
train_num[skewed_list] = train_num[skewed_list].apply(lambda i: np.log1p(i))
test_num[skewed_list] = test_num[skewed_list].apply(lambda i: np.log1p(i))
y = train_num['SalePrice']
train_num = train_num.iloc[:, :-1]
train_num.info()
train_cat.isnull().sum()

def dependency(data, feature):
    table = pd.crosstab(data['SalePrice'], data[feature])
    (stat, p, dof, excpected) = chi2_contingency(table)
    print(dof)
    significance_level = 0.05
    print('p value: ' + str(p))
    if p <= significance_level:
        print(f'{feature}: REJECT NULL HYPOTHESIS, THE VARIABLES ARE DEPENDENT')
        return feature
    else:
        print(f'{feature}: ACCEPT NULL HYPOTHESIS, THE VARIABLES ARE INDEPENDENT')
for i in train_cat.columns:
    dependency(_input1, i)
dependent = ['MSZoning', 'Street', 'LotConfig', 'Neighborhood', 'MasVnrType', 'Foundation', 'CentralAir', 'SaleType', 'SaleCondition']
train_cat = pd.get_dummies(train_cat[dependent], drop_first=True)
test_cat = pd.get_dummies(test_cat[dependent], drop_first=True)
train_num.shape
fe_list
train_num_2 = train_num.copy()
test_num_2 = test_num.copy()
train_num_2['BsmtUnfSF'] = train_num_2['BsmtUnfSF'] ** 2
test_num_2['BsmtUnfSF'] = test_num_2['BsmtUnfSF'] ** 2
train_num_2['BsmtCond'] = train_num_2['BsmtCond'] ** 2
test_num_2['BsmtCond'] = test_num_2['BsmtCond'] ** 2
train_num_2['BedroomAbvGr'] = train_num_2['BedroomAbvGr'] ** 2
test_num_2['BedroomAbvGr'] = test_num_2['BedroomAbvGr'] ** 2
train_num_2['PoolQC'] = train_num_2['PoolQC'] ** 2
test_num_2['PoolQC'] = test_num_2['PoolQC'] ** 2
train_num_2['ScreenPorch'] = train_num_2['ScreenPorch'] ** 2
test_num_2['ScreenPorch'] = test_num_2['ScreenPorch'] ** 2
train_num_2['MoSold'] = train_num_2['MoSold'] ** 2
test_num_2['MoSold'] = test_num_2['MoSold'] ** 2
train_num_2['3SsnPorch'] = train_num_2['3SsnPorch'] ** 2
test_num_2['3SsnPorch'] = test_num_2['3SsnPorch'] ** 2
train_num_2['ExterCond'] = train_num_2['ExterCond'] ** 2
test_num_2['ExterCond'] = test_num_2['ExterCond'] ** 2
train_num_2['BsmtFinSF2'] = train_num_2['BsmtFinSF2'] ** 2
test_num_2['BsmtFinSF2'] = test_num_2['BsmtFinSF2'] ** 2
train_num_2['BsmtHalfBath'] = train_num_2['BsmtHalfBath'] ** 2
test_num_2['BsmtHalfBath'] = test_num_2['BsmtHalfBath'] ** 2
train_num_2['MiscVal'] = train_num_2['MiscVal'] ** 2
test_num_2['MiscVal'] = test_num_2['MiscVal'] ** 2
train_num_2['LowQualFinSF'] = train_num_2['LowQualFinSF'] ** 2
test_num_2['LowQualFinSF'] = test_num_2['LowQualFinSF'] ** 2
frames = [train_num, train_cat]
frames2 = [test_num, test_cat]
train = pd.concat(frames, axis=1)
test = pd.concat(frames2, axis=1)
frames_2 = [train_num_2, train_cat]
frames2_2 = [test_num_2, test_cat]
train_2 = pd.concat(frames_2, axis=1)
test_2 = pd.concat(frames2_2, axis=1)
train_cat.info()
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.fit_transform(test)
train_2 = sc.fit_transform(train_2)
test_2 = sc.fit_transform(test_2)
sns.distplot(y)
y = np.log1p(y)
sns.distplot(y)
lr = LinearRegression()
cv_lr = cross_val_score(lr, train, y, cv=10, scoring='neg_root_mean_squared_error')
cv_lr.mean()