import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data = pd.concat((_input1, _input0)).reset_index(drop=True)
data = data.drop(['SalePrice'], axis=1, inplace=False)
_input1.head()
(_input1.shape, _input0.shape)
_input1.describe()
_input1.keys()
_input1.dtypes
_input1['SalePrice'].hist(bins=40)
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(30, 19))
sns.set(font_scale=1.45)
sns.heatmap(corrmat, square=True, cmap='coolwarm')
correlations = corrmat['SalePrice'].sort_values(ascending=False)
features = correlations.index[0:10]
features
sns.pairplot(_input1[features], size=2.5)
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
data = data.drop(['Id'], axis=1, inplace=False)
training_null = pd.isnull(_input1).sum()
testing_null = pd.isnull(_input0).sum()
null = pd.concat([training_null, testing_null], axis=1, keys=['Training', 'Testing'])
null
null_with_meaning = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for i in null_with_meaning:
    _input1[i] = _input1[i].fillna('None', inplace=False)
    _input0[i] = _input0[i].fillna('None', inplace=False)
    data[i] = data[i].fillna('None', inplace=False)
null_many = null[null.sum(axis=1) > 200]
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]
null_many
_input1 = _input1.drop('LotFrontage', axis=1, inplace=False)
_input0 = _input0.drop('LotFrontage', axis=1, inplace=False)
data = data.drop('LotFrontage', axis=1, inplace=False)
null_few
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mean(), inplace=False)
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].mean(), inplace=False)
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean(), inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mean(), inplace=False)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mean(), inplace=False)
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None', inplace=False)
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('None', inplace=False)
data['MasVnrType'] = data['MasVnrType'].fillna('None', inplace=False)
types_train = _input1.dtypes
num_train = types_train[(types_train == int) | (types_train == float)]
cat_train = types_train[types_train == object]
types_test = _input0.dtypes
num_test = types_test[(types_test == int) | (types_test == float)]
cat_test = types_test[types_test == object]
numerical_values_train = list(num_train.index)
numerical_values_test = list(num_test.index)
fill_num = numerical_values_train + numerical_values_test
print(fill_num)
for i in fill_num:
    _input1[i] = _input1[i].fillna(_input1[i].mean(), inplace=False)
fill_num.remove('SalePrice')
print(fill_num)
for i in fill_num:
    _input0[i] = _input0[i].fillna(_input0[i].mean(), inplace=False)
    data[i] = data[i].fillna(data[i].mean(), inplace=False)
(_input1.shape, _input0.shape)
categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)
fill_cat = []
for i in categorical_values_train:
    if i in list(null_few.index):
        fill_cat.append(i)
print(fill_cat)

def most_common_term(lst):
    lst = list(lst)
    return max(set(lst), key=lst.count)
most_common = []
for i in fill_cat:
    most_common.append(most_common_term(data[i]))
most_common
most_common_dictionary = {fill_cat[0]: [most_common[0]], fill_cat[1]: [most_common[1]], fill_cat[2]: [most_common[2]], fill_cat[3]: [most_common[3]], fill_cat[4]: [most_common[4]], fill_cat[5]: [most_common[5]], fill_cat[6]: [most_common[6]], fill_cat[7]: [most_common[7]], fill_cat[8]: [most_common[8]]}
most_common_dictionary
k = 0
for i in fill_cat:
    _input1[i] = _input1[i].fillna(most_common[k], inplace=False)
    _input0[i] = _input0[i].fillna(most_common[k], inplace=False)
    data[i] = data[i].fillna(most_common[k], inplace=False)
    k += 1
training_null = pd.isnull(_input1).sum()
testing_null = pd.isnull(_input0).sum()
null = pd.concat([training_null, testing_null], axis=1, keys=['Training', 'Testing'])
null[null.sum(axis=1) > 0]
np.log(_input1['SalePrice']).hist(bins=40)
_input1['LogPrice'] = np.log(_input1['SalePrice'])
_input1.head()
df_train_add = _input1.copy()
df_train_add['TotalSF'] = df_train_add['TotalBsmtSF'] + df_train_add['1stFlrSF'] + df_train_add['2ndFlrSF']
df_train_add['Total_Bathrooms'] = df_train_add['FullBath'] + 0.5 * df_train_add['HalfBath'] + df_train_add['BsmtFullBath'] + 0.5 * df_train_add['BsmtHalfBath']
df_train_add['Total_porch_sf'] = df_train_add['OpenPorchSF'] + df_train_add['3SsnPorch'] + df_train_add['EnclosedPorch'] + df_train_add['ScreenPorch'] + df_train_add['WoodDeckSF']
df_test_add = _input0.copy()
df_test_add['TotalSF'] = df_test_add['TotalBsmtSF'] + df_test_add['1stFlrSF'] + df_test_add['2ndFlrSF']
df_test_add['Total_Bathrooms'] = df_test_add['FullBath'] + 0.5 * df_test_add['HalfBath'] + df_test_add['BsmtFullBath'] + 0.5 * df_test_add['BsmtHalfBath']
df_test_add['Total_porch_sf'] = df_test_add['OpenPorchSF'] + df_test_add['3SsnPorch'] + df_test_add['EnclosedPorch'] + df_test_add['ScreenPorch'] + df_test_add['WoodDeckSF']
df_train_add['haspool'] = df_train_add['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['has2ndfloor'] = df_train_add['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['hasgarage'] = df_train_add['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['hasbsmt'] = df_train_add['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_train_add['hasfireplace'] = df_train_add['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['haspool'] = df_test_add['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['has2ndfloor'] = df_test_add['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['hasgarage'] = df_test_add['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['hasbsmt'] = df_test_add['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_test_add['hasfireplace'] = df_test_add['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
_input1[_input1['SalePrice'] > 600000]
categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)
print(categorical_values_train)
df_train_add = _input1.copy()
df_test_add = _input0.copy()
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
for i in categorical_values_train:
    df_train_add[i] = lb_make.fit_transform(_input1[i])
for i in categorical_values_test:
    df_test_add[i] = lb_make.fit_transform(_input0[i])
for i in categorical_values_train:
    feature_set = set(_input1[i])
    for j in feature_set:
        feature_list = list(feature_set)
        _input1.loc[_input1[i] == j, i] = feature_list.index(j)
        df_train_add.loc[_input1[i] == j, i] = feature_list.index(j)
for i in categorical_values_test:
    feature_set2 = set(_input0[i])
    for j in feature_set2:
        feature_list2 = list(feature_set2)
        _input0.loc[_input0[i] == j, i] = feature_list2.index(j)
        df_test_add.loc[_input0[i] == j, i] = feature_list2.index(j)
df_train_add.head()
df_test_add.head()
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
X_train = df_train_add.drop(['SalePrice', 'LogPrice'], axis=1)
y_train = df_train_add['LogPrice']
from sklearn.model_selection import train_test_split
(X_training, X_valid, y_training, y_valid) = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()