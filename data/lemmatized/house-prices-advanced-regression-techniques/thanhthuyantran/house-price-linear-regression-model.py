import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data = pd.concat((_input1, _input0)).reset_index(drop=True)
data = data.drop(['SalePrice'], axis=1, inplace=False)
print('Train set size:', _input1.shape)
_input1.head()
_input1.columns
_input1.count()
_input1 = pd.DataFrame(_input1)
print(_input1)
_input1.info()
_input1.describe()
print('Test set size:', _input0.shape)
_input0.count()
_input0 = pd.DataFrame(_input0)
print(_input0)
_input0.head()
_input0.info()
_input0.describe()
print(_input1.isnull().any())
print(_input1.dtypes)
pd.isnull(_input1).sum()
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=True, cmap='plasma')
sns.set_style('whitegrid')
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar()
_input1 = _input1.drop(columns=['FireplaceQu', 'Alley', 'MiscFeature', 'PoolQC', 'Fence', 'LotFrontage'], inplace=False)
sns.set_style('whitegrid')
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar()
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='plasma')
print(_input0.isnull().any())
print(_input0.dtypes)
pd.isnull(_input0).sum()
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=True, cmap='plasma')
sns.set_style('whitegrid')
missing = _input0.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar()
sns.set_style('whitegrid')
missing = _input0.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar()
sns.heatmap(_input0.isnull(), yticklabels=False, cbar=True, cmap='plasma')
training_null = pd.isnull(_input1).sum()
testing_null = pd.isnull(_input0).sum()
null = pd.concat([training_null, testing_null], axis=1, keys=['Training', 'Testing'])
null
meaning_null = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for i in meaning_null:
    _input1[i] = _input1[i].fillna('None', inplace=False)
    _input0[i] = _input0[i].fillna('None', inplace=False)
    data[i] = data[i].fillna('None', inplace=False)
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mean(), inplace=False)
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].mean(), inplace=False)
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean(), inplace=False)
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mean(), inplace=False)
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mean(), inplace=False)
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('None', inplace=False)
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('None', inplace=False)
data['MasVnrType'] = data['MasVnrType'].fillna('None', inplace=False)
sns.set_style('whitegrid')
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar()
sns.set_style('whitegrid')
missing = _input0.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar()
train_null_2 = pd.isnull(_input1).sum()
test_null_2 = pd.isnull(_input0).sum()
null_2 = pd.concat([train_null_2, test_null_2], axis=1, keys=['Training', 'Testing'])
null_2
null_values_2 = null_2[null_2.sum(axis=1) > 0]
null_values_2
_input1 = _input1.dropna()
print(_input1.isnull().sum())
types_test = _input0.dtypes
num_test = types_test[(types_test == int) | (types_test == float)]
cat_test = types_test[types_test == object]
numerical_values_test = list(num_test.index)
print(numerical_values_test)
for i in numerical_values_test:
    _input0[i] = _input0[i].fillna(_input0[i].mean(), inplace=False)
categorical_values_test = list(cat_test.index)
fill_cat = []
for i in categorical_values_test:
    if i in list(null.index):
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
    _input0[i] = _input0[i].fillna(most_common[k], inplace=False)
    k += 1
training_null_3 = pd.isnull(_input1).sum()
testing_null_3 = pd.isnull(_input0).sum()
null_3 = pd.concat([training_null_3, testing_null_3], axis=1, keys=['Training', 'Testing'])
null_3[null_3.sum(axis=1) > 0]
_input1['MSSubClass'] = _input1['MSSubClass'].apply(str)
_input1['YrSold'] = _input1['YrSold'].astype(str)
_input1['MoSold'] = _input1['MoSold'].astype(str)
_input0['MSSubClass'] = _input0['MSSubClass'].apply(str)
_input0['YrSold'] = _input0['YrSold'].astype(str)
_input0['MoSold'] = _input0['MoSold'].astype(str)
print(_input1.shape)
print(_input0.shape)
y_train = _input1['SalePrice'].copy()
x_train = _input1.copy().drop(columns=['Id', 'SalePrice'])
x_test = _input0.copy().drop(columns=['Id'])
print(x_train.shape)
print(x_test.shape)
num_cols = x_train.select_dtypes(include=['number'])
cat_cols = x_train.select_dtypes(include=['object'])
print(f'The dataset contains {len(num_cols.columns.tolist())} numerical columns and {len(cat_cols.columns.tolist())} categorical columns')
df = pd.concat([_input1, _input0])
categorical_cols = df.select_dtypes(include=np.object).columns
df = pd.get_dummies(df, prefix=categorical_cols)
df
df = df.drop(columns=['Id'])
print(df.shape)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
categorical_cols = _input1.select_dtypes(include=np.object).columns
_input1 = pd.get_dummies(_input1, prefix=categorical_cols)
_input1
print(_input1.shape)
y = _input1['SalePrice']
x = _input1.drop('SalePrice', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=0)
lr = LinearRegression()