import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data = pd.concat((train, test)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)
print('Train set size:', train.shape)
train.head()
train.columns
train.count()
train = pd.DataFrame(train)
print(train)
train.info()
train.describe()
print('Test set size:', test.shape)
test.count()
test = pd.DataFrame(test)
print(test)
test.head()
test.info()
test.describe()
print(train.isnull().any())
print(train.dtypes)
pd.isnull(train).sum()
sns.heatmap(train.isnull(), yticklabels=False, cbar=True, cmap='plasma')
sns.set_style('whitegrid')
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
train.drop(columns=['FireplaceQu', 'Alley', 'MiscFeature', 'PoolQC', 'Fence', 'LotFrontage'], inplace=True)
sns.set_style('whitegrid')
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='plasma')
print(test.isnull().any())
print(test.dtypes)
pd.isnull(test).sum()
sns.heatmap(test.isnull(), yticklabels=False, cbar=True, cmap='plasma')
sns.set_style('whitegrid')
missing = test.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
sns.set_style('whitegrid')
missing = test.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
sns.heatmap(test.isnull(), yticklabels=False, cbar=True, cmap='plasma')
training_null = pd.isnull(train).sum()
testing_null = pd.isnull(test).sum()
null = pd.concat([training_null, testing_null], axis=1, keys=['Training', 'Testing'])
null
meaning_null = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for i in meaning_null:
    train[i].fillna('None', inplace=True)
    test[i].fillna('None', inplace=True)
    data[i].fillna('None', inplace=True)
train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean(), inplace=True)
test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean(), inplace=True)
data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean(), inplace=True)
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(), inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mean(), inplace=True)
data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace=True)
train['MasVnrType'].fillna('None', inplace=True)
test['MasVnrType'].fillna('None', inplace=True)
data['MasVnrType'].fillna('None', inplace=True)
sns.set_style('whitegrid')
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
sns.set_style('whitegrid')
missing = test.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
train_null_2 = pd.isnull(train).sum()
test_null_2 = pd.isnull(test).sum()
null_2 = pd.concat([train_null_2, test_null_2], axis=1, keys=['Training', 'Testing'])
null_2
null_values_2 = null_2[null_2.sum(axis=1) > 0]
null_values_2
train = train.dropna()
print(train.isnull().sum())
types_test = test.dtypes
num_test = types_test[(types_test == int) | (types_test == float)]
cat_test = types_test[types_test == object]
numerical_values_test = list(num_test.index)
print(numerical_values_test)
for i in numerical_values_test:
    test[i].fillna(test[i].mean(), inplace=True)
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
    test[i].fillna(most_common[k], inplace=True)
    k += 1
training_null_3 = pd.isnull(train).sum()
testing_null_3 = pd.isnull(test).sum()
null_3 = pd.concat([training_null_3, testing_null_3], axis=1, keys=['Training', 'Testing'])
null_3[null_3.sum(axis=1) > 0]
train['MSSubClass'] = train['MSSubClass'].apply(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)
print(train.shape)
print(test.shape)
y_train = train['SalePrice'].copy()
x_train = train.copy().drop(columns=['Id', 'SalePrice'])
x_test = test.copy().drop(columns=['Id'])
print(x_train.shape)
print(x_test.shape)
num_cols = x_train.select_dtypes(include=['number'])
cat_cols = x_train.select_dtypes(include=['object'])
print(f'The dataset contains {len(num_cols.columns.tolist())} numerical columns and {len(cat_cols.columns.tolist())} categorical columns')
df = pd.concat([train, test])
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
categorical_cols = train.select_dtypes(include=np.object).columns
train = pd.get_dummies(train, prefix=categorical_cols)
train
print(train.shape)
y = train['SalePrice']
x = train.drop('SalePrice', axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=0)
lr = LinearRegression()