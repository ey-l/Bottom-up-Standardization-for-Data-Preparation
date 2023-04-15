import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
data = pd.concat((df_train, df_test)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)
df_train.head()
(df_train.shape, df_test.shape)
df_train.describe()
df_train.keys()
df_train.dtypes
df_train['SalePrice'].hist(bins=40)
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())
corrmat = df_train.corr()
(f, ax) = plt.subplots(figsize=(30, 19))
sns.set(font_scale=1.45)
sns.heatmap(corrmat, square=True, cmap='coolwarm')
correlations = corrmat['SalePrice'].sort_values(ascending=False)
features = correlations.index[0:10]
features
sns.pairplot(df_train[features], size=2.5)

df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)
data.drop(['Id'], axis=1, inplace=True)
training_null = pd.isnull(df_train).sum()
testing_null = pd.isnull(df_test).sum()
null = pd.concat([training_null, testing_null], axis=1, keys=['Training', 'Testing'])
null
null_with_meaning = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for i in null_with_meaning:
    df_train[i].fillna('None', inplace=True)
    df_test[i].fillna('None', inplace=True)
    data[i].fillna('None', inplace=True)
null_many = null[null.sum(axis=1) > 200]
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]
null_many
df_train.drop('LotFrontage', axis=1, inplace=True)
df_test.drop('LotFrontage', axis=1, inplace=True)
data.drop('LotFrontage', axis=1, inplace=True)
null_few
df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].mean(), inplace=True)
df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].mean(), inplace=True)
data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean(), inplace=True)
df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mean(), inplace=True)
df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean(), inplace=True)
data['MasVnrArea'].fillna(data['MasVnrArea'].mean(), inplace=True)
df_train['MasVnrType'].fillna('None', inplace=True)
df_test['MasVnrType'].fillna('None', inplace=True)
data['MasVnrType'].fillna('None', inplace=True)
types_train = df_train.dtypes
num_train = types_train[(types_train == int) | (types_train == float)]
cat_train = types_train[types_train == object]
types_test = df_test.dtypes
num_test = types_test[(types_test == int) | (types_test == float)]
cat_test = types_test[types_test == object]
numerical_values_train = list(num_train.index)
numerical_values_test = list(num_test.index)
fill_num = numerical_values_train + numerical_values_test
print(fill_num)
for i in fill_num:
    df_train[i].fillna(df_train[i].mean(), inplace=True)
fill_num.remove('SalePrice')
print(fill_num)
for i in fill_num:
    df_test[i].fillna(df_test[i].mean(), inplace=True)
    data[i].fillna(data[i].mean(), inplace=True)
(df_train.shape, df_test.shape)
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
    df_train[i].fillna(most_common[k], inplace=True)
    df_test[i].fillna(most_common[k], inplace=True)
    data[i].fillna(most_common[k], inplace=True)
    k += 1
training_null = pd.isnull(df_train).sum()
testing_null = pd.isnull(df_test).sum()
null = pd.concat([training_null, testing_null], axis=1, keys=['Training', 'Testing'])
null[null.sum(axis=1) > 0]
np.log(df_train['SalePrice']).hist(bins=40)
df_train['LogPrice'] = np.log(df_train['SalePrice'])
df_train.head()
df_train_add = df_train.copy()
df_train_add['TotalSF'] = df_train_add['TotalBsmtSF'] + df_train_add['1stFlrSF'] + df_train_add['2ndFlrSF']
df_train_add['Total_Bathrooms'] = df_train_add['FullBath'] + 0.5 * df_train_add['HalfBath'] + df_train_add['BsmtFullBath'] + 0.5 * df_train_add['BsmtHalfBath']
df_train_add['Total_porch_sf'] = df_train_add['OpenPorchSF'] + df_train_add['3SsnPorch'] + df_train_add['EnclosedPorch'] + df_train_add['ScreenPorch'] + df_train_add['WoodDeckSF']
df_test_add = df_test.copy()
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
df_train[df_train['SalePrice'] > 600000]
categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)
print(categorical_values_train)
df_train_add = df_train.copy()
df_test_add = df_test.copy()
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
for i in categorical_values_train:
    df_train_add[i] = lb_make.fit_transform(df_train[i])
for i in categorical_values_test:
    df_test_add[i] = lb_make.fit_transform(df_test[i])
for i in categorical_values_train:
    feature_set = set(df_train[i])
    for j in feature_set:
        feature_list = list(feature_set)
        df_train.loc[df_train[i] == j, i] = feature_list.index(j)
        df_train_add.loc[df_train[i] == j, i] = feature_list.index(j)
for i in categorical_values_test:
    feature_set2 = set(df_test[i])
    for j in feature_set2:
        feature_list2 = list(feature_set2)
        df_test.loc[df_test[i] == j, i] = feature_list2.index(j)
        df_test_add.loc[df_test[i] == j, i] = feature_list2.index(j)
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