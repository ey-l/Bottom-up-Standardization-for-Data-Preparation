import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', 100)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.shape)
_input1.head()
print(_input1.columns)
plt.figure(figsize=(10, 6))
sns.heatmap(_input1.isnull(), cmap='Greens')
na_features = [x for x in _input1.columns if _input1[x].isnull().sum() > 1]
for i in na_features:
    print(i, np.round(_input1[i].isnull().mean(), 4), ' % missing values')
for feature in na_features:
    data = _input1.copy()
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    plt.style.use('seaborn')
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
print(f'id of houses {len(_input1.Id)}')
num_feature = [x for x in _input1.columns if _input1[x].dtypes != 'O']
print('Num of Numerical Variables', len(num_feature))
_input1[num_feature].head()
yr_features = [x for x in num_feature if 'Year' in x or 'Yr' in x]
yr_features
for i in yr_features:
    print(i, _input1[i].unique())
plt.plot(_input1.groupby('YrSold')['SalePrice'].median())
plt.xticks(ticks=[2006, 2007, 2008, 2009, 2010])
plt.title('Year Sold VS Selling price')
for i in yr_features:
    if i != 'YrSold':
        data = _input1.copy()
        data[i] = data['YrSold'] - data[i]
        plt.scatter(data[i], data['SalePrice'])
        plt.xlabel(i)
        plt.ylabel('Sales Price')
discrete_feature = [x for x in num_feature if len(_input1[x].unique()) < 25 and x not in yr_features and (x != 'Id')]
print('Number of Discrete Fetures', len(discrete_feature))
discrete_feature
for x in discrete_feature:
    data = _input1.copy()
    data.groupby(x)['SalePrice'].median().plot.bar()
    plt.xlabel(x)
    plt.ylabel('Sales Price')
cont_features = [x for x in num_feature if x not in discrete_feature and x not in yr_features and (x not in 'Id')]
len(cont_features)
cont_features
for i in cont_features:
    data = _input1.copy()
    data[i].hist(bins=25)
    plt.xlabel(i)
    plt.ylabel('Sales Price')
    plt.title(i)
for feature in cont_features:
    data = _input1.copy()
    if 0 in data[feature].unique():
        pass
    if True:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Sales Price')
        plt.title(feature)
for feature in cont_features:
    data = _input1.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
cat_features = [x for x in _input1.columns if _input1[x].dtypes == 'O']
len(cat_features)
cat_features
_input1[cat_features].head()
for x in cat_features:
    print(f'The feature is {x} and the number of categories are {len(_input1[x].unique())}')
for x in cat_features:
    data = _input1.copy()
    data.groupby(x)['SalePrice'].median().plot.bar()
    plt.xlabel(x)
    plt.ylabel('Sale Price')
    plt.title(x)
_input1.head()
na_features_cat = [x for x in _input1.columns if _input1[x].isnull().sum() > 1 and _input1[x].dtypes == 'O']
for i in na_features:
    print(i, '=', np.round(_input1[i].isnull().mean(), 4), '% missing values')

def replace_na_cat_feature(df, features_nan):
    data = df.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data
_input1 = replace_na_cat_feature(_input1, na_features_cat)
_input1[na_features].isnull().sum()
na_features_num = [x for x in _input1.columns if _input1[x].isnull().sum() > 1 and _input1[x].dtypes != 'O']
for i in na_features_num:
    print(i, '=', np.round(_input1[i].isnull().mean(), 4), '% missing values')
for x in na_features_num:
    median_value = _input1[x].median()
    _input1[x + 'nan'] = np.where(_input1[x].isnull(), 1, 0)
    _input1[x] = _input1[x].fillna(median_value, inplace=False)
_input1[na_features_num].isnull().sum()
_input1.YearRemodAdd
for x in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    _input1[x] = _input1['YrSold'] - _input1[x]
_input1[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
import numpy as np
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for x in num_features:
    _input1[x] = np.log(_input1[x])
_input1.head()
categorical_features = [x for x in _input1.columns if _input1[x].dtype == 'O']
categorical_features
for x in categorical_features:
    temp = _input1.groupby(x)['SalePrice'].count() / len(_input1)
    temp_df = temp[temp > 0.01].index
    _input1[x] = np.where(_input1[x].isin(temp_df), _input1[x], 'Rare_var')
_input1.head()
for feature in categorical_features:
    labels_ordered = _input1.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    _input1[feature] = _input1[feature].map(labels_ordered)
_input1.head(10)
features_scale = [x for x in _input1.columns if x not in ['Id', 'SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()