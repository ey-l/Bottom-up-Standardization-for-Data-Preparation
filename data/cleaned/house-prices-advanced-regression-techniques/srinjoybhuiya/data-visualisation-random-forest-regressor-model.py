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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(train_df.shape)
train_df.head()
print(train_df.columns)
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cmap='Greens')
na_features = [x for x in train_df.columns if train_df[x].isnull().sum() > 1]
for i in na_features:
    print(i, np.round(train_df[i].isnull().mean(), 4), ' % missing values')
for feature in na_features:
    data = train_df.copy()
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    plt.style.use('seaborn')
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)

print(f'id of houses {len(train_df.Id)}')
num_feature = [x for x in train_df.columns if train_df[x].dtypes != 'O']
print('Num of Numerical Variables', len(num_feature))
train_df[num_feature].head()
yr_features = [x for x in num_feature if 'Year' in x or 'Yr' in x]
yr_features
for i in yr_features:
    print(i, train_df[i].unique())
plt.plot(train_df.groupby('YrSold')['SalePrice'].median())
plt.xticks(ticks=[2006, 2007, 2008, 2009, 2010])
plt.title('Year Sold VS Selling price')

for i in yr_features:
    if i != 'YrSold':
        data = train_df.copy()
        data[i] = data['YrSold'] - data[i]
        plt.scatter(data[i], data['SalePrice'])
        plt.xlabel(i)
        plt.ylabel('Sales Price')

discrete_feature = [x for x in num_feature if len(train_df[x].unique()) < 25 and x not in yr_features and (x != 'Id')]
print('Number of Discrete Fetures', len(discrete_feature))
discrete_feature
for x in discrete_feature:
    data = train_df.copy()
    data.groupby(x)['SalePrice'].median().plot.bar()
    plt.xlabel(x)
    plt.ylabel('Sales Price')

cont_features = [x for x in num_feature if x not in discrete_feature and x not in yr_features and (x not in 'Id')]
len(cont_features)
cont_features
for i in cont_features:
    data = train_df.copy()
    data[i].hist(bins=25)
    plt.xlabel(i)
    plt.ylabel('Sales Price')
    plt.title(i)

for feature in cont_features:
    data = train_df.copy()
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
    data = train_df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)

cat_features = [x for x in train_df.columns if train_df[x].dtypes == 'O']
len(cat_features)
cat_features
train_df[cat_features].head()
for x in cat_features:
    print(f'The feature is {x} and the number of categories are {len(train_df[x].unique())}')
for x in cat_features:
    data = train_df.copy()
    data.groupby(x)['SalePrice'].median().plot.bar()
    plt.xlabel(x)
    plt.ylabel('Sale Price')
    plt.title(x)

train_df.head()
na_features_cat = [x for x in train_df.columns if train_df[x].isnull().sum() > 1 and train_df[x].dtypes == 'O']
for i in na_features:
    print(i, '=', np.round(train_df[i].isnull().mean(), 4), '% missing values')

def replace_na_cat_feature(df, features_nan):
    data = df.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data
train_df = replace_na_cat_feature(train_df, na_features_cat)
train_df[na_features].isnull().sum()
na_features_num = [x for x in train_df.columns if train_df[x].isnull().sum() > 1 and train_df[x].dtypes != 'O']
for i in na_features_num:
    print(i, '=', np.round(train_df[i].isnull().mean(), 4), '% missing values')
for x in na_features_num:
    median_value = train_df[x].median()
    train_df[x + 'nan'] = np.where(train_df[x].isnull(), 1, 0)
    train_df[x].fillna(median_value, inplace=True)
train_df[na_features_num].isnull().sum()
train_df.YearRemodAdd
for x in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    train_df[x] = train_df['YrSold'] - train_df[x]
train_df[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
import numpy as np
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for x in num_features:
    train_df[x] = np.log(train_df[x])
train_df.head()
categorical_features = [x for x in train_df.columns if train_df[x].dtype == 'O']
categorical_features
for x in categorical_features:
    temp = train_df.groupby(x)['SalePrice'].count() / len(train_df)
    temp_df = temp[temp > 0.01].index
    train_df[x] = np.where(train_df[x].isin(temp_df), train_df[x], 'Rare_var')
train_df.head()
for feature in categorical_features:
    labels_ordered = train_df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    train_df[feature] = train_df[feature].map(labels_ordered)
train_df.head(10)
features_scale = [x for x in train_df.columns if x not in ['Id', 'SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()