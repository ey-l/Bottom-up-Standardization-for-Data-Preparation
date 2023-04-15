import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
pd.pandas.set_option('display.max_columns', None)
dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(dataset.shape)
dataset.head()
features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum() > 1]
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4), ' % missing values')
for feature in features_with_na:
    data = dataset.copy()
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)

print('Id of Houses {}'.format(len(dataset.Id)))
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
dataset[numerical_features].head()
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
year_feature
for feature in year_feature:
    print(feature, dataset[feature].unique())
dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs YearSold')
year_feature
for feature in year_feature:
    if feature != 'YrSold':
        data = dataset.copy()
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')

discrete_feature = [feature for feature in numerical_features if len(dataset[feature].unique()) < 25 and feature not in year_feature + ['Id']]
print('Discrete Variables Count: {}'.format(len(discrete_feature)))
discrete_feature
dataset[discrete_feature].head()
for feature in discrete_feature:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)

continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature + year_feature + ['Id']]
print('Continuous feature Count {}'.format(len(continuous_feature)))
for feature in continuous_feature:
    data = dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)

for feature in continuous_feature:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)

for feature in continuous_feature:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)

categorical_features = [feature for feature in dataset.columns if data[feature].dtypes == 'O']
categorical_features
dataset[categorical_features].head()
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature, len(dataset[feature].unique())))
for feature in categorical_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(dataset, dataset['SalePrice'], test_size=0.1, random_state=0)
(X_train.shape, X_test.shape)
features_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes == 'O']
for feature in features_nan:
    print('{}: {}% missing values'.format(feature, np.round(dataset[feature].isnull().mean(), 4)))

def replace_cat_feature(dataset, features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data
dataset = replace_cat_feature(dataset, features_nan)
dataset[features_nan].isnull().sum()
dataset.head()
numerical_with_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes != 'O']
for feature in numerical_with_nan:
    print('{}: {}% missing value'.format(feature, np.around(dataset[feature].isnull().mean(), 4)))
for feature in numerical_with_nan:
    median_value = dataset[feature].median()
    dataset[feature + 'nan'] = np.where(dataset[feature].isnull(), 1, 0)
    dataset[feature].fillna(median_value, inplace=True)
dataset[numerical_with_nan].isnull().sum()
dataset.head(50)
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    dataset[feature] = dataset['YrSold'] - dataset[feature]
dataset.head()
dataset[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
dataset.head()
import numpy as np
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_features:
    dataset[feature] = np.log(dataset[feature])
dataset.head()
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == 'O']
categorical_features
for feature in categorical_features:
    temp = dataset.groupby(feature)['SalePrice'].count() / len(dataset)
    temp_df = temp[temp > 0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df), dataset[feature], 'Rare_var')
dataset.head(50)
for feature in categorical_features:
    labels_ordered = dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    dataset[feature] = dataset[feature].map(labels_ordered)
dataset.head(10)
scaling_feature = [feature for feature in dataset.columns if feature not in ['Id', 'SalePerice']]
len(scaling_feature)
scaling_feature
dataset.head()
feature_scale = [feature for feature in dataset.columns if feature not in ['Id', 'SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()