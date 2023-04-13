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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(_input1.shape)
_input1.head()
features_with_na = [features for features in _input1.columns if _input1[features].isnull().sum() > 1]
for feature in features_with_na:
    print(feature, np.round(_input1[feature].isnull().mean(), 4), ' % missing values')
for feature in features_with_na:
    data = _input1.copy()
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
print('Id of Houses {}'.format(len(_input1.Id)))
numerical_features = [feature for feature in _input1.columns if _input1[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
_input1[numerical_features].head()
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
year_feature
for feature in year_feature:
    print(feature, _input1[feature].unique())
_input1.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs YearSold')
year_feature
for feature in year_feature:
    if feature != 'YrSold':
        data = _input1.copy()
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
discrete_feature = [feature for feature in numerical_features if len(_input1[feature].unique()) < 25 and feature not in year_feature + ['Id']]
print('Discrete Variables Count: {}'.format(len(discrete_feature)))
discrete_feature
_input1[discrete_feature].head()
for feature in discrete_feature:
    data = _input1.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature + year_feature + ['Id']]
print('Continuous feature Count {}'.format(len(continuous_feature)))
for feature in continuous_feature:
    data = _input1.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
for feature in continuous_feature:
    data = _input1.copy()
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
    data = _input1.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
categorical_features = [feature for feature in _input1.columns if data[feature].dtypes == 'O']
categorical_features
_input1[categorical_features].head()
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature, len(_input1[feature].unique())))
for feature in categorical_features:
    data = _input1.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(_input1, _input1['SalePrice'], test_size=0.1, random_state=0)
(X_train.shape, X_test.shape)
features_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1 and _input1[feature].dtypes == 'O']
for feature in features_nan:
    print('{}: {}% missing values'.format(feature, np.round(_input1[feature].isnull().mean(), 4)))

def replace_cat_feature(dataset, features_nan):
    data = _input1.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data
_input1 = replace_cat_feature(_input1, features_nan)
_input1[features_nan].isnull().sum()
_input1.head()
numerical_with_nan = [feature for feature in _input1.columns if _input1[feature].isnull().sum() > 1 and _input1[feature].dtypes != 'O']
for feature in numerical_with_nan:
    print('{}: {}% missing value'.format(feature, np.around(_input1[feature].isnull().mean(), 4)))
for feature in numerical_with_nan:
    median_value = _input1[feature].median()
    _input1[feature + 'nan'] = np.where(_input1[feature].isnull(), 1, 0)
    _input1[feature] = _input1[feature].fillna(median_value, inplace=False)
_input1[numerical_with_nan].isnull().sum()
_input1.head(50)
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    _input1[feature] = _input1['YrSold'] - _input1[feature]
_input1.head()
_input1[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
_input1.head()
import numpy as np
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_features:
    _input1[feature] = np.log(_input1[feature])
_input1.head()
categorical_features = [feature for feature in _input1.columns if _input1[feature].dtype == 'O']
categorical_features
for feature in categorical_features:
    temp = _input1.groupby(feature)['SalePrice'].count() / len(_input1)
    temp_df = temp[temp > 0.01].index
    _input1[feature] = np.where(_input1[feature].isin(temp_df), _input1[feature], 'Rare_var')
_input1.head(50)
for feature in categorical_features:
    labels_ordered = _input1.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    _input1[feature] = _input1[feature].map(labels_ordered)
_input1.head(10)
scaling_feature = [feature for feature in _input1.columns if feature not in ['Id', 'SalePerice']]
len(scaling_feature)
scaling_feature
_input1.head()
feature_scale = [feature for feature in _input1.columns if feature not in ['Id', 'SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()