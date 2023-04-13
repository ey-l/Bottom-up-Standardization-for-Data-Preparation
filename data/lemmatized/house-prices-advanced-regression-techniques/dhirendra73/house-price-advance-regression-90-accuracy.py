import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns', None)
DATA_PATH = '_data/input/house-prices-advanced-regression-techniques/'
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df = pd.concat([_input0.assign(ind='test'), _input1.assign(ind='train')])
df.head()
df.info()
df.columns
df.shape
nan_cols = [i for i in df.columns if df[i].isnull().sum() >= 1]
print(nan_cols)
pd.set_option('display.max_rows', None)
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
missing_value_df.sort_values(['percent_missing'], ascending=False)
df = df.loc[:, df.isnull().mean() < 0.9]
df.shape
df_train_categorical = df.select_dtypes(include=['object']).columns.tolist()
print(len(df_train_categorical))
df_train_numeric = df.select_dtypes(exclude=['object']).columns.tolist()
df_year_feature = [year_feature for year_feature in df_train_numeric if 'Yr' in year_feature or 'Year' in year_feature]
print(df_year_feature)
for feature in df_year_feature:
    if feature != 'YrSold':
        data = df.copy()
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
_input1.duplicated().sum()
discrete_df = [feature for feature in df_train_numeric if len(df[feature].unique()) < 25 and feature not in df_year_feature + ['ID']]
for feature in discrete_df:
    N = 25
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar(color=plt.cm.Paired(np.arange(N)))
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
continous_df = [feature for feature in df_train_numeric if feature not in discrete_df + ['Id'] + df_year_feature + ['SalePrice']]
for feature in continous_df:
    data = _input1.copy()
    data.groupby(feature)['SalePrice'].median().plot.hist()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
for feature in df_train_categorical:
    N = 25
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar(color=plt.cm.Paired(np.arange(N)))
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
numerical_with_nan = [feature for feature in df.columns if df[feature].isnull().sum() > 1 and df[feature].dtypes != 'O']
numerical_with_nan.remove('SalePrice')
numerical_with_nan
for feature in numerical_with_nan:
    median_value = df[feature].median()
    df[feature] = df[feature].fillna(median_value, inplace=False)
df[numerical_with_nan].isnull().sum()
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    df[feature] = df['YrSold'] - df[feature]
df[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
for feature in continous_df:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
import numpy as np
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for feature in num_features:
    df[feature] = np.log(df[feature])
for feature in num_features:
    sns.kdeplot(data=_input1, x=feature, legend=True)
num_feature = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

def iqr_feature(features, df):
    dict = {}
    max_lmt = []
    min_lmt = []
    for feature in features:
        q1 = df[feature].quantile(0.25)
        q2 = df[feature].quantile(0.75)
        IQR = q2 - q1
        max_limit = q2 + 1.5 * IQR
        max_lmt.append(max_limit)
        min_limit = q1 - 1.5 * IQR
        min_lmt.append(min_limit)
    return (max_lmt, min_lmt)
(max_lmt, min_lmt) = iqr_feature(num_feature, df)
min_lmt
s1 = pd.Series(max_lmt)
s2 = pd.Series(min_lmt)
df_min_max = pd.DataFrame(list(zip(num_feature, s1, s2)), columns=['num_features', 'max_lmt', 'min_lmt'])
df_min_max
df.shape
df['LotFrontage'] = np.where(df['LotFrontage'] < 3.700798, df['LotFrontage'].quantile(0.05), df['LotFrontage'])
df['LotFrontage'] = np.where(df['LotFrontage'] > 4.750255, df['LotFrontage'].quantile(0.95), df['LotFrontage'])
df.shape
sns.boxplot(y=df['LotFrontage'])
df['LotArea'] = np.where(df['LotArea'] < 8.65046, df['LotArea'].quantile(0.05), df['LotArea'])
df['LotArea'] = np.where(df['LotArea'] > 10.010846, df['LotArea'].quantile(0.95), df['LotArea'])
sns.boxplot(y=df['LotArea'])
df['1stFlrSF'] = np.where(df['1stFlrSF'] < 6.085527, df['1stFlrSF'].quantile(0.05), df['1stFlrSF'])
df['1stFlrSF'] = np.where(df['1stFlrSF'] > 7.925098, df['1stFlrSF'].quantile(0.95), df['1stFlrSF'])
df.shape
df['GrLivArea'] = np.where(df['GrLivArea'] < 6.370592, df['GrLivArea'].quantile(0.05), df['GrLivArea'])
df['GrLivArea'] = np.where(df['GrLivArea'] > 8.119484, df['GrLivArea'].quantile(0.95), df['GrLivArea'])
df.shape
df['SalePrice'] = np.where(df['SalePrice'] < 3937.5, df['SalePrice'].quantile(0.05), df['SalePrice'])
df['SalePrice'] = np.where(df['SalePrice'] > 340037.5, df['SalePrice'].quantile(0.95), df['SalePrice'])
df.shape
df['SalePrice'].tail()
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
categorical_features.remove('ind')
for feature in categorical_features:
    print(df[feature].value_counts())
for feature in categorical_features:
    temp = df.groupby(feature)['SalePrice'].count() / len(df)
    temp_df = temp[temp > 0.01].index
    df[feature] = np.where(df[feature].isin(temp_df), df[feature], 'Rare_var')
for feature in categorical_features:
    labels_ordered = df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for (i, k) in enumerate(labels_ordered, 0)}
    df[feature] = df[feature].map(labels_ordered)
df.head()
(test, train) = (df[df['ind'].eq('test')], df[df['ind'].eq('train')])
test = test.drop('SalePrice', axis=1, inplace=False)
train = train.drop('ind', axis=1, inplace=False)
nan_feature = [feature for feature in test.columns if test[feature].isnull().sum() >= 1]
nan_feature.remove('GarageCars')
for feature in nan_feature:
    median_value = df[feature].median()
    test[feature] = test[feature].fillna(median_value, inplace=False)
test[nan_feature].isnull().sum()
md = test['GarageCars'].mode()
print(md)
test['GarageCars'] = test['GarageCars'].fillna(2, inplace=False)
test.isnull().sum()
test = test.drop('ind', axis=1, inplace=False)
y_data = train['SalePrice']
x_data = train.drop('SalePrice', axis=1)
(x_data.shape, test.shape)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x_data, y_data, test_size=0.2)
x_train.isnull().sum()
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(x_train, y_train)
mutual_info
mutual_info = pd.Series(mutual_info)
mutual_info.index = x_train.columns
mutual_info.sort_values(ascending=False)
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=80)