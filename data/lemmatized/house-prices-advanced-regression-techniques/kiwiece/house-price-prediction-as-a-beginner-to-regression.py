import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
pd.set_option('display.max_columns', None)
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()
print(_input1.shape)
print(_input0.shape)
_input1.columns.drop('SalePrice') == _input0.columns
_input1['SalePrice'].isnull().sum()
all_df = pd.concat([_input1.drop('SalePrice', axis=1), _input0], axis=0, sort=False)
all_df.head()
len(all_df)
nan_count = all_df.isnull().sum().sort_values(ascending=False)[all_df.isnull().sum() > 0]
nan_count.head()
data_type = all_df.dtypes
data_type.head()
nan_table = pd.concat([nan_count, data_type], axis=1, keys=['nan_count', 'data_type'])
nan_table
nan_table = nan_table.dropna(inplace=False)
nan_table = nan_table.sort_values(['data_type', 'nan_count'])
nan_table
nan_table[['favourable_value', 'nan_percentage']] = [' ', 0.0]
for (i, row) in nan_table.iterrows():
    nan_table.at[i, 'nan_percentage'] = nan_table.loc[i].nan_count / len(all_df) * 100
    if all_df[i].dtype == 'float64':
        nan_table.at[i, 'favourable_value'] = all_df[i].median()
    elif all_df[i].dtype == 'object':
        nan_table.at[i, 'favourable_value'] = all_df[i].mode().values[0]
nan_table
arr_na = ['BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtExposure', 'BsmtCond', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC']

def search(array, data):
    for i in range(len(array)):
        if array[i] == data:
            return True
    return False
for (i, row) in nan_table.iterrows():
    if nan_table.loc[i].data_type == 'float64':
        all_df[i] = all_df[i].fillna(nan_table.loc[i].favourable_value, inplace=False)
    elif search(arr_na, i):
        all_df[i] = all_df[i].fillna('NA', inplace=False)
        nan_table.at[i, 'favourable_value'] = 'NA'
    else:
        all_df[i] = all_df[i].fillna(nan_table.loc[i].favourable_value, inplace=False)
all_df.isnull().sum().max()
nan_table
for (i, row) in nan_table.iterrows():
    if nan_table.at[i, 'nan_percentage'] >= 80:
        all_df = all_df.drop(i, axis=1, inplace=False)
all_df.columns
train_df_no_nan = all_df[:len(_input1)]
train_df_no_nan['SalePrice'] = _input1['SalePrice']
train_df_no_nan.head()
spcorr = train_df_no_nan.corr()
highly_corr_features = spcorr.index[abs(spcorr['SalePrice']) > 0.3]
plt.figure(figsize=(15, 15))
g = sns.heatmap(train_df_no_nan[highly_corr_features].corr(), annot=True, cmap='RdYlGn')
desired_numeric_features = highly_corr_features.drop(['TotalBsmtSF', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt'])
desired_numeric_features
(fig, axs) = plt.subplots(5, 3, figsize=(20, 20))
for (i, ax) in enumerate(fig.axes):
    ax.scatter(train_df_no_nan[desired_numeric_features[i]], train_df_no_nan['SalePrice'])
    ax.set_title(desired_numeric_features[i] + ' vs SalePrice')
    ax.set_xlabel(desired_numeric_features[i])
    ax.set_ylabel('SalePrice')
fig.tight_layout(pad=2.0)
from scipy.stats import norm, skew

def skew_fix(dframe, feat, run):
    if run == 'train':
        skewed_features = dframe[feat].apply(lambda x: skew(x)).sort_values(ascending=False)
        global highly_skewed
        highly_skewed = skewed_features[abs(skewed_features) > 0.5]
        print(highly_skewed)
    no_skew = dframe
    for i in highly_skewed.index:
        no_skew[i] = np.log1p(dframe[i])
    return no_skew
run = 'train'
train_df_no_skew = skew_fix(train_df_no_nan[desired_numeric_features], desired_numeric_features, run)
run = 'test'
highly_skewed = highly_skewed.drop('SalePrice', inplace=False)
test_df_no_nan = all_df[len(_input1):]
test_df_no_skew = skew_fix(test_df_no_nan[desired_numeric_features.drop('SalePrice')], desired_numeric_features.drop('SalePrice'), run)
categorical_features = all_df.dtypes[all_df.dtypes == 'object'].index
categorical_features
train_df_no_skew[categorical_features] = train_df_no_nan[categorical_features]
test_df_no_skew[categorical_features] = test_df_no_nan[categorical_features]
train_df_no_skew.columns.drop('SalePrice') == test_df_no_skew.columns
train_test_ns_df = pd.concat([train_df_no_skew.drop('SalePrice', axis=1), test_df_no_skew])
len(train_test_ns_df)
train_test_clean_df = pd.get_dummies(train_test_ns_df)
print(len(train_test_clean_df.columns))
train_test_clean_df.head()
x_train = train_test_clean_df[:len(_input1)]
x_test = train_test_clean_df[len(_input1):]
y_train = train_df_no_skew['SalePrice']
x = sm.add_constant(x_train)