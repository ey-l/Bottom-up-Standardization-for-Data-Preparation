"""Importing Data Manipulattion Moduls"""
import numpy as np
import pandas as pd
'Seaborn and Matplotlib Visualization'
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
sns.set_style({'axes.grid': False})

'plotly Visualization'
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
'Ignore deprecation and future, and user warnings.'
import warnings as wrn
wrn.filterwarnings('ignore', category=DeprecationWarning)
wrn.filterwarnings('ignore', category=FutureWarning)
wrn.filterwarnings('ignore', category=UserWarning)
'Read in train and test data from csv files'
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
'Train and test data at a glance'
train.head()
test.head()
'Dimensions of train and test data'
print('Dimensions of train data:', train.shape)
print('Dimensions of test data:', test.shape)
"Let's check the columns names"
train.columns.values
"Let's merge the train and test data and inspect the data type"
merged = pd.concat([train, test], axis=0, sort=True)

print('Dimensions of data:', merged.shape)
'Extracting numerical variables first'
num_merged = merged.select_dtypes(include=['int64', 'float64'])

print('\n')

'Plot histogram of numerical variables to validate pandas intuition.'

def draw_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure()
    for (i, var_name) in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=40, ax=ax, color='green', alpha=0.5, figsize=(40, 200))
        ax.set_title(var_name, fontsize=43)
        ax.tick_params(axis='both', which='major', labelsize=35)
        ax.tick_params(axis='both', which='minor', labelsize=35)
        ax.set_xlabel('')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

draw_histograms(num_merged, num_merged.columns, 19, 2)
'Convert MSSubClass, OverallQual, OverallCond, MoSold, YrSold into categorical variables.'
merged.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = merged.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')
'Check out the data type after correction'
merged.dtypes.value_counts()
'Function to plot scatter plot'

def scatter_plot(x, y, title, xaxis, yaxis, size, c_scale):
    trace = go.Scatter(x=x, y=y, mode='markers', marker=dict(color=y, size=size, showscale=True, colorscale=c_scale))
    layout = go.Layout(hovermode='closest', title=title, xaxis=dict(title=xaxis), yaxis=dict(title=yaxis))
    fig = go.Figure(data=[trace], layout=layout)
    return iplot(fig)
'Function to plot bar chart'

def bar_plot(x, y, title, yaxis, c_scale):
    trace = go.Bar(x=x, y=y, marker=dict(color=y, colorscale=c_scale))
    layout = go.Layout(hovermode='closest', title=title, yaxis=dict(title=yaxis))
    fig = go.Figure(data=[trace], layout=layout)
    return iplot(fig)
'Function to plot histogram'

def histogram_plot(x, title, yaxis, color):
    trace = go.Histogram(x=x, marker=dict(color=color))
    layout = go.Layout(hovermode='closest', title=title, yaxis=dict(title=yaxis))
    fig = go.Figure(data=[trace], layout=layout)
    return iplot(fig)
corr = train.corr()
(f, ax) = plt.subplots(figsize=(15, 12))
sns.heatmap(corr, linewidths=0.5, vmin=0, vmax=1, square=True)
k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

'Sactter plot of GrLivArea vs SalePrice.'
scatter_plot(train.GrLivArea, train.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
'Drop observations where GrLivArea is greater than 4000 sq.ft'
train.drop(train[train.GrLivArea > 4000].index, inplace=True)
train.reset_index(drop=True, inplace=True)
'Sactter plot of GrLivArea vs SalePrice.'
scatter_plot(train.GrLivArea, train.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
'Scatter plot of TotalBsmtSF Vs SalePrice'
scatter_plot(train.TotalBsmtSF, train.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
'Drop observations where TotlaBsmtSF is greater than 3000 sq.ft'
train.drop(train[train.TotalBsmtSF > 3000].index, inplace=True)
train.reset_index(drop=True, inplace=True)
'Scatter plot of TotalBsmtSF Vs SalePrice'
scatter_plot(train.TotalBsmtSF, train.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
'Scatter plot of YearBuilt Vs SalePrice'
scatter_plot(train.YearBuilt, np.log1p(train.SalePrice), 'YearBuilt Vs SalePrice', 'YearBuilt', 'SalePrice', 10, 'viridis')
'Drop observations where YearBulit is less than 1893 sq.ft'
train.drop(train[train.YearBuilt < 1900].index, inplace=True)
train.reset_index(drop=True, inplace=True)
'Scatter plot of YearBuilt Vs SalePrice'
scatter_plot(train.YearBuilt, np.log1p(train.SalePrice), 'YearBuilt Vs SalePrice', 'YearBuilt', 'SalePrice', 10, 'viridis')
'Scatter plot of GarageCars Vs SalePrice'
scatter_plot(train.GarageCars, np.log(train.SalePrice), 'GarageCars Vs SalePrice', 'GarageCars', 'SalePrice', 10, 'Electric')
'Scatter plot of GarageCars Vs SalePrice'
scatter_plot(train.OverallQual, np.log(train.SalePrice), 'OverallQual Vs SalePrice', 'OverallQual', 'SalePrice', 10, 'Bluered')
'Scatter plot of FullBath Vs SalePrice'
scatter_plot(train.FullBath, np.log(train.SalePrice), 'FullBath Vs SalePrice', 'FullBath', 'SalePrice', 10, 'RdBu')
'separate our target variable first'
y_train = train.SalePrice
'Drop SalePrice from train data.'
train.drop('SalePrice', axis=1, inplace=True)
'Now combine train and test data frame together.'
df_merged = pd.concat([train, test], axis=0)
'Dimensions of new data frame'
df_merged.shape
'Again convert MSSubClass, OverallQual, OverallCond, MoSold, YrSold into categorical variables.'
df_merged.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = df_merged.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')
df_merged.dtypes.value_counts()
'columns with missing observation'
missing_columns = df_merged.columns[df_merged.isnull().any()].values
'Number of columns with missing obervation'
total_missing_columns = np.count_nonzero(df_merged.isnull().sum())
print('We have ', total_missing_columns, 'features with missing values and those features (with missing values) are: \n\n', missing_columns)
'Simple visualization of missing variables'
plt.figure(figsize=(20, 8))
sns.heatmap(df_merged.isnull(), yticklabels=False, cbar=False, cmap='summer')
'Get and plot only the features (with missing values) and their corresponding missing values.'
missing_columns = len(df_merged) - df_merged.loc[:, np.sum(df_merged.isnull()) > 0].count()
x = missing_columns.index
y = missing_columns
title = 'Variables with Missing Values'
scatter_plot(x, y, title, 'Features Having Missing Observations', 'Missing Values', 20, 'Viridis')
missing_columns
'Impute by None where NaN means something.'
to_impute_by_none = df_merged.loc[:, ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType']]
for i in to_impute_by_none.columns:
    df_merged[i].fillna('None', inplace=True)
'These are categorical variables and will be imputed by mode.'
to_impute_by_mode = df_merged.loc[:, ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']]
for i in to_impute_by_mode.columns:
    df_merged[i].fillna(df_merged[i].mode()[0], inplace=True)
'The following variables are either discrete numerical or continuous numerical variables.So the will be imputed by median.'
to_impute_by_median = df_merged.loc[:, ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']]
for i in to_impute_by_median.columns:
    df_merged[i].fillna(df_merged[i].median(), inplace=True)
'We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical.'
df = df_merged.drop(columns=['Id', 'LotFrontage'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = df.apply(le.fit_transform)
df.head(2)
df['LotFrontage'] = df_merged['LotFrontage']
df = df.set_index('LotFrontage').reset_index()
df.head(2)
'correlation of df'
corr = df.corr()


'Impute LotFrontage with median of respective columns (i.e., BldgType)'
df_merged['LotFrontage'] = df_merged.groupby(['BldgType'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
'Is there any missing values left untreated??'
print('Missing variables left untreated: ', df_merged.columns[df_merged.isna().any()].values)
'Skewness and Kurtosis of SalePrice'
print('Skewness: %f' % y_train.skew())
print('Kurtosis: %f' % y_train.kurt())
'Plot the distribution of SalePrice with skewness.'
histogram_plot(y_train, 'SalePrice without Transformation', 'Abs Frequency', 'deepskyblue')
'Plot the distribution of SalePrice with skewness'
y_train = np.log1p(y_train)
title = 'SalePrice after Transformation (skewness: {:0.4f})'.format(y_train.skew())
histogram_plot(y_train, title, 'Abs Frequency', ' darksalmon')
'Now calculate the rest of the explanetory variables'
skew_num = pd.DataFrame(data=df_merged.select_dtypes(include=['int64', 'float64']).skew(), columns=['Skewness'])
skew_num_sorted = skew_num.sort_values(ascending=False, by='Skewness')
skew_num_sorted
' plot the skewness for rest of the explanetory variables'
bar_plot(skew_num_sorted.index, skew_num_sorted.Skewness, 'Skewness in Explanetory Variables', 'Skewness', 'Blackbody')
'Extract numeric variables merged data.'
df_merged_num = df_merged.select_dtypes(include=['int64', 'float64'])
'Make the tranformation of the explanetory variables'
df_merged_skewed = np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew() > 0.5].index])
df_merged_normal = df_merged_num[df_merged_num.skew()[df_merged_num.skew() < 0.5].index]
df_merged_num_all = pd.concat([df_merged_skewed, df_merged_normal], axis=1)
'Update numerical variables with transformed variables.'
df_merged_num.update(df_merged_num_all)
'Standarize numeric features with RobustScaler'
from sklearn.preprocessing import RobustScaler
'Creating scaler object.'
scaler = RobustScaler()
'Fit scaler object on train data.'