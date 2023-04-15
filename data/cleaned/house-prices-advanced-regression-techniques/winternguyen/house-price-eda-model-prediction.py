import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
d_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
d_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
d_train.head(2)
d_train.shape
d_test.shape
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.style as style
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
d_train.describe(include='all').T
d_train.describe()
d_test.isnull().sum().sort_values()
d_train['YrSold'] = d_train['YrSold'].apply(str)
d_train['MoSold'] = d_train['MoSold'].apply(str)
d_test['YrSold'] = d_test['YrSold'].apply(str)
d_test['MoSold'] = d_test['MoSold'].apply(str)
d_test.columns
cat_cols = np.array(d_test.columns[d_test.dtypes == object])
for feature in cat_cols:
    d_train[feature].fillna(d_train[feature].mode()[0], inplace=True)
    d_test[feature].fillna(d_test[feature].mode()[0], inplace=True)
num_cols = np.array(d_test.columns[d_test.dtypes != object])
for feature in num_cols:
    d_train = d_train.fillna(0)
    d_test = d_test.fillna(0)
d_train = d_train.fillna('Other')
d_test = d_test.fillna('Other')
d_train.plot(subplots=True, sharex=True, figsize=(20, 50))
d_train.corr()['SalePrice'].sort_values(ascending=False)
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize=(30, 30))
mask = np.zeros_like(d_train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(d_train.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, mask=mask, center=0)
plt.title('Heatmap of all the Features of Train data set', fontsize=25)
sns.set(style='ticks')
x = d_train['SalePrice']
(f, (ax_box, ax_hist)) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12, 7))
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)
plt.axvline(x=x.mean(), c='red')
plt.axvline(x=x.median(), c='green')
ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

print('Skewness: %f' % d_train['SalePrice'].skew())
print('Kurtosis: %f' % d_train['SalePrice'].kurt())
sns.set(style='ticks')
x = np.log1p(d_train['SalePrice'])
(f, (ax_box, ax_hist)) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12, 7))
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)
plt.axvline(x=x.mean(), c='red')
plt.axvline(x=x.median(), c='green')
ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

fig = px.box(d_train, x='OverallQual', y='SalePrice')
fig.show()
yprop = 'SalePrice'
xprop = 'OverallQual'
h = 'LotArea'
px.scatter(d_train, x=xprop, y=yprop, color=h, marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
yprop = 'SalePrice'
xprop = 'LotArea'
h = 'OverallCond'
px.scatter(d_train, x=xprop, y=yprop, color=h, marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
d_train = d_train.drop(d_train[(d_train['SalePrice'] > 740000) & (d_train['SalePrice'] < 756000)].index).reset_index(drop=True)
df = px.data.gapminder()
fig = px.scatter(d_train, y='SalePrice', x='LotArea', size='SalePrice', color='TotalBsmtSF', hover_name='LotArea', log_x=True, log_y=True, size_max=20)
fig.show()
df = px.data.iris()
fig = px.scatter(d_train, x='1stFlrSF', y='SalePrice', color='GarageCars', marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
fig.show()
d_train = d_train.drop(d_train[(d_train['1stFlrSF'] > 4690) & (d_train['1stFlrSF'] < 4700)].index).reset_index(drop=True)
sns.jointplot(data=d_train, x='GrLivArea', y='SalePrice', kind='reg', height=8)
d_train = d_train.drop(d_train[(d_train['GrLivArea'] > 4000) & (d_train['SalePrice'] < 250000)].index).reset_index(drop=True)
fig = px.violin(d_train, y='SalePrice', x='GarageCars', color=None, box=True, points='all', hover_data=d_train.columns)
fig.show()
d_train = d_train.drop(d_train[(d_train['GarageCars'] > 3) & (d_train['SalePrice'] < 290000)].index).reset_index(drop=True)
fig = px.scatter(d_train, x='GarageArea', y='SalePrice', color='OverallCond', marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
fig.show()
d_train = d_train.drop(d_train[(d_train['GarageArea'] > 1240) & (d_train['GarageArea'] < 1400)].index).reset_index(drop=True)
plt.figure(figsize=[15, 20])
feafures = ['LotArea', 'MSSubClass', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']
n = 1
for f in feafures:
    plt.subplot(6, 2, n)
    sns.boxplot(x=f, y='SalePrice', data=d_train)
    plt.title('Sale Price in function of {}'.format(f))
    n = n + 1
plt.tight_layout()

from scipy.stats import ttest_ind

def Series_stats(var, category, prop1, prop2):
    s1 = d_train[d_train[category] == prop1][var]
    s2 = d_train[d_train[category] == prop2][var]
    (t, p) = ttest_ind(s1, s2, equal_var=False)
    print('Two-sample t-test: t={}, p={}'.format(round(t, 5), p))
    if p < 0.05 and np.abs(t) > 1.96:
        print('\n REJECT the Null Hypothesis and state that: \n at 5% significance level, the mean {} of {}-{} and {}-{} are not equal.'.format(var, prop1, category, prop2, category))
        print('\n YES, the {} of {}-{} differ significantly from {}-{} in the current dataset.'.format(var, prop1, category, prop2, category))
        print('\n The mean value of {} for {}-{} is {} and for {}-{} is {}'.format(var, prop1, category, round(s1.mean(), 2), prop2, category, round(s2.mean(), 2)))
    else:
        print('\n FAIL to Reject the Null Hypothesis and state that: \n at 5% significance level, the mean {} of {} - {} and {} - {} are equal.'.format(var, prop1, category, prop2, category))
        print('\n NO, the {} of {}-{} NOT differ significantly from {}-{} in the current dataset'.format(var, prop1, category, prop2, category))
        print('\n The mean value of {} for {}-{} is {} and for {}-{} is {}'.format(var, prop1, category, round(s1.mean(), 2), prop2, category, round(s2.mean(), 2)))
Series_stats('SalePrice', 'OverallQual', 1, 10)
Series_stats('SalePrice', 'LotArea', 8450, 13175)
Series_stats('SalePrice', 'Street', 'Pave', 'Grvl')
d_test.Functional.unique()
Check = pd.DataFrame(index=None, columns=['Feature', 'Missing from Test to Train', 'Items'])
cols = np.array(d_test.columns[d_test.dtypes == object])
for fe in cols:
    listtrain = d_train[fe]
    listtest = d_test[fe]
    Check = Check.append(pd.Series({'Feature': fe, 'Missing from Test to Train': len(set(listtest).difference(listtrain)), 'Items': set(listtest).difference(listtrain)}), ignore_index=True)
Check
d_train.head(2)
d_train.isnull().sum()
d_test.isnull().sum()
Check = pd.DataFrame(index=None, columns=['Feature', 'Missing from Test to Train', 'Items'])
cols = np.array(d_test.columns[d_test.dtypes == object])
for fe in cols:
    listtrain = d_train[fe]
    listtest = d_test[fe]
    Check = Check.append(pd.Series({'Feature': fe, 'Missing from Test to Train': len(set(listtest).difference(listtrain)), 'Items': set(listtest).difference(listtrain)}), ignore_index=True)
Check
f_train = ['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']
f_test = ['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
df_train = pd.DataFrame(d_train, columns=f_train)
df_test = pd.DataFrame(d_test, columns=f_test)
from scipy.stats import norm, skew
numeric_feats = df_test.dtypes[df_test.dtypes != 'object'].index
skewed_feats = df_test[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 1]
high_skew
for feature in high_skew.index:
    df_train[feature] = np.log1p(df_train[feature])
    df_test[feature] = np.log1p(df_test[feature])
import copy
train = copy.deepcopy(df_train)
test = copy.deepcopy(df_test)
cols = np.array(df_train.columns[df_train.dtypes != object])
for i in train.columns:
    if i not in cols:
        train[i] = train[i].map(str)
        test[i] = test[i].map(str)
train.drop(columns=cols, inplace=True)
test.drop(columns=np.delete(cols, len(cols) - 1), inplace=True)
df_train.head(3)
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
cols = np.array(df_train.columns[df_train.dtypes != object])
d = defaultdict(LabelEncoder)
train = train.apply(lambda x: d[x.name].fit_transform(x))
test = test.apply(lambda x: d[x.name].transform(x))
train[cols] = df_train[cols]
test[np.delete(cols, len(cols) - 1)] = df_test[np.delete(cols, len(cols) - 1)]
train.head(2)
test.head(2)
test['YrBltAndRemod'] = test['YearBuilt'] + test['YearRemodAdd']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
test['Total_sqr_footage'] = test['BsmtFinSF1'] + test['1stFlrSF'] + test['2ndFlrSF']
test['Total_Bathrooms'] = test['FullBath'] + 0.5 * test['HalfBath'] + test['BsmtFullBath']
test['Total_porch_sf'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['WoodDeckSF']
train['YrBltAndRemod'] = train['YearBuilt'] + train['YearRemodAdd']
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train['Total_sqr_footage'] = train['BsmtFinSF1'] + train['1stFlrSF'] + train['2ndFlrSF']
train['Total_Bathrooms'] = train['FullBath'] + 0.5 * train['HalfBath'] + train['BsmtFullBath']
train['Total_porch_sf'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['WoodDeckSF']
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def Errors(model, X_train, y_train, X_test, y_test):
    ATrS = model.score(X_train, y_train)
    ATeS = model.score(X_test, y_test)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    MSE = mean_squared_error(y_test, y_pred)
    return (ATrS, ATeS, RMSE, MSE)
train.isnull().sum()
X = train.drop(columns=['SalePrice']).values
y = np.log1p(train['SalePrice'])
Z = test.values