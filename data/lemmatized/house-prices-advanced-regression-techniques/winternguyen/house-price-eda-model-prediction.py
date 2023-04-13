import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head(2)
_input1.shape
_input0.shape
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
_input1.describe(include='all').T
_input1.describe()
_input0.isnull().sum().sort_values()
_input1['YrSold'] = _input1['YrSold'].apply(str)
_input1['MoSold'] = _input1['MoSold'].apply(str)
_input0['YrSold'] = _input0['YrSold'].apply(str)
_input0['MoSold'] = _input0['MoSold'].apply(str)
_input0.columns
cat_cols = np.array(_input0.columns[_input0.dtypes == object])
for feature in cat_cols:
    _input1[feature] = _input1[feature].fillna(_input1[feature].mode()[0], inplace=False)
    _input0[feature] = _input0[feature].fillna(_input0[feature].mode()[0], inplace=False)
num_cols = np.array(_input0.columns[_input0.dtypes != object])
for feature in num_cols:
    _input1 = _input1.fillna(0)
    _input0 = _input0.fillna(0)
_input1 = _input1.fillna('Other')
_input0 = _input0.fillna('Other')
_input1.plot(subplots=True, sharex=True, figsize=(20, 50))
_input1.corr()['SalePrice'].sort_values(ascending=False)
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize=(30, 30))
mask = np.zeros_like(_input1.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(_input1.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, mask=mask, center=0)
plt.title('Heatmap of all the Features of Train data set', fontsize=25)
sns.set(style='ticks')
x = _input1['SalePrice']
(f, (ax_box, ax_hist)) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12, 7))
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)
plt.axvline(x=x.mean(), c='red')
plt.axvline(x=x.median(), c='green')
ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis: %f' % _input1['SalePrice'].kurt())
sns.set(style='ticks')
x = np.log1p(_input1['SalePrice'])
(f, (ax_box, ax_hist)) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12, 7))
sns.boxplot(x, ax=ax_box)
sns.distplot(x, ax=ax_hist)
plt.axvline(x=x.mean(), c='red')
plt.axvline(x=x.median(), c='green')
ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)
fig = px.box(_input1, x='OverallQual', y='SalePrice')
fig.show()
yprop = 'SalePrice'
xprop = 'OverallQual'
h = 'LotArea'
px.scatter(_input1, x=xprop, y=yprop, color=h, marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
yprop = 'SalePrice'
xprop = 'LotArea'
h = 'OverallCond'
px.scatter(_input1, x=xprop, y=yprop, color=h, marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
_input1 = _input1.drop(_input1[(_input1['SalePrice'] > 740000) & (_input1['SalePrice'] < 756000)].index).reset_index(drop=True)
df = px.data.gapminder()
fig = px.scatter(_input1, y='SalePrice', x='LotArea', size='SalePrice', color='TotalBsmtSF', hover_name='LotArea', log_x=True, log_y=True, size_max=20)
fig.show()
df = px.data.iris()
fig = px.scatter(_input1, x='1stFlrSF', y='SalePrice', color='GarageCars', marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
fig.show()
_input1 = _input1.drop(_input1[(_input1['1stFlrSF'] > 4690) & (_input1['1stFlrSF'] < 4700)].index).reset_index(drop=True)
sns.jointplot(data=_input1, x='GrLivArea', y='SalePrice', kind='reg', height=8)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 250000)].index).reset_index(drop=True)
fig = px.violin(_input1, y='SalePrice', x='GarageCars', color=None, box=True, points='all', hover_data=_input1.columns)
fig.show()
_input1 = _input1.drop(_input1[(_input1['GarageCars'] > 3) & (_input1['SalePrice'] < 290000)].index).reset_index(drop=True)
fig = px.scatter(_input1, x='GarageArea', y='SalePrice', color='OverallCond', marginal_y='violin', marginal_x='box', trendline='ols', template='simple_white')
fig.show()
_input1 = _input1.drop(_input1[(_input1['GarageArea'] > 1240) & (_input1['GarageArea'] < 1400)].index).reset_index(drop=True)
plt.figure(figsize=[15, 20])
feafures = ['LotArea', 'MSSubClass', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual']
n = 1
for f in feafures:
    plt.subplot(6, 2, n)
    sns.boxplot(x=f, y='SalePrice', data=_input1)
    plt.title('Sale Price in function of {}'.format(f))
    n = n + 1
plt.tight_layout()
from scipy.stats import ttest_ind

def Series_stats(var, category, prop1, prop2):
    s1 = _input1[_input1[category] == prop1][var]
    s2 = _input1[_input1[category] == prop2][var]
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
_input0.Functional.unique()
Check = pd.DataFrame(index=None, columns=['Feature', 'Missing from Test to Train', 'Items'])
cols = np.array(_input0.columns[_input0.dtypes == object])
for fe in cols:
    listtrain = _input1[fe]
    listtest = _input0[fe]
    Check = Check.append(pd.Series({'Feature': fe, 'Missing from Test to Train': len(set(listtest).difference(listtrain)), 'Items': set(listtest).difference(listtrain)}), ignore_index=True)
Check
_input1.head(2)
_input1.isnull().sum()
_input0.isnull().sum()
Check = pd.DataFrame(index=None, columns=['Feature', 'Missing from Test to Train', 'Items'])
cols = np.array(_input0.columns[_input0.dtypes == object])
for fe in cols:
    listtrain = _input1[fe]
    listtest = _input0[fe]
    Check = Check.append(pd.Series({'Feature': fe, 'Missing from Test to Train': len(set(listtest).difference(listtrain)), 'Items': set(listtest).difference(listtrain)}), ignore_index=True)
Check
f_train = ['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']
f_test = ['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
df_train = pd.DataFrame(_input1, columns=f_train)
df_test = pd.DataFrame(_input0, columns=f_test)
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
train = train.drop(columns=cols, inplace=False)
test = test.drop(columns=np.delete(cols, len(cols) - 1), inplace=False)
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