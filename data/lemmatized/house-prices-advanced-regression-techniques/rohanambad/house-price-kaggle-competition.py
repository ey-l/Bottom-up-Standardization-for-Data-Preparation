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
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import math
import warnings
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': (20, 15)})
sns.set_style('whitegrid')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1
_input1.head(10).style.background_gradient(cmap='viridis')
_input1.head()
_input1.tail()
_input1.describe().transpose().style.background_gradient(cmap='magma')
_input1.info()
var_num = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[var_num])
sns.distplot(_input1['SalePrice'])
_input1['SalePrice'].describe()
_input1['LogSalePrice'] = np.log10(_input1['SalePrice'])
sns.distplot(_input1['LogSalePrice'], color='r')
_input1.duplicated().sum()
pd.options.display.max_rows = None
_input1.isnull().sum()
pd.reset_option('max_rows')
_input1.LotFrontage.value_counts()
sns.distplot(_input1.LotFrontage)
_input1['LotFrontage'].mean()
print('The Percentage of data missing in LotFrontage is ', _input1.LotFrontage.isnull().sum() / len(_input1) * 100)
_input1.loc[_input1['LotFrontage'].isnull() == True]
_input1.loc[_input1['LotFrontage'].isnull() == True, 'LotFrontage'] = 70.0
d1 = _input1[['Alley', 'PoolQC', 'Fence', 'MiscFeature']]
a = d1.isnull().sum() / len(d1) * 100
a
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
d2 = _input1[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]
b = d2.isnull().sum() / len(d2) * 100
b
_input1['BsmtQual'] = _input1['BsmtQual'].fillna('NA')
_input1['BsmtCond'] = _input1['BsmtCond'].fillna('NA')
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('NA')
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna('NA')
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna('NA')
_input1.MasVnrType.value_counts()
_input1.loc[_input1['MasVnrType'].isnull() == True, 'MasVnrType'] = 'None'
sns.distplot(_input1.MasVnrArea)
_input1['MasVnrArea'].median()
_input1.loc[_input1['MasVnrArea'].isnull() == True, 'MasVnrArea'] = 0.0
d3 = _input1[['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']]
c = d3.isnull().sum() / len(d3) * 100
c
_input1['GarageType'] = _input1['GarageType'].fillna('NA')
_input1['GarageFinish'] = _input1['GarageFinish'].fillna('NA')
_input1['GarageQual'] = _input1['GarageQual'].fillna('NA')
_input1['GarageCond'] = _input1['GarageCond'].fillna('NA')
sns.distplot(_input1.GarageYrBlt)
_input1.GarageYrBlt.median()
_input1.loc[_input1['GarageYrBlt'].isnull() == True, 'GarageYrBlt'] = 1980.0
_input1.Electrical.value_counts()
_input1.loc[_input1['Electrical'].isnull() == True, 'Electrical'] = 'SBrkr'
_input1.FireplaceQu.value_counts()
_input1.loc[_input1['FireplaceQu'].isnull() == True, 'FireplaceQu'] = 'NA'
_input1.isnull().sum()
cate_feat = list(_input1.select_dtypes(include=[object]).columns)
num_feat = list(_input1.select_dtypes(include=[int, float]).columns)
print(cate_feat)
print('\n')
print(num_feat)
pd.options.display.float_format = '{:,.2f}'.format
corr_matrix = _input1[num_feat].corr()
corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0
sns.heatmap(corr_matrix, vmax=1.0, vmin=-1.0, linewidths=0.1, annot_kws={'size': 9, 'color': 'black'}, annot=True)
plt.title('SalePrice Correlation')
corr = _input1.corr()['SalePrice'].sort_values(ascending=False)[2:8]
corr
(f, ax) = plt.subplots(nrows=6, ncols=1, figsize=(20, 40))
for (i, col) in enumerate(corr.index):
    sns.scatterplot(x=col, y='SalePrice', data=_input1, ax=ax[i], color='darkorange')
    ax[i].set_title(f'{col} vs SalePrice')
(f, ax) = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=_input1)
fig.axis(ymin=0, ymax=900000)
plt.xticks(rotation=90)
plt.tight_layout()
yr_built = pd.DataFrame({'Count': _input1['YearBuilt'].value_counts()[:10]}).reset_index()
yr_built = yr_built.rename(columns={'index': 'Year'}, inplace=False)
plt.figure(figsize=(20, 10))
sns.barplot(x='Year', y='Count', data=yr_built)
plt.title('Year Built')
_input1.groupby('MoSold').mean()['SalePrice'].sort_values(ascending=False).plot(kind='bar')
dtype_objects = list((columns for columns in _input1.select_dtypes([object]).columns))
dtype_objects
len(dtype_objects)
plt.figure(figsize=(15, 150))
plotnumber = 1
for c in dtype_objects:
    ax = plt.subplot(20, 2, plotnumber)
    b = sns.countplot(x=_input1[c], palette='Set2')
    plt.xticks(rotation=70)
    plotnumber += 1
    for bar in b.patches:
        b.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 6), textcoords='offset points')
dtype_float = list((columns for columns in _input1.select_dtypes([float]).columns))
dtype_float
plt.figure(figsize=(15, 150))
plotnumber = 1
for a in dtype_float:
    ax = plt.subplot(20, 2, plotnumber)
    sns.distplot(x=_input1[a], color='purple')
    plt.xticks(rotation=70)
    plotnumber += 1
plt.figure(figsize=(5, 5))
labels = ['Y', 'N']
size = _input1['CentralAir'].value_counts()
colors = ['lightgreen', 'lightslategray']
explode = [0, 0.3]
plt.pie(size, labels=labels, colors=colors, explode=explode, autopct='%.2f%%', shadow=True, startangle=-30, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.legend(labels, loc='upper right', title='Category')
plt.figure(figsize=(5, 5))
labels = ['Fin', 'RFn', 'Unf', 'NA']
size = _input1['GarageFinish'].value_counts()
colors = ['purple', 'lightblue', 'pink', 'yellow']
explode = [0, 0.1, 0, 0]
plt.pie(size, labels=labels, colors=colors, explode=explode, autopct='%.2f%%', shadow=True)
circle = plt.Circle((0, 0), 0.8, color='white')
p = plt.gcf()
p.gca().add_artist(circle)
plt.figure(figsize=(15, 9))
sns.countplot(x='CentralAir', hue='BedroomAbvGr', palette='terrain', data=_input1).set(title='BedroomAbvGr vs CentralAir')
sns.scatterplot(x='GarageArea', y='GarageYrBlt', data=_input1, color='lightcoral')
data = _input1
plt.figure(figsize=(20, 150), facecolor='white')
plotnumber = 1
for c in dtype_objects:
    ax = plt.subplot(20, 2, plotnumber)
    sns.barplot(x=data[c], y=data.SalePrice, palette='Set3')
    plotnumber += 1
    plt.xticks(rotation=70)
plt.figure(figsize=(15, 9))
splot = sns.barplot(x='Electrical', y='MSSubClass', hue='CentralAir', palette='nipy_spectral', data=data)
box = _input1[['LotArea', 'YearBuilt', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']]
plt.figure(figsize=(15, 15), facecolor='white')
plotnum = 1
for c in box:
    if plotnum < 9:
        a = plt.subplot(4, 2, plotnum)
        sns.distplot(box[c])
    plotnum += 1
plt.tight_layout()
box = _input1[['LotArea', 'YearBuilt', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']]
box
plt.figure(figsize=(10, 30), facecolor='white')
plotnumber = 1
for c in box:
    ax = plt.subplot(8, 1, plotnumber)
    sns.boxplot(data[c], color='green')
    plotnumber = plotnumber + 1
from scipy.stats import skew
numerical_features = data.dtypes[data.dtypes != 'object'].index
skewed_features = _input1[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'skew': skewed_features})
skewness
skewness = skewness[abs(skewness > 0.8)]
print('There are {} skewed numerical features to box cox transform'.format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_features = skewness.index
lamda = 0.15
for features in skewed_features:
    _input1[features] += 1
    _input1[features] = boxcox1p(_input1[features], lamda)
_input1[skewed_features] = np.log1p(_input1[skewed_features])
print('Skewness has been Handled using Box Cox Transformation')
data_object = _input1.select_dtypes(include='object').columns
print(data_object)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for features in data_object:
    _input1[features] = le.fit_transform(_input1[features].astype(str))
print(_input1.info())
x = _input1.drop(['Id', 'SalePrice'], axis=1)
y = _input1['SalePrice']
from sklearn.preprocessing import MinMaxScaler
mc = MinMaxScaler()
scaled_x = mc.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(scaled_x, y, test_size=0.2, random_state=0)
print('Traning Shape = ', x_train.shape)
print('Testing Shape = ', x_test.shape)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()