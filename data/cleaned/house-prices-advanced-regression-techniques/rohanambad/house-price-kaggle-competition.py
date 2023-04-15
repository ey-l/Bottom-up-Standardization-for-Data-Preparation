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
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df
df.head(10).style.background_gradient(cmap='viridis')
df.head()
df.tail()
df.describe().transpose().style.background_gradient(cmap='magma')
df.info()
var_num = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[var_num])
sns.distplot(df['SalePrice'])
df['SalePrice'].describe()
df['LogSalePrice'] = np.log10(df['SalePrice'])
sns.distplot(df['LogSalePrice'], color='r')
df.duplicated().sum()
pd.options.display.max_rows = None
df.isnull().sum()
pd.reset_option('max_rows')
df.LotFrontage.value_counts()
sns.distplot(df.LotFrontage)

df['LotFrontage'].mean()
print('The Percentage of data missing in LotFrontage is ', df.LotFrontage.isnull().sum() / len(df) * 100)
df.loc[df['LotFrontage'].isnull() == True]
df.loc[df['LotFrontage'].isnull() == True, 'LotFrontage'] = 70.0
d1 = df[['Alley', 'PoolQC', 'Fence', 'MiscFeature']]
a = d1.isnull().sum() / len(d1) * 100
a
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
d2 = df[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]
b = d2.isnull().sum() / len(d2) * 100
b
df['BsmtQual'] = df['BsmtQual'].fillna('NA')
df['BsmtCond'] = df['BsmtCond'].fillna('NA')
df['BsmtExposure'] = df['BsmtExposure'].fillna('NA')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NA')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NA')
df.MasVnrType.value_counts()
df.loc[df['MasVnrType'].isnull() == True, 'MasVnrType'] = 'None'
sns.distplot(df.MasVnrArea)
df['MasVnrArea'].median()
df.loc[df['MasVnrArea'].isnull() == True, 'MasVnrArea'] = 0.0
d3 = df[['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']]
c = d3.isnull().sum() / len(d3) * 100
c
df['GarageType'] = df['GarageType'].fillna('NA')
df['GarageFinish'] = df['GarageFinish'].fillna('NA')
df['GarageQual'] = df['GarageQual'].fillna('NA')
df['GarageCond'] = df['GarageCond'].fillna('NA')
sns.distplot(df.GarageYrBlt)
df.GarageYrBlt.median()
df.loc[df['GarageYrBlt'].isnull() == True, 'GarageYrBlt'] = 1980.0
df.Electrical.value_counts()
df.loc[df['Electrical'].isnull() == True, 'Electrical'] = 'SBrkr'
df.FireplaceQu.value_counts()
df.loc[df['FireplaceQu'].isnull() == True, 'FireplaceQu'] = 'NA'
df.isnull().sum()
cate_feat = list(df.select_dtypes(include=[object]).columns)
num_feat = list(df.select_dtypes(include=[int, float]).columns)
print(cate_feat)
print('\n')
print(num_feat)
pd.options.display.float_format = '{:,.2f}'.format
corr_matrix = df[num_feat].corr()
corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0
sns.heatmap(corr_matrix, vmax=1.0, vmin=-1.0, linewidths=0.1, annot_kws={'size': 9, 'color': 'black'}, annot=True)
plt.title('SalePrice Correlation')
corr = df.corr()['SalePrice'].sort_values(ascending=False)[2:8]
corr
(f, ax) = plt.subplots(nrows=6, ncols=1, figsize=(20, 40))
for (i, col) in enumerate(corr.index):
    sns.scatterplot(x=col, y='SalePrice', data=df, ax=ax[i], color='darkorange')
    ax[i].set_title(f'{col} vs SalePrice')
(f, ax) = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=df)
fig.axis(ymin=0, ymax=900000)
plt.xticks(rotation=90)
plt.tight_layout()
yr_built = pd.DataFrame({'Count': df['YearBuilt'].value_counts()[:10]}).reset_index()
yr_built.rename(columns={'index': 'Year'}, inplace=True)
plt.figure(figsize=(20, 10))
sns.barplot(x='Year', y='Count', data=yr_built)
plt.title('Year Built')
df.groupby('MoSold').mean()['SalePrice'].sort_values(ascending=False).plot(kind='bar')
dtype_objects = list((columns for columns in df.select_dtypes([object]).columns))
dtype_objects
len(dtype_objects)
plt.figure(figsize=(15, 150))
plotnumber = 1
for c in dtype_objects:
    ax = plt.subplot(20, 2, plotnumber)
    b = sns.countplot(x=df[c], palette='Set2')
    plt.xticks(rotation=70)
    plotnumber += 1
    for bar in b.patches:
        b.annotate(format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 6), textcoords='offset points')

dtype_float = list((columns for columns in df.select_dtypes([float]).columns))
dtype_float
plt.figure(figsize=(15, 150))
plotnumber = 1
for a in dtype_float:
    ax = plt.subplot(20, 2, plotnumber)
    sns.distplot(x=df[a], color='purple')
    plt.xticks(rotation=70)
    plotnumber += 1

plt.figure(figsize=(5, 5))
labels = ['Y', 'N']
size = df['CentralAir'].value_counts()
colors = ['lightgreen', 'lightslategray']
explode = [0, 0.3]
plt.pie(size, labels=labels, colors=colors, explode=explode, autopct='%.2f%%', shadow=True, startangle=-30, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.legend(labels, loc='upper right', title='Category')

plt.figure(figsize=(5, 5))
labels = ['Fin', 'RFn', 'Unf', 'NA']
size = df['GarageFinish'].value_counts()
colors = ['purple', 'lightblue', 'pink', 'yellow']
explode = [0, 0.1, 0, 0]
plt.pie(size, labels=labels, colors=colors, explode=explode, autopct='%.2f%%', shadow=True)
circle = plt.Circle((0, 0), 0.8, color='white')
p = plt.gcf()
p.gca().add_artist(circle)

plt.figure(figsize=(15, 9))
sns.countplot(x='CentralAir', hue='BedroomAbvGr', palette='terrain', data=df).set(title='BedroomAbvGr vs CentralAir')

sns.scatterplot(x='GarageArea', y='GarageYrBlt', data=df, color='lightcoral')

data = df
plt.figure(figsize=(20, 150), facecolor='white')
plotnumber = 1
for c in dtype_objects:
    ax = plt.subplot(20, 2, plotnumber)
    sns.barplot(x=data[c], y=data.SalePrice, palette='Set3')
    plotnumber += 1
    plt.xticks(rotation=70)

plt.figure(figsize=(15, 9))
splot = sns.barplot(x='Electrical', y='MSSubClass', hue='CentralAir', palette='nipy_spectral', data=data)

box = df[['LotArea', 'YearBuilt', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']]
plt.figure(figsize=(15, 15), facecolor='white')
plotnum = 1
for c in box:
    if plotnum < 9:
        a = plt.subplot(4, 2, plotnum)
        sns.distplot(box[c])
    plotnum += 1
plt.tight_layout()
box = df[['LotArea', 'YearBuilt', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt']]
box
plt.figure(figsize=(10, 30), facecolor='white')
plotnumber = 1
for c in box:
    ax = plt.subplot(8, 1, plotnumber)
    sns.boxplot(data[c], color='green')
    plotnumber = plotnumber + 1

from scipy.stats import skew
numerical_features = data.dtypes[data.dtypes != 'object'].index
skewed_features = df[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'skew': skewed_features})
skewness
skewness = skewness[abs(skewness > 0.8)]
print('There are {} skewed numerical features to box cox transform'.format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_features = skewness.index
lamda = 0.15
for features in skewed_features:
    df[features] += 1
    df[features] = boxcox1p(df[features], lamda)
df[skewed_features] = np.log1p(df[skewed_features])
print('Skewness has been Handled using Box Cox Transformation')
data_object = df.select_dtypes(include='object').columns
print(data_object)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for features in data_object:
    df[features] = le.fit_transform(df[features].astype(str))
print(df.info())
x = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']
from sklearn.preprocessing import MinMaxScaler
mc = MinMaxScaler()
scaled_x = mc.fit_transform(x)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(scaled_x, y, test_size=0.2, random_state=0)
print('Traning Shape = ', x_train.shape)
print('Testing Shape = ', x_test.shape)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()