import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('The shape of our training set: ', df_train.shape[0], 'houses', 'and', df_train.shape[1], 'features')
print('The shape of our testing set: ', df_test.shape[0], 'houses', 'and', df_test.shape[1], 'features')
print('The testing set has 1 feature less than the training set, which is SalePrice, the target to predict  ')
df_train.head()
df_test.head()
df_train.describe()
df_test.describe()
df_train.columns
df_test.columns
numeric = df_train.select_dtypes(exclude='object')
categorical = df_train.select_dtypes(include='object')
print('\nNumber of numeric features : ', len(numeric.axes[1]))
print('\n', numeric.axes[1])
print('\nNumber of categorical features : ', len(categorical.axes[1]))
print('\n', categorical.axes[1])
num_corr = numeric.corr()
table = num_corr['SalePrice'].sort_values(ascending=False).to_frame()
cm = sns.light_palette('green', as_cmap=True)
tb = table.style.background_gradient(cmap=cm)
tb
(f, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(df_train.corr(), annot=True, linewidths=0.1, fmt='.1f', ax=ax, cmap='YlGnBu')
k = 10
cols = df_train.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
na = df_train.shape[0]
nb = df_test.shape[0]
y_train = df_train['SalePrice'].to_frame()
c1 = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)
c1.drop(['SalePrice'], axis=1, inplace=True)
c1.drop(['Id'], axis=1, inplace=True)
print('Total size for train and test sets is :', c1.shape)

def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    plt.figure(figsize=(width, height))
    percentage = data.isnull().mean() * 100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column', fontsize=20, weight='bold')
    plt.text(len(data.isnull().sum() / len(data)) / 1.7, thresh + 12.5, 'Columns with more than %s%s missing values' % (thresh, '%'), fontsize=12, color='crimson', ha='left', va='top')
    plt.text(len(data.isnull().sum() / len(data)) / 1.7, thresh - 5, 'Columns with less than %s%s missing values' % (thresh, '%'), fontsize=12, color='green', ha='left', va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight='bold')

msv1(c1, 20, color=('silver', 'gold', 'lightgreen', 'skyblue', 'lightpink'))
c = c1.dropna(thresh=len(c1) * 0.8, axis=1)
print('We dropped ', c1.shape[1] - c.shape[1], ' features in the combined set')
print('The shape of the combined dataset after dropping features with more than 80% M.V.', c.shape)
allna = c.isnull().sum() / len(c) * 100
allna = allna.drop(allna[allna == 0].index).sort_values()

def msv2(data, width=12, height=8, color=('silver', 'gold', 'lightgreen', 'skyblue', 'lightpink'), edgecolor='black'):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    (fig, ax) = plt.subplots(figsize=(width, height))
    allna = data.isnull().sum() / len(data) * 100
    tightout = 0.008 * max(allna)
    allna = allna.drop(allna[allna == 0].index).sort_values().reset_index()
    mn = ax.barh(allna.iloc[:, 0], allna.iloc[:, 1], color=color, edgecolor=edgecolor)
    ax.set_title('Missing values percentage per column', fontsize=15, weight='bold')
    ax.set_xlabel('Percentage', weight='bold', size=15)
    ax.set_ylabel('Features with missing values', weight='bold')
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    for i in ax.patches:
        ax.text(i.get_width() + tightout, i.get_y() + 0.1, str(round(i.get_width(), 2)) + '%', fontsize=10, fontweight='bold', color='grey')

msv2(c)
NA = c[allna.index.to_list()]
NAcat = NA.select_dtypes(include='object')
NAnum = NA.select_dtypes(exclude='object')
print('We have :', NAcat.shape[1], 'categorical features with missing values')
print('We have :', NAnum.shape[1], 'numerical features with missing values')
NAnum.head()
NANUM = NAnum.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette('lime', as_cmap=True)
NANUM = NANUM.style.background_gradient(cmap=cm)
NANUM
c['MasVnrArea'] = c.MasVnrArea.fillna(0)
c['LotFrontage'] = c.LotFrontage.fillna(c.LotFrontage.median())
c['GarageYrBlt'] = c['GarageYrBlt'].fillna(1980)
bb = c[allna.index.to_list()]
nan = bb.select_dtypes(exclude='object')
N = nan.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette('lime', as_cmap=True)
N = N.style.background_gradient(cmap=cm)
N
NAcat.head()
NAcat1 = NAcat.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette('lime', as_cmap=True)
NAcat1 = NAcat1.style.background_gradient(cmap=cm)
NAcat1
fill_cols = ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']
for col in c[fill_cols]:
    c[col] = c[col].fillna(method='ffill')
dd = c[allna.index.to_list()]
w = dd.select_dtypes(include='object')
a = w.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette('lime', as_cmap=True)
a = a.style.background_gradient(cmap=cm)
a
NAcols = c.columns
for col in NAcols:
    if c[col].dtype == 'object':
        c[col] = c[col].fillna('None')
for col in NAcols:
    if c[col].dtype != 'object':
        c[col] = c[col].fillna(0)
c.isnull().sum().sort_values(ascending=False).head()
FillNA = c[allna.index.to_list()]
FillNAcat = FillNA.select_dtypes(include='object')
FC = FillNAcat.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette('lime', as_cmap=True)
FC = FC.style.background_gradient(cmap=cm)
FC
FillNAnum = FillNA.select_dtypes(exclude='object')
FM = FillNAnum.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette('lime', as_cmap=True)
FM = FM.style.background_gradient(cmap=cm)
FM
c.shape
c['TotalArea'] = c['TotalBsmtSF'] + c['1stFlrSF'] + c['2ndFlrSF'] + c['GrLivArea'] + c['GarageArea']
c['Bathrooms'] = c['FullBath'] + c['HalfBath'] * 0.5
c['Year average'] = (c['YearRemodAdd'] + c['YearBuilt']) / 2
c['MSSubClass'] = c['MSSubClass'].apply(str)
c['YrSold'] = c['YrSold'].astype(str)
c.shape
c['HasBsmt'] = pd.Series(len(c['TotalBsmtSF']), index=c.index)
c['HasBsmt'] = 0
c.loc[c['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
c.loc[c['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(c['TotalBsmtSF'])
cb = pd.get_dummies(c)
print('the shape of the original dataset', c.shape)
print('the shape of the encoded dataset', cb.shape)
print('We have ', cb.shape[1] - c.shape[1], 'new encoded features')
Train = cb[:na]
Test = cb[na:]
print(Train.shape)
print(y_train.shape)
print(Test.shape)
fig = plt.figure(figsize=(15, 15))
ax1 = plt.subplot2grid((3, 2), (0, 0))
plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'], color='yellowgreen', alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold')
ax1 = plt.subplot2grid((3, 2), (0, 1))
plt.scatter(x=df_train['TotalBsmtSF'], y=df_train['SalePrice'], color='red', alpha=0.5)
plt.axvline(x=5900, color='r', linestyle='-')
plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold')
ax1 = plt.subplot2grid((3, 2), (1, 0))
plt.scatter(x=df_train['1stFlrSF'], y=df_train['SalePrice'], color='deepskyblue', alpha=0.5)
plt.axvline(x=4000, color='r', linestyle='-')
plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold')
ax1 = plt.subplot2grid((3, 2), (1, 1))
plt.scatter(x=df_train['MasVnrArea'], y=df_train['SalePrice'], color='gold', alpha=0.9)
plt.axvline(x=1500, color='r', linestyle='-')
plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold')
ax1 = plt.subplot2grid((3, 2), (2, 0))
plt.scatter(x=df_train['GarageArea'], y=df_train['SalePrice'], color='orchid', alpha=0.5)
plt.axvline(x=1230, color='r', linestyle='-')
plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold')
ax1 = plt.subplot2grid((3, 2), (2, 1))
plt.scatter(x=df_train['TotRmsAbvGrd'], y=df_train['SalePrice'], color='tan', alpha=0.9)
plt.axvline(x=13, color='r', linestyle='-')
plt.title('TotRmsAbvGrd - Price scatter plot', fontsize=15, weight='bold')

df_train['GrLivArea'].sort_values(ascending=False).head(2)
df_train['TotalBsmtSF'].sort_values(ascending=False).head(1)
df_train['MasVnrArea'].sort_values(ascending=False).head(1)
df_train['1stFlrSF'].sort_values(ascending=False).head(1)
df_train['GarageArea'].sort_values(ascending=False).head(4)
df_train['TotRmsAbvGrd'].sort_values(ascending=False).head(1)
train = Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500)]
print('We removed ', Train.shape[0] - train.shape[0], 'outliers')
target = df_train[['SalePrice']]
target.loc[1298]
target.loc[523]
pos = [1298, 523, 297]
target.drop(target.index[pos], inplace=True)
print('We make sure that both train and target sets have the same row number after removing the outliers:')
print('Train: ', train.shape[0], 'rows')
print('Target:', target.shape[0], 'rows')
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15, 5))
ax1 = plt.subplot2grid((1, 2), (0, 0))
plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'], color='orchid', alpha=0.5)
plt.title('Area-Price plot with outliers', weight='bold', fontsize=18)
plt.axvline(x=4600, color='r', linestyle='-')
ax1 = plt.subplot2grid((1, 2), (0, 1))
plt.scatter(x=train['GrLivArea'], y=target['SalePrice'], color='navy', alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Area-Price plot without outliers', weight='bold', fontsize=18)

print('Skewness before log transform: ', df_train['GrLivArea'].skew())
print('Kurtosis before log transform: ', df_train['GrLivArea'].kurt())
from scipy.stats import skew
print('Skewness after log transform: ', train['GrLivArea'].skew())
print('Kurtosis after log transform: ', train['GrLivArea'].kurt())
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15, 10))
ax1 = plt.subplot2grid((2, 2), (0, 0))
sns.distplot(df_train.GrLivArea, color='plum')
plt.title('Before: Distribution of GrLivArea', weight='bold', fontsize=18)
ax1 = plt.subplot2grid((2, 2), (0, 1))
sns.distplot(df_train['1stFlrSF'], color='tan')
plt.title('Before: Distribution of 1stFlrSF', weight='bold', fontsize=18)
ax1 = plt.subplot2grid((2, 2), (1, 0))
sns.distplot(train.GrLivArea, color='plum')
plt.title('After: Distribution of GrLivArea', weight='bold', fontsize=18)
ax1 = plt.subplot2grid((2, 2), (1, 1))
sns.distplot(train['1stFlrSF'], color='tan')
plt.title('After: Distribution of 1stFlrSF', weight='bold', fontsize=18)

sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
print('Skewness before log transform: ', target['SalePrice'].skew())
print('Kurtosis before log transform: ', target['SalePrice'].kurt())
target['SalePrice'] = np.log1p(target['SalePrice'])
sns.distplot(target['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(target['SalePrice'], plot=plt)
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15, 5))
ax1 = plt.subplot2grid((1, 2), (0, 0))
plt.hist(df_train.SalePrice, bins=10, color='mediumpurple', alpha=0.5)
plt.title('Sale price distribution before normalization', weight='bold', fontsize=18)
ax1 = plt.subplot2grid((1, 2), (0, 1))
plt.hist(target.SalePrice, bins=10, color='darkcyan', alpha=0.5)
plt.title('Sale price distribution after normalization', weight='bold', fontsize=18)

print('Skewness after log transform: ', target['SalePrice'].skew())
print('Kurtosis after log transform: ', target['SalePrice'].kurt())
X = train
y = np.array(target)
print(X.shape)
print(y.shape)
print(Test.shape)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.simplefilter(action='ignore')
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
X_test = scaler.transform(Test)
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
MSEs = ms.cross_val_score(lreg, X, y, scoring='neg_mean_squared_error', cv=5)
meanMSE = np.mean(MSEs)
print(meanMSE)
print('RMSE = ' + str(math.sqrt(-meanMSE)))
import sklearn.model_selection as GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha': [x for x in range(1, 101)]}
ridge_reg = ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)