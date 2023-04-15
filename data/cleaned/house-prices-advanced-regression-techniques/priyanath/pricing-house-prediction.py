import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(train_df.info())
print('-*' * 20)
print(test_df.info())
sample = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample.head()
train_df.columns
train_df.head()
train_df.describe()
train_df['SalePrice'].describe()
import seaborn as sns
from matplotlib import pyplot as plt
sns.distplot(train_df['SalePrice'])
print('Skewness: %f' % train_df['SalePrice'].skew())
print('Kurtosis: %f' % train_df['SalePrice'].kurt())
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.regplot(x=var, y='SalePrice', data=data)
var = 'TotalBsmtSF'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.regplot(x=var, y='SalePrice', data=data)
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
(f, ax) = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
var = 'YearBuilt'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
(f, ax) = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
corrmat = train_df.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], size=2.5)

total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train_df = train_df.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)
train_df.isnull().sum().max()
from sklearn.preprocessing import StandardScaler
saleprice_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
from scipy.stats import norm
from scipy import stats

def diagnostic_plots(train_df, variable):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.distplot(train_df[variable], fit=norm)
    plt.subplot(1, 2, 2)
    stats.probplot(train_df[variable], dist='norm', plot=plt)

diagnostic_plots(train_df, 'SalePrice')
train_df['SalePrice'] = np.log(train_df['SalePrice'] + 1)
diagnostic_plots(train_df, 'SalePrice')
diagnostic_plots(train_df, 'GrLivArea')
train_df['GrLivArea'] = np.log(train_df['GrLivArea'] + 1)
diagnostic_plots(train_df, 'GrLivArea')
diagnostic_plots(train_df, 'TotalBsmtSF')
train_df['GrLivArea'] = 1 / (train_df['GrLivArea'] + 1)
diagnostic_plots(train_df, 'GrLivArea')
train_df.head()
train_df.columns
train_df.info()
train_df.describe()
corrmat = train_df.corr()
(f, ax) = plt.subplots(figsize=(50, 20))
sns.heatmap(corrmat, vmax=0.8, square=True, annot=True)
train_df.drop(['MSSubClass', 'OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtFullBath', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Id'], axis=1, inplace=True)
train_df.info()
from sklearn.preprocessing import LabelEncoder
cols = ('ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'Functional', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'CentralAir')
for c in cols:
    lbl = LabelEncoder()