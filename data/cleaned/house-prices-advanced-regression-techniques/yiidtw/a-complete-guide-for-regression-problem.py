import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')

train_csv = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
final_csv = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_csv['SalePrice'].describe()
print('How many feature candidates do we have? %d' % (len(train_csv.columns) - 1))
null_in_train_csv = train_csv.isnull().sum()
null_in_train_csv = null_in_train_csv[null_in_train_csv > 0]
null_in_train_csv.sort_values(inplace=True)
null_in_train_csv.plot.bar()
sns.heatmap(train_csv.corr(), vmax=0.8, square=True)
arr_train_cor = train_csv.corr()['SalePrice']
idx_train_cor_gt0 = arr_train_cor[arr_train_cor > 0].sort_values(ascending=False).index.tolist()
print('How many feature candidates have positive correlation with SalePrice(including itself)? %d' % len(idx_train_cor_gt0))
arr_train_cor[idx_train_cor_gt0]
idx_meta = ['SalePrice', 'GrLivArea', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'OverallQual', 'Fireplaces', 'GarageCars']
train_meta = train_csv[idx_meta].copy()
train_meta.head(n=5)
null_in_masvnrarea = train_meta[train_meta['MasVnrArea'].isnull()].index.tolist()
zero_in_masvnrarea = train_meta['MasVnrArea'][train_meta['MasVnrArea'] == 0].index.tolist()
print('How many null value in MasVnrArea? %d / 1460' % len(null_in_masvnrarea))
print('How many zero value in MasVnrArea? %d / 1460' % len(zero_in_masvnrarea))
train_meta['MasVnrArea'][null_in_masvnrarea] = 0
print('How many null value in MasVnrArea after filling in null value? %d / 1460' % train_meta['MasVnrArea'].isnull().sum())
sns.pairplot(train_meta)
train_meta[(train_meta['GrLivArea'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()
train_meta[(train_meta['TotalBsmtSF'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()
train_meta[(train_meta['1stFlrSF'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()
train_clean = train_meta.drop([523, 1298])
nonzero_in_masvnrarea = train_clean['MasVnrArea'][train_clean['MasVnrArea'] != 0].index.tolist()
print('How many non-zero value in MasVnrArea now? %d / 1458' % len(nonzero_in_masvnrarea))
train_clean['has_MasVnrArea'] = 0
train_clean['has_MasVnrArea'][nonzero_in_masvnrarea] = 1
train_clean['TotalBsmtSF'][train_clean['TotalBsmtSF'] > 0].describe()
bins_totalbsmtsf = [-1, 1, 1004, 4000]
train_clean['binned_TotalBsmtSF'] = np.digitize(train_clean['TotalBsmtSF'], bins_totalbsmtsf)
train_clean['1stFlrSF'].describe()
bins_1stflrsf = [0, 882, 1086, 1390, 4000]
train_clean['binned_1stFlrSF'] = np.digitize(train_clean['1stFlrSF'], bins_1stflrsf)
train_clean['2ndFlrSF'][train_clean['2ndFlrSF'] > 0].describe()
bins_2ndflrsf = [-1, 1, 625, 772, 924, 4000]
train_clean['binned_2ndFlrSF'] = np.digitize(train_clean['2ndFlrSF'], bins_2ndflrsf)
train_clean['SFcross'] = (train_clean['binned_TotalBsmtSF'] - 1) * (5 * 4) + (train_clean['binned_1stFlrSF'] - 1) * 5 + train_clean['binned_2ndFlrSF']

def draw2by2log(arr):
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    sns.distplot(arr, fit=norm)
    plt.subplot(2, 2, 3)
    stats.probplot(arr, plot=plt)
    plt.subplot(2, 2, 2)
    sns.distplot(np.log(arr), fit=norm)
    plt.subplot(2, 2, 4)
    stats.probplot(np.log(arr), plot=plt)
draw2by2log(train_clean['SalePrice'])
draw2by2log(train_clean['GrLivArea'])
train_clean.head(n=5)
idx_tree = ['SalePrice', 'GrLivArea', 'OverallQual', 'Fireplaces', 'GarageCars', 'has_MasVnrArea', 'SFcross']
train_tree = train_clean[idx_tree]
train_tree.head(n=5)
sns.pairplot(train_tree)
print('Max Fireplaces value in train.csv: %d, in test.csv: %d' % (train_csv['Fireplaces'].max(), final_csv['Fireplaces'].max()))
print('Min Fireplaces value in train.csv: %d, in test.csv: %d' % (train_csv['Fireplaces'].min(), final_csv['Fireplaces'].min()))
print('Max GarageCars value in train.csv: %d, in test.csv: %d' % (train_csv['GarageCars'].max(), final_csv['GarageCars'].max()))
print('Min GarageCars value in train.csv: %d, in test.csv: %d' % (train_csv['GarageCars'].min(), final_csv['GarageCars'].min()))
dummy_fields = ['OverallQual', 'Fireplaces', 'GarageCars', 'has_MasVnrArea', 'SFcross']
train_dist = train_tree[['SalePrice', 'OverallQual', 'GrLivArea']].copy()
for field in dummy_fields:
    dummies = pd.get_dummies(train_tree.loc[:, field], prefix=field)
    train_dist = pd.concat([train_dist, dummies], axis=1)
train_dist['GarageCars_5'] = 0
train_dist['Fireplaces_4'] = 0
train_dist.head(n=5)
print('The dimension for the input of distance-based model is %d x %d' % (train_dist.shape[0], train_dist.shape[1] - 1))
from sklearn.model_selection import train_test_split
random_state = 7
(xt_train_test, xt_valid, yt_train_test, yt_valid) = train_test_split(train_tree['SalePrice'], train_tree.drop('SalePrice', axis=1), test_size=0.2, random_state=random_state)
(xd_train_test, xd_valid, yd_train_test, yd_valid) = train_test_split(train_dist['SalePrice'], train_dist.drop('SalePrice', axis=1), test_size=0.2, random_state=random_state)
(xt_train, xt_test, yt_train, yt_test) = train_test_split(yt_train_test, xt_train_test, test_size=0.2, random_state=random_state)
(xd_train, xd_test, yd_train, yd_test) = train_test_split(yd_train_test, xd_train_test, test_size=0.2, random_state=random_state)
print('number of training set: %d\nnumber of testing set: %d\nnumber of validation set: %d\ntotal: %d' % (len(xt_train), len(xt_test), len(xt_valid), len(xt_train) + len(xt_test) + len(xt_valid)))

def rmse(arr1, arr2):
    return np.sqrt(np.mean((arr1 - arr2) ** 2))
from sklearn.linear_model import LinearRegression
lm = LinearRegression()