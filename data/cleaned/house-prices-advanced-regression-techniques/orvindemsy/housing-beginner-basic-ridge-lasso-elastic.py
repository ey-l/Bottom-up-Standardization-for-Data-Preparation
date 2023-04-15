import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
print('Data shape: ', train.shape)
print('There are %d instances' % train.shape[0])
print('There are %d features' % train.shape[1])
corrmat = train.corr()
plt.figure(figsize=(10, 10))
ax = sns.heatmap(corrmat, square=True, vmax=1, vmin=-1)
ax.set_title('Correlation Heatmap of Housing Pricing Train data')

train['OverallQual']
sns.set(style='darkgrid', palette='muted')
target = 'SalePrice'
var1 = 'OverallQual'
var2 = 'GrLivArea'
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.boxplot(x=var1, y=target, data=train, ax=ax1)
ax1.set_title('Correlation values %.3f' % corrmat.loc[target, var1])
sns.scatterplot(x=var2, y=target, data=train, ax=ax2)
ax2.set_title('Correlation values %.3f' % corrmat.loc[target, var2])
fig.tight_layout()

train.shape
total = train.isna().sum().sort_values(ascending=False)
percent = total / len(train)
missing = pd.concat([total, percent], axis=1)
missing.columns = ['total', 'percentage']
corr_tmp = corrmat.SalePrice
corr_tmp.name = 'corrval'
missingcorr = missing.merge(corr_tmp, how='outer', left_index=True, right_index=True).sort_values(by='percentage', ascending=False)
missingcorr.head(20)
del_cols = missingcorr[missingcorr['percentage'] > missingcorr.loc['Electrical', 'percentage']].index
del_cols
print('Initial data shape:', train.shape)
train_nona = train.drop(columns=del_cols)
print('After dropping columns:', train_nona.shape)
train_nona = train_nona.dropna(axis=0, how='any')
print('After dropping instance of `Electrical`: ', train_nona.shape)
print('Total missing values in data after cleaning: ', train_nona.isna().sum().sum())
highcorr = corrmat.SalePrice.sort_values(ascending=False)
highcorr = highcorr[highcorr > 0.5]
highcorr
highcorrmat = corrmat.loc[highcorr.index, highcorr.index]
highcorrmat
plt.figure(figsize=(8, 5))
sns.heatmap(highcorrmat, annot=True, fmt='.2f')

drop_cols = ['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
sel_train = train_nona[highcorrmat.index]
sel_train = sel_train.drop(columns=drop_cols)
sel_train.shape
sns.pairplot(sel_train)

print(sel_train['GrLivArea'].sort_values()[-2:].index)
print(sel_train['TotalBsmtSF'].sort_values()[-1:].index)
sel_train = sel_train.drop(index=sel_train['GrLivArea'].sort_values()[-2:].index)
(fig, (ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
fig.suptitle('After Removing Outliers')
sns.scatterplot(x='GrLivArea', y='SalePrice', data=sel_train, ax=ax1)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=sel_train, ax=ax2)

from scipy.stats import norm
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4))
sns.distplot(sel_train.SalePrice, kde=True, fit=norm, ax=ax1)
_ = stats.probplot(sel_train.SalePrice, plot=ax2)
fig.tight_layout()

sel_train['SalePrice'] = np.log1p(sel_train.SalePrice)
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4))
sns.distplot(sel_train.SalePrice, kde=True, fit=norm, ax=ax1)
_ = stats.probplot(sel_train.SalePrice, plot=ax2)
fig.tight_layout()

(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4))
sns.distplot(sel_train.GrLivArea, kde=True, fit=norm, ax=ax1)
_ = stats.probplot(sel_train.GrLivArea, plot=ax2)
fig.tight_layout()

sel_train.GrLivArea = np.log(sel_train.GrLivArea)
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4))
sns.distplot(sel_train.GrLivArea, kde=True, fit=norm, ax=ax1)
_ = stats.probplot(sel_train.GrLivArea, plot=ax2)
fig.tight_layout()

(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4))
sns.distplot(sel_train.TotalBsmtSF, kde=True, fit=norm, ax=ax1)
_ = stats.probplot(sel_train.TotalBsmtSF, plot=ax2)
fig.tight_layout()

sel_train['TotalBsmtSF'] = np.log1p(sel_train.TotalBsmtSF)
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(9, 4))
sns.distplot(sel_train.TotalBsmtSF[sel_train.TotalBsmtSF > 0], kde=True, fit=norm, ax=ax1)
_ = stats.probplot(sel_train.TotalBsmtSF[sel_train.TotalBsmtSF > 0], plot=ax2)
fig.tight_layout()

y = sel_train.SalePrice
X = sel_train.drop(columns='SalePrice')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42)

def rmse(model, X, y, cv):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))
    return rmse.mean()
lin = LinearRegression()
rmse_sc = rmse(lin, X, y, 5)
rmse_sc
all_scores = []
all_scores.append(dict(model='OLD', score=rmse_sc))
ridge = Ridge(alpha=1)
rmse(ridge, X, y, cv=5)
alphas = np.logspace(5, -5, 50)
alphas
ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')