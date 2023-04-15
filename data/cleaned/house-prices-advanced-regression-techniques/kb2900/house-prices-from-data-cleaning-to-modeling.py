import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5)

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.shape
test.head()
test.shape
train_num = train.select_dtypes(exclude=['object']).columns
train_num
train.describe()
from scipy.stats import shapiro
(stat, p) = shapiro(train['BsmtFinSF1'])
print('Skewness=%.3f' % train['BsmtFinSF1'].skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks normal (fail to reject H0)')
else:
    print('Data does not look normal (reject H0)')
sns.distplot(train['BsmtFinSF1'])
(stat, p) = shapiro(train['BsmtFullBath'])
print('Skewness=%.3f' % train['BsmtFullBath'].skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(train['BsmtFullBath'])
(stat, p) = shapiro(train['LotArea'])
print('Skewness=%.3f' % train['LotArea'].skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(train['LotArea'])
(stat, p) = shapiro(np.log(train['LotArea']))
print('After log transformation...')
print('Skewness=%.3f' % np.log(train['LotArea']).skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(np.log(train['LotArea']))
(stat, p) = shapiro(train['MasVnrArea'].dropna())
print('Skewness=%.3f' % train['MasVnrArea'].skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(train['MasVnrArea'].dropna())
masvnrarea_std = (train['MasVnrArea'] - np.mean(train['MasVnrArea'])) / np.std(train['MasVnrArea'])
(stat, p) = shapiro(masvnrarea_std.dropna())
print('Skewness=%.3f' % masvnrarea_std.skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(masvnrarea_std.dropna())
(stat, p) = shapiro(train['SalePrice'])
print('Skewness=%.3f' % train['SalePrice'].skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(train['SalePrice'])
(stat, p) = shapiro(np.log(train['SalePrice']))
print('After log transformation...')
print('Skewness=%.3f' % np.log(train['SalePrice']).skew())
print('Statistics=%.3f, p=%.3f' % (stat, p))
sns.distplot(np.log(train['SalePrice']))
train_num_corr = train[train_num].drop(['Id'], axis=1)
corr = pd.DataFrame(train_num_corr.corr(method='pearson')['SalePrice'])
corr.sort_values(['SalePrice'], ascending=False)
cmap = sns.cubehelix_palette(light=0.95, as_cmap=True)
sns.set(font_scale=1.2)
plt.figure(figsize=(9, 9))
sns.heatmap(abs(train_num_corr.corr(method='pearson')), vmin=0, vmax=1, square=True, cmap=cmap)
train_cat = train.select_dtypes(include=['object']).columns
train_cat
pd.set_option('display.max_rows', 300)
df_output = pd.DataFrame()
for i in range(len(train_cat)):
    c = train_cat[i]
    df = pd.DataFrame({'Variable': [c] * len(train[c].unique()), 'Level': train[c].unique(), 'Count': train[c].value_counts(dropna=False)})
    df['Percentage'] = 100 * df['Count'] / df['Count'].sum()
    df_output = df_output.append(df, ignore_index=True)
df_output
sns.set(style='whitegrid', rc={'figure.figsize': (10, 7), 'axes.labelsize': 12})
sns.boxplot(x='MSZoning', y='SalePrice', palette='Set2', data=train, linewidth=1.5)
col_order = train.groupby(['Neighborhood'])['SalePrice'].aggregate(np.median).reset_index().sort_values('SalePrice')
p = sns.boxplot(x='Neighborhood', y='SalePrice', palette='Set2', data=train, order=col_order['Neighborhood'], linewidth=1.5)
plt.setp(p.get_xticklabels(), rotation=45)
sns.boxplot(x='HouseStyle', y='SalePrice', palette='Set2', data=train, linewidth=1.5)
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train, hue='HouseStyle', style='HouseStyle', palette='colorblind')
print(train.isnull().sum())
train_missing = pd.DataFrame(train.isnull().sum() / len(train.index) * 100)
train_missing.columns = ['percent']
train_missing.loc[train_missing['percent'] > 15, 'column_select'] = True
train_col_select = train_missing.index[train_missing['column_select'] == True].tolist()
train_col_select
test_missing = pd.DataFrame(test.isnull().sum() / len(test.index) * 100)
test_missing.columns = ['percent']
test_missing.loc[test_missing['percent'] > 15, 'column_select'] = True
test_col_select = test_missing.index[test_missing['column_select'] == True].tolist()
test_col_select
train_col_select.pop(0)
test_col_select.pop(0)
train.drop(train_col_select, inplace=True, axis=1, errors='ignore')
test.drop(test_col_select, inplace=True, axis=1, errors='ignore')
train.head()
train.shape
test.head()
test.shape
from sklearn.base import TransformerMixin

class MissingDataImputer(TransformerMixin):

    def fit(self, X, y=None):
        """Extract mode for categorical features and median for numeric features"""
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].median() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        """Replace missingness with the array got from fit"""
        return X.fillna(self.fill)
train_nmissing = MissingDataImputer().fit_transform(train.iloc[:, 1:-1])
test_nmissing = MissingDataImputer().fit_transform(test.iloc[:, 1:])
train_nmissing.head()
print(train_nmissing.isnull().sum())
train_cat = train_nmissing.select_dtypes(include=['object']).columns
test_cat = test_nmissing.select_dtypes(include=['object']).columns
train_cat.difference(test_cat)
train_w_dummy = pd.get_dummies(train_nmissing, prefix_sep='_', drop_first=True, columns=train_cat)
test_w_dummy = pd.get_dummies(test_nmissing, prefix_sep='_', drop_first=True, columns=test_cat)
cat_dummies = [col for col in train_w_dummy if '_' in col and col.split('_')[0] in train_cat]
for col in test_w_dummy.columns:
    if '_' in col and col.split('_')[0] in train_cat and (col not in cat_dummies):
        test_w_dummy.drop(col, axis=1, inplace=True)
for col in cat_dummies:
    if col not in test_w_dummy.columns:
        test_w_dummy[col] = 0
train_cols = list(train_w_dummy.columns[:])
test_w_dummy = test_w_dummy[train_cols]
train_w_dummy.shape
test_w_dummy.shape
train_num = train_nmissing.select_dtypes(exclude=['object']).columns
test_num = test_nmissing.select_dtypes(exclude=['object']).columns
test_num.difference(train_num)
train_num_std = [col for col in train_num if abs(train_w_dummy[col].skew()) <= 1]
train_num_yjt = [col for col in train_num if abs(train_w_dummy[col].skew()) > 1]
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer