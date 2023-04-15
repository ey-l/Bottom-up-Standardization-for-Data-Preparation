import numpy as np
import pandas as pd
import os
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
import matplotlib.gridspec as gridspec
import missingno as msno
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train
test
test['Id']
print(f'Train Set has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test Set has {test.shape[0]} rows and {test.shape[1]} columns.')
train.describe().T
train.info()
msno.matrix(train)

def missing_total_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = total / len(df) * 100
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_total_percentage(train)
missing = train.isnull().sum() / len(train)
plt.figure(figsize=(15, 5))
missing.plot.bar()
plt.axhline(0.5, color='r')
missing = test.isnull().sum() / len(train)
plt.figure(figsize=(15, 5))
missing.plot.bar()
plt.axhline(0.5, color='r')
original_train = train.copy()
original_test = test.copy()
train.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)
numerical_features = [col for col in train.columns if train[col].dtype != 'object']
categorical_features = [col for col in train.columns if train[col].dtype == 'object']
print(f'Numerical features dimension: {len(numerical_features)}')
print(f'Categorical features dimension: {len(categorical_features)}')
numerical_features
from scipy import stats

def target_analysis(target):
    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0, :3])
    ax1.set_title('Histogram')
    sns.histplot(target, ax=ax1, kde=True)
    ax2 = fig.add_subplot(grid[1, :1])
    ax2.set_title('Q-Q Plot')
    stats.probplot(target, plot=ax2)
    ax3 = fig.add_subplot(grid[2, :3])
    ax3.set_title('Box Plot')
    sns.boxplot(target, orient='v', ax=ax3)
    print(f'skweness is {target.skew()}')

target_analysis(train['SalePrice'])
print('Skewness: ' + str(train['SalePrice'].skew()))
print('Kurtosis: ' + str(train['SalePrice'].kurt()))
target_analysis(np.log1p(train['SalePrice']))
print('Skewness: ' + str(np.log1p(train['SalePrice']).skew()))
print('Kurtosis: ' + str(np.log1p(train['SalePrice']).kurt()))
train['SalePrice'][:5]
from scipy.stats import shapiro

def check_normality(data):
    (stat, p) = shapiro(data)
    print('stat = %.2f, P-Value = %.2f' % (stat, p))
    if p > 0.05:
        print('Normal Distribution')
    else:
        print('Not Normal.')
check_normality(train['SalePrice'])
check_normality(np.log1p(train['SalePrice']))
corr_mat = train[numerical_features].corr()
plt.figure(figsize=(25, 25))
sns.heatmap(corr_mat, annot=True)
corr_mat['SalePrice'].sort_values(ascending=False)
print(corr_mat[corr_mat['SalePrice'] > 0.3].index)
print(len(corr_mat[corr_mat['SalePrice'] > 0.3].index))
selected_numerical_features = corr_mat['SalePrice'].sort_values(ascending=False)[:19].index
selected_numerical_features
plt.figure(figsize=(15, 10))
sns.heatmap(train[selected_numerical_features].corr(), annot=True)
original_train2 = train.copy()
original_test2 = test.copy()
train.drop(['TotRmsAbvGrd', 'GarageArea', '1stFlrSF', 'GarageYrBlt'], axis=1, inplace=True)
final_numerical_features = [col for col in train.columns if train[col].dtype != 'object']
print(f'Number of numerical features after features selection: {len(final_numerical_features)}')
final_corr_mat = train[final_numerical_features].corr()
plt.figure(figsize=(20, 20))
sns.heatmap(final_corr_mat, annot=True)
print(final_corr_mat[final_corr_mat['SalePrice'] > 0.3].index)
print('Number of selected numerical features: ', len(final_corr_mat[final_corr_mat['SalePrice'] > 0.3].index))
final_selected_numerical_features = final_corr_mat['SalePrice'].sort_values(ascending=False)[:15].index
final_selected_numerical_features
plt.figure(figsize=(10, 8))
sns.heatmap(train[final_selected_numerical_features].corr(), annot=True)
numerical_features = final_selected_numerical_features
numerical_features
numerical_features
sns.regplot(data=train, y=np.log1p(train['SalePrice']), x='OverallQual', color='orange')
sns.scatterplot(data=train, y=np.log1p(train['SalePrice']), x='OverallQual')
plt.xlim(-1, 11)
import seaborn as sns
splot = sns.countplot(train['OverallQual'])
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
sns.boxplot(data=train, y='SalePrice', x='OverallQual')
train[train['OverallQual'] == 4]['SalePrice'].hist(bins=30)
train[train['OverallQual'] == 4]['SalePrice'].sort_values(ascending=False).index[0]
train.iloc[457][['OverallQual', 'SalePrice']]
train.drop([457], axis=0)
len(train.drop([457], axis=0))
import statsmodels.api as sm
X = train['OverallQual']
y = np.log1p(train['SalePrice'])
X = sm.add_constant(X)
X.columns = ['intercept', 'OverallQual']
lin_reg = sm.OLS(y, X)