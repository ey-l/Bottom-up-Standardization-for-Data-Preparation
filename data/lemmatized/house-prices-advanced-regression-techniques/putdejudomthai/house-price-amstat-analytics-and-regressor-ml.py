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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1
_input0
_input0['Id']
print(f'Train Set has {_input1.shape[0]} rows and {_input1.shape[1]} columns.')
print(f'Test Set has {_input0.shape[0]} rows and {_input0.shape[1]} columns.')
_input1.describe().T
_input1.info()
msno.matrix(_input1)

def missing_total_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = total / len(df) * 100
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_total_percentage(_input1)
missing = _input1.isnull().sum() / len(_input1)
plt.figure(figsize=(15, 5))
missing.plot.bar()
plt.axhline(0.5, color='r')
missing = _input0.isnull().sum() / len(_input1)
plt.figure(figsize=(15, 5))
missing.plot.bar()
plt.axhline(0.5, color='r')
original_train = _input1.copy()
original_test = _input0.copy()
_input1 = _input1.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=False)
numerical_features = [col for col in _input1.columns if _input1[col].dtype != 'object']
categorical_features = [col for col in _input1.columns if _input1[col].dtype == 'object']
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
target_analysis(_input1['SalePrice'])
print('Skewness: ' + str(_input1['SalePrice'].skew()))
print('Kurtosis: ' + str(_input1['SalePrice'].kurt()))
target_analysis(np.log1p(_input1['SalePrice']))
print('Skewness: ' + str(np.log1p(_input1['SalePrice']).skew()))
print('Kurtosis: ' + str(np.log1p(_input1['SalePrice']).kurt()))
_input1['SalePrice'][:5]
from scipy.stats import shapiro

def check_normality(data):
    (stat, p) = shapiro(data)
    print('stat = %.2f, P-Value = %.2f' % (stat, p))
    if p > 0.05:
        print('Normal Distribution')
    else:
        print('Not Normal.')
check_normality(_input1['SalePrice'])
check_normality(np.log1p(_input1['SalePrice']))
corr_mat = _input1[numerical_features].corr()
plt.figure(figsize=(25, 25))
sns.heatmap(corr_mat, annot=True)
corr_mat['SalePrice'].sort_values(ascending=False)
print(corr_mat[corr_mat['SalePrice'] > 0.3].index)
print(len(corr_mat[corr_mat['SalePrice'] > 0.3].index))
selected_numerical_features = corr_mat['SalePrice'].sort_values(ascending=False)[:19].index
selected_numerical_features
plt.figure(figsize=(15, 10))
sns.heatmap(_input1[selected_numerical_features].corr(), annot=True)
original_train2 = _input1.copy()
original_test2 = _input0.copy()
_input1 = _input1.drop(['TotRmsAbvGrd', 'GarageArea', '1stFlrSF', 'GarageYrBlt'], axis=1, inplace=False)
final_numerical_features = [col for col in _input1.columns if _input1[col].dtype != 'object']
print(f'Number of numerical features after features selection: {len(final_numerical_features)}')
final_corr_mat = _input1[final_numerical_features].corr()
plt.figure(figsize=(20, 20))
sns.heatmap(final_corr_mat, annot=True)
print(final_corr_mat[final_corr_mat['SalePrice'] > 0.3].index)
print('Number of selected numerical features: ', len(final_corr_mat[final_corr_mat['SalePrice'] > 0.3].index))
final_selected_numerical_features = final_corr_mat['SalePrice'].sort_values(ascending=False)[:15].index
final_selected_numerical_features
plt.figure(figsize=(10, 8))
sns.heatmap(_input1[final_selected_numerical_features].corr(), annot=True)
numerical_features = final_selected_numerical_features
numerical_features
numerical_features
sns.regplot(data=_input1, y=np.log1p(_input1['SalePrice']), x='OverallQual', color='orange')
sns.scatterplot(data=_input1, y=np.log1p(_input1['SalePrice']), x='OverallQual')
plt.xlim(-1, 11)
import seaborn as sns
splot = sns.countplot(_input1['OverallQual'])
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2.0, p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
sns.boxplot(data=_input1, y='SalePrice', x='OverallQual')
_input1[_input1['OverallQual'] == 4]['SalePrice'].hist(bins=30)
_input1[_input1['OverallQual'] == 4]['SalePrice'].sort_values(ascending=False).index[0]
_input1.iloc[457][['OverallQual', 'SalePrice']]
_input1.drop([457], axis=0)
len(_input1.drop([457], axis=0))
import statsmodels.api as sm
X = _input1['OverallQual']
y = np.log1p(_input1['SalePrice'])
X = sm.add_constant(X)
X.columns = ['intercept', 'OverallQual']
lin_reg = sm.OLS(y, X)