import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
with open('_data/input/house-prices-advanced-regression-techniques/data_description.txt', 'r') as f:
    texts = f.readlines()
newlist = list()
for col in df.columns:
    for text in texts:
        if col in text:
            newlist.append(text.split(':'))
desc = dict()
for item in newlist:
    if len(item) == 2:
        desc[item[0]] = item[1]
print(desc['YearRemodAdd'])
print(desc['LandSlope'])
print(desc['MoSold'])
df_train.head(10).style.background_gradient(cmap='viridis')
df.info()
df.describe().transpose().style.background_gradient(cmap='magma')
print(df_train.shape)
print(df_test.shape)
var_num = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[var_num])
sns.distplot(df['SalePrice'])
df['SalePrice'].describe()
df['LogSalePrice'] = np.log10(df['SalePrice'])
sns.distplot(df['LogSalePrice'], color='r')
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
df[cate_feat].isnull().sum()
df[num_feat].isnull().sum()
sns.lmplot(x='LotArea', y='LotFrontage', data=df)
plt.ylabel('LotFrontage')
plt.xlabel('LotArea')
plt.title('LotArea vs LotFrontage')

lm = LinearRegression()
lm_X = df[df['LotFrontage'].notnull()]['LotArea'].values.reshape(-1, 1)
lm_y = df[df['LotFrontage'].notnull()]['LotFrontage'].values