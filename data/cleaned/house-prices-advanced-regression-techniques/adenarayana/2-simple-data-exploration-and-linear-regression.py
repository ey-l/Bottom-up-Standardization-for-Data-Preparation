import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
out_sample = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
data_train['SalePrice'].describe().transpose()
g = sns.displot(data_train['SalePrice'], kde=True, bins=20)
g.fig.set_size_inches(10, 8)
plt.savefig('./displot.png', dpi=150, bbox_inches='tight')

print('Sale Price skewness:', data_train['SalePrice'].skew())
data_train.info()
data_train_num = data_train.select_dtypes(exclude='object')
data_train_num.shape
corr = data_train_num.corr()
corr_target = corr['SalePrice']
corr_target.sort_values(ascending=False)
corr_target = corr_target[corr_target > 0.5]
corr_target.sort_values(ascending=False)
col_name = corr_target.keys()
corr_map = data_train_num[col_name].corr()
(fig, ax) = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_map, square=True, annot=True, ax=ax)
plt.savefig('./heat_map.png', dpi=150, bbox_inches='tight')

col_name = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'FullBath', '1stFlrSF', 'GrLivArea', 'GarageCars', 'SalePrice']
data_train[col_name].isnull().sum()
new_data = data_train[col_name]
plt.figure(figsize=(20, 15))
plt.subplot(3, 3, 1)
sns.scatterplot(x=new_data['OverallQual'], y=new_data['SalePrice'])
plt.subplot(3, 3, 2)
sns.scatterplot(x=new_data['YearBuilt'], y=new_data['SalePrice'])
plt.subplot(3, 3, 3)
sns.scatterplot(x=new_data['YearRemodAdd'], y=new_data['SalePrice'])
plt.subplot(3, 3, 4)
sns.scatterplot(x=new_data['FullBath'], y=new_data['SalePrice'])
plt.subplot(3, 3, 5)
sns.scatterplot(x=new_data['1stFlrSF'], y=new_data['SalePrice'])
plt.subplot(3, 3, 6)
sns.scatterplot(x=new_data['GrLivArea'], y=new_data['SalePrice'])
plt.subplot(3, 3, 7)
sns.scatterplot(x=new_data['GarageCars'], y=new_data['SalePrice'])

new_data[new_data['GrLivArea'] > 4000]
new_data.drop(new_data[new_data['GrLivArea'] > 4000].index, inplace=True)
sns.scatterplot(x=new_data['GrLivArea'], y=new_data['SalePrice'])

X_train = new_data.drop(['SalePrice'], axis=1)
Y_train = new_data['SalePrice']
(X_train, X_val, Y_train, Y_val) = train_test_split(X_train, Y_train, test_size=0.3, random_state=101)
lr = LinearRegression()