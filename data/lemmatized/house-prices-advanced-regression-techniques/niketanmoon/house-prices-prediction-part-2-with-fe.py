import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.info()
_input1.hist(bins=50, figsize=(16, 16))
(fig, ax) = plt.subplots(figsize=(10, 6))
ax.grid()
ax.scatter(_input1['GrLivArea'], _input1['SalePrice'], c='#3f72af', zorder=3, alpha=0.9)
ax.axvline(4500, c='#112d4e', ls='--', zorder=2)
ax.set_xlabel('Ground living area (sq. ft)', labelpad=10)
ax.set_ylabel('Sale price ($)', labelpad=10)
sns.boxplot(_input1.GrLivArea)
numerical_df = _input1.select_dtypes(exclude=['object'])
numerical_df = numerical_df.drop(['Id'], axis=1)
for column in numerical_df:
    plt.figure(figsize=(16, 4))
    sns.set_theme(style='whitegrid')
    sns.boxplot(numerical_df[column])
_input1.get('SalePrice').describe()
(f, ax) = plt.subplots(figsize=(16, 16))
sns.distplot(_input1.get('SalePrice'), kde=False)
corrmat = _input1.corr()
(f, ax) = plt.subplots(figsize=(16, 16))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.figure(figsize=(16, 16))
columns = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
correlation_matrix = np.corrcoef(_input1[columns].values.T)
sns.set(font_scale=1.25)
heat_map = sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values, xticklabels=columns.values)
_input1 = _input1[_input1.GrLivArea < 4500]
total = _input0.isna().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total'])
missing_data.head(45)
total = total[total > 0]
(fig, ax) = plt.subplots(figsize=(10, 6))
ax.grid()
ax.bar(total.index, total.values, zorder=2, color='#3f72af')
ax.set_ylabel('No. of missing values', labelpad=10)
ax.set_xlim(-0.6, len(total) - 0.4)
ax.xaxis.set_tick_params(rotation=90)
_input1 = _input1.drop(missing_data[missing_data.Total > 0].index, axis=1)
_input0 = _input0.dropna(axis=1)
_input0 = _input0.drop(['Electrical'], axis=1)
full_dataset = pd.concat([_input1, _input0])
full_dataset = pd.get_dummies(full_dataset)
X = full_dataset.iloc[_input1.index]
X_test = full_dataset.iloc[_input0.index]
X = X.drop(['SalePrice'], axis=1)
X.shape
y = _input1.SalePrice
y.shape
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, train_size=0.8, random_state=42)
X.isna().sum().sort_values(ascending=False)
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
regressor = LinearRegression()