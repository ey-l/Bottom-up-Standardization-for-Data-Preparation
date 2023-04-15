import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
train_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_dataset = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_dataset.head()
train_dataset.info()

train_dataset.hist(bins=50, figsize=(16, 16))

(fig, ax) = plt.subplots(figsize=(10, 6))
ax.grid()
ax.scatter(train_dataset['GrLivArea'], train_dataset['SalePrice'], c='#3f72af', zorder=3, alpha=0.9)
ax.axvline(4500, c='#112d4e', ls='--', zorder=2)
ax.set_xlabel('Ground living area (sq. ft)', labelpad=10)
ax.set_ylabel('Sale price ($)', labelpad=10)

sns.boxplot(train_dataset.GrLivArea)

numerical_df = train_dataset.select_dtypes(exclude=['object'])
numerical_df = numerical_df.drop(['Id'], axis=1)
for column in numerical_df:
    plt.figure(figsize=(16, 4))
    sns.set_theme(style='whitegrid')
    sns.boxplot(numerical_df[column])
train_dataset.get('SalePrice').describe()
(f, ax) = plt.subplots(figsize=(16, 16))
sns.distplot(train_dataset.get('SalePrice'), kde=False)

corrmat = train_dataset.corr()
(f, ax) = plt.subplots(figsize=(16, 16))
sns.heatmap(corrmat, vmax=0.8, square=True)

plt.figure(figsize=(16, 16))
columns = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
correlation_matrix = np.corrcoef(train_dataset[columns].values.T)
sns.set(font_scale=1.25)
heat_map = sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values, xticklabels=columns.values)

train_dataset = train_dataset[train_dataset.GrLivArea < 4500]
total = test_dataset.isna().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total'])
missing_data.head(45)
total = total[total > 0]
(fig, ax) = plt.subplots(figsize=(10, 6))
ax.grid()
ax.bar(total.index, total.values, zorder=2, color='#3f72af')
ax.set_ylabel('No. of missing values', labelpad=10)
ax.set_xlim(-0.6, len(total) - 0.4)
ax.xaxis.set_tick_params(rotation=90)

train_dataset = train_dataset.drop(missing_data[missing_data.Total > 0].index, axis=1)
test_dataset = test_dataset.dropna(axis=1)
test_dataset = test_dataset.drop(['Electrical'], axis=1)
full_dataset = pd.concat([train_dataset, test_dataset])
full_dataset = pd.get_dummies(full_dataset)
X = full_dataset.iloc[train_dataset.index]
X_test = full_dataset.iloc[test_dataset.index]
X = X.drop(['SalePrice'], axis=1)
X.shape
y = train_dataset.SalePrice
y.shape
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, train_size=0.8, random_state=42)
X.isna().sum().sort_values(ascending=False)
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
regressor = LinearRegression()