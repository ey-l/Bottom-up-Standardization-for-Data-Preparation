import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.tail()
fig_ = _input1.hist(figsize=(25, 30), bins=50, color='darkcyan', edgecolor='black', xlabelsize=8, ylabelsize=8)
fig_ = _input0.hist(figsize=(25, 30), bins=50, color='red', edgecolor='black', xlabelsize=8, ylabelsize=8)
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.scatterplot(x=_input1.Id, y=_input1.SalePrice, size=_input1.SalePrice, hue=_input1.OverallCond, style=_input1.YrSold, sizes=(60, 300), palette='magma')
plt.title('Scatterplot of raw data')
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=_input1.isnull(), cbar=False, cmap='viridis')
sns.heatmap(ax=axes[1], yticklabels=False, data=_input0.isnull(), cbar=False, cmap='tab20c')
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')

def show_values(axs, orient='v', space=0.01):

    def _single(ax):
        if orient == 'v':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + p.get_height() * 0.01
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha='center')
        elif orient == 'h':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - p.get_height() * 0.5
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha='left')
    if isinstance(axs, np.ndarray):
        for (idx, ax) in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
nanTrain = {}
for column in _input1.columns[1:]:
    perc = _input1[column].isna().sum() / len(_input1[column])
    if perc >= 0.01:
        nanTrain[str(column)] = perc
nanTrain = {key: value * 100 for (key, value) in sorted(nanTrain.items(), key=lambda item: item[1], reverse=True)}
a = sns.barplot(ax=axes[0], y=list(nanTrain.keys()), x=list(nanTrain.values()), palette='coolwarm', ci=None)
plt.xlabel('NaN Values (%)')
plt.ylabel('Labels')
plt.title('NaN values in training data:')
nanTest = {}
for column in _input0.columns[1:]:
    perc = _input0[column].isna().sum() / len(_input0[column])
    if perc >= 0.01:
        nanTest[str(column)] = perc
nanTest = {key: value * 100 for (key, value) in sorted(nanTest.items(), key=lambda item: item[1], reverse=True)}
b = sns.barplot(ax=axes[1], y=list(nanTest.keys()), x=list(nanTest.values()), palette='flare', ci=None)
axes[0].set_title('Missing data in training set')
axes[1].set_title('Missing data in training set')
axes[0].set_xlabel('NaN Values (%)')
axes[0].set_ylabel('Labels')
axes[1].set_xlabel('NaN Values (%)')
axes[1].set_ylabel('Labels')
show_values(a, 'h', space=0.3)
show_values(b, 'h', space=0.3)
THRESHOLD = 0.5
data = _input1.corr()['SalePrice'].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    if data[indices[i]] > THRESHOLD:
        labels.append(indices[i])
        corr.append(data[i])
sns.barplot(x=corr, y=labels)
plt.title('Lables with correlation coefficient > Threshold (0.5)')
unnecessary = []
lab = _input1.SalePrice
idCol = _input0.Id
_input1 = _input1.drop(columns=[str(item) for item in _input1.columns[1:] if str(item) not in labels])
_input0 = _input0.drop(columns=[str(item) for item in _input0.columns[1:] if str(item) not in labels])
_input1 = _input1.drop(columns=['Id'])
_input0 = _input0.drop(columns=['Id'])
_input1 = _input1.fillna(method='bfill')
_input0 = _input0.fillna(method='bfill')
(sum(_input1.isnull().sum()), sum(_input0.isnull().sum()))
yTrain = lab
xTest = _input0.to_numpy()
xTrain = _input1.to_numpy()
(xTrain.shape, yTrain.shape, xTest.shape)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
reg = GradientBoostingRegressor(random_state=42, loss='ls', learning_rate=0.1)