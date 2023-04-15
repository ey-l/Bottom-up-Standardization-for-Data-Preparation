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
dataTrain = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
dataTrain.head()
dataTest = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
dataTest.tail()
fig_ = dataTrain.hist(figsize=(25, 30), bins=50, color='darkcyan', edgecolor='black', xlabelsize=8, ylabelsize=8)
fig_ = dataTest.hist(figsize=(25, 30), bins=50, color='red', edgecolor='black', xlabelsize=8, ylabelsize=8)
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.scatterplot(x=dataTrain.Id, y=dataTrain.SalePrice, size=dataTrain.SalePrice, hue=dataTrain.OverallCond, style=dataTrain.YrSold, sizes=(60, 300), palette='magma')
plt.title('Scatterplot of raw data')

(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=dataTrain.isnull(), cbar=False, cmap='viridis')
sns.heatmap(ax=axes[1], yticklabels=False, data=dataTest.isnull(), cbar=False, cmap='tab20c')
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
for column in dataTrain.columns[1:]:
    perc = dataTrain[column].isna().sum() / len(dataTrain[column])
    if perc >= 0.01:
        nanTrain[str(column)] = perc
nanTrain = {key: value * 100 for (key, value) in sorted(nanTrain.items(), key=lambda item: item[1], reverse=True)}
a = sns.barplot(ax=axes[0], y=list(nanTrain.keys()), x=list(nanTrain.values()), palette='coolwarm', ci=None)
plt.xlabel('NaN Values (%)')
plt.ylabel('Labels')
plt.title('NaN values in training data:')
nanTest = {}
for column in dataTest.columns[1:]:
    perc = dataTest[column].isna().sum() / len(dataTest[column])
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
data = dataTrain.corr()['SalePrice'].sort_values(ascending=False)
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
lab = dataTrain.SalePrice
idCol = dataTest.Id
dataTrain = dataTrain.drop(columns=[str(item) for item in dataTrain.columns[1:] if str(item) not in labels])
dataTest = dataTest.drop(columns=[str(item) for item in dataTest.columns[1:] if str(item) not in labels])
dataTrain = dataTrain.drop(columns=['Id'])
dataTest = dataTest.drop(columns=['Id'])
dataTrain = dataTrain.fillna(method='bfill')
dataTest = dataTest.fillna(method='bfill')
(sum(dataTrain.isnull().sum()), sum(dataTest.isnull().sum()))
yTrain = lab
xTest = dataTest.to_numpy()
xTrain = dataTrain.to_numpy()
(xTrain.shape, yTrain.shape, xTest.shape)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
reg = GradientBoostingRegressor(random_state=42, loss='ls', learning_rate=0.1)