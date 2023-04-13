import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=_input1.isnull(), cbar=False, cmap='viridis')
sns.heatmap(ax=axes[1], yticklabels=False, data=_input0.isnull(), cbar=False, cmap='tab20c')
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')
print('Unique HomePlanet:', _input1.HomePlanet.unique(), '\nUnique Destination:', _input1.Destination.unique())
plt.figure(figsize=(12, 8))
data = _input1.corr()['Transported'].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    labels.append(indices[i])
    corr.append(data[i])
sns.barplot(x=corr, y=labels, palette='magma')
plt.title('Correlation coefficient between different features and Transported')
plt.figure(figsize=(18, 9))
sns.heatmap(_input1.corr(), cmap='YlGnBu', annot=True)
tPlanet = pd.crosstab(_input1['Transported'], _input1['HomePlanet'])
tDest = pd.crosstab(_input1['Transported'], _input1['Destination'])
plt.figure(figsize=(12, 8))
colors = sns.color_palette('pastel')
plt.pie([item / len(_input1.HomePlanet) for item in _input1.HomePlanet.value_counts()], labels=['Earth', 'Europa', 'Mars'], colors=colors, autopct='%.0f%%')
plt.figure(figsize=(20, 15))
plt.subplot(2, 2, 1)
sns.countplot(x=_input1.HomePlanet, hue=_input1.Transported, palette='viridis')
plt.title('Transported individuals - Home Planets', fontsize=15)
plt.xlabel('HomePlanet', fontsize=15)
plt.ylabel('Number of Individuals', fontsize=15)
plt.subplot(2, 2, 2)
sns.countplot(x=_input1.HomePlanet, hue=_input1.CryoSleep, palette='viridis')
plt.title('Transported individuals - Cryosleep', fontsize=14)
plt.xlabel('HomePlanet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)
plt.figure(figsize=(20, 8))
sns.histplot(_input1.Age, color=sns.color_palette('magma')[2])
trainAge = _input1.copy()
testAge = _input0.copy()
trainAge['type'] = 'Train'
testAge['type'] = 'Test'
ageDf = pd.concat([trainAge, testAge])
fig = px.histogram(data_frame=ageDf, x='Age', color='type', color_discrete_sequence=['#FFA500', '#87CEEB'], marginal='box', nbins=100, template='plotly_white')
fig.update_layout(title='Distribution of Age', title_x=0.5)
fig.show()
plt.figure(figsize=(10, 5))
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
df_av = AV.AutoViz('data/input/spaceship-titanic/train.csv')
plt.figure(figsize=(10, 5))
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
df_av = AV.AutoViz('data/input/spaceship-titanic/test.csv')
idCol = _input0.PassengerId.to_numpy()
_input1 = _input1.set_index('PassengerId', inplace=False)
_input0 = _input0.set_index('PassengerId', inplace=False)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
_input1 = pd.DataFrame(imputer.fit_transform(_input1), columns=_input1.columns, index=_input1.index)
_input0 = pd.DataFrame(imputer.fit_transform(_input0), columns=_input0.columns, index=_input0.index)
_input1 = _input1.reset_index(drop=True)
_input0 = _input0.reset_index(drop=True)
_input1.Transported = _input1.Transported.astype('int')
_input1.VIP = _input1.VIP.astype('int')
_input1.CryoSleep = _input1.CryoSleep.astype('int')
_input1 = _input1.drop(columns=['Cabin', 'Name'], inplace=False)
_input0 = _input0.drop(columns=['Cabin', 'Name'], inplace=False)
_input1.head()
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'CryoSleep', 'Destination'])
_input0 = pd.get_dummies(_input0, columns=['HomePlanet', 'CryoSleep', 'Destination'])
_input1.head()
yTrain = _input1.pop('Transported').to_numpy()
xTrain = _input1.to_numpy()
xTest = _input0.to_numpy()
(xTrain.shape, yTrain.shape, xTest.shape)
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(3)