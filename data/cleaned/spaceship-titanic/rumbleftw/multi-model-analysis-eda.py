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
dataTrain = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataTest = pd.read_csv('data/input/spaceship-titanic/test.csv')
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=dataTrain.isnull(), cbar=False, cmap='viridis')
sns.heatmap(ax=axes[1], yticklabels=False, data=dataTest.isnull(), cbar=False, cmap='tab20c')
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')

print('Unique HomePlanet:', dataTrain.HomePlanet.unique(), '\nUnique Destination:', dataTrain.Destination.unique())
plt.figure(figsize=(12, 8))
data = dataTrain.corr()['Transported'].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    labels.append(indices[i])
    corr.append(data[i])
sns.barplot(x=corr, y=labels, palette='magma')
plt.title('Correlation coefficient between different features and Transported')
plt.figure(figsize=(18, 9))
sns.heatmap(dataTrain.corr(), cmap='YlGnBu', annot=True)

tPlanet = pd.crosstab(dataTrain['Transported'], dataTrain['HomePlanet'])
tDest = pd.crosstab(dataTrain['Transported'], dataTrain['Destination'])
plt.figure(figsize=(12, 8))
colors = sns.color_palette('pastel')
plt.pie([item / len(dataTrain.HomePlanet) for item in dataTrain.HomePlanet.value_counts()], labels=['Earth', 'Europa', 'Mars'], colors=colors, autopct='%.0f%%')

plt.figure(figsize=(20, 15))
plt.subplot(2, 2, 1)
sns.countplot(x=dataTrain.HomePlanet, hue=dataTrain.Transported, palette='viridis')
plt.title('Transported individuals - Home Planets', fontsize=15)
plt.xlabel('HomePlanet', fontsize=15)
plt.ylabel('Number of Individuals', fontsize=15)
plt.subplot(2, 2, 2)
sns.countplot(x=dataTrain.HomePlanet, hue=dataTrain.CryoSleep, palette='viridis')
plt.title('Transported individuals - Cryosleep', fontsize=14)
plt.xlabel('HomePlanet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)
plt.figure(figsize=(20, 8))
sns.histplot(dataTrain.Age, color=sns.color_palette('magma')[2])

trainAge = dataTrain.copy()
testAge = dataTest.copy()
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

idCol = dataTest.PassengerId.to_numpy()
dataTrain.set_index('PassengerId', inplace=True)
dataTest.set_index('PassengerId', inplace=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataTrain = pd.DataFrame(imputer.fit_transform(dataTrain), columns=dataTrain.columns, index=dataTrain.index)
dataTest = pd.DataFrame(imputer.fit_transform(dataTest), columns=dataTest.columns, index=dataTest.index)
dataTrain = dataTrain.reset_index(drop=True)
dataTest = dataTest.reset_index(drop=True)
dataTrain.Transported = dataTrain.Transported.astype('int')
dataTrain.VIP = dataTrain.VIP.astype('int')
dataTrain.CryoSleep = dataTrain.CryoSleep.astype('int')
dataTrain.drop(columns=['Cabin', 'Name'], inplace=True)
dataTest.drop(columns=['Cabin', 'Name'], inplace=True)
dataTrain.head()
dataTrain = pd.get_dummies(dataTrain, columns=['HomePlanet', 'CryoSleep', 'Destination'])
dataTest = pd.get_dummies(dataTest, columns=['HomePlanet', 'CryoSleep', 'Destination'])
dataTrain.head()
yTrain = dataTrain.pop('Transported').to_numpy()
xTrain = dataTrain.to_numpy()
xTest = dataTest.to_numpy()
(xTrain.shape, yTrain.shape, xTest.shape)
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(3)