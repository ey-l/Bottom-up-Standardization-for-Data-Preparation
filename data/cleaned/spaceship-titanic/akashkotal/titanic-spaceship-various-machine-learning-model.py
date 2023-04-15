import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
Train = pd.read_csv('data/input/spaceship-titanic/train.csv')
Test = pd.read_csv('data/input/spaceship-titanic/test.csv')
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=Train.isnull(), cbar=False, cmap='viridis')
sns.heatmap(ax=axes[1], yticklabels=False, data=Test.isnull(), cbar=False, cmap='tab20c')
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')

print('Unique HomePlanet:', Train.HomePlanet.unique(), '\nUnique Destination:', Train.Destination.unique())
plt.figure(figsize=(12, 8))
data = Train.corr()['Transported'].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    labels.append(indices[i])
    corr.append(data[i])
sns.barplot(x=corr, y=labels, palette='viridis')
plt.title('Correlation coefficient between different features and Transported')
tPlanet = pd.crosstab(Train['Transported'], Train['HomePlanet'])
tDest = pd.crosstab(Train['Transported'], Train['Destination'])
plt.figure(figsize=(12, 8))
colors = sns.color_palette('pastel')
plt.pie([item / len(Train.HomePlanet) for item in Train.HomePlanet.value_counts()], labels=['Earth', 'Europa', 'Mars'], colors=colors, autopct='%.0f%%')

plt.figure(figsize=(20, 15))
plt.subplot(2, 2, 1)
sns.countplot(x=Train.HomePlanet, hue=Train.Transported, palette='viridis')
plt.title('Transported individuals - Home Planets', fontsize=15)
plt.xlabel('HomePlanet', fontsize=15)
plt.ylabel('Number of Individuals', fontsize=15)
plt.subplot(2, 2, 2)
sns.countplot(x=Train.HomePlanet, hue=Train.CryoSleep, palette='viridis')
plt.title('Transported individuals - Cryosleep', fontsize=14)
plt.xlabel('HomePlanet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)
plt.figure(figsize=(20, 8))
sns.histplot(Train.Age, color=sns.color_palette('magma')[2])

Test.head()
idCol = Test.PassengerId.to_numpy()
Train.set_index('PassengerId', inplace=True)
Test.set_index('PassengerId', inplace=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
Train = pd.DataFrame(imputer.fit_transform(Train), columns=Train.columns, index=Train.index)
Test = pd.DataFrame(imputer.fit_transform(Test), columns=Test.columns, index=Test.index)
Train = Train.reset_index(drop=True)
Test = Test.reset_index(drop=True)
Train.Transported = Train.Transported.astype('int')
Train.VIP = Train.VIP.astype('int')
Train.CryoSleep = Train.CryoSleep.astype('int')
Train.drop(columns=['Cabin', 'Name'], inplace=True)
Test.drop(columns=['Cabin', 'Name'], inplace=True)
Train.head()
Train = pd.get_dummies(Train, columns=['HomePlanet', 'CryoSleep', 'Destination'])
Test = pd.get_dummies(Test, columns=['HomePlanet', 'CryoSleep', 'Destination'])
Train.head()
y_Train = Train.pop('Transported').to_numpy()
x_Train = Train.to_numpy()
x_Test = Test.to_numpy()
y_Test = Test.to_numpy()
(x_Train.shape, y_Train.shape, x_Test.shape, y_Test.shape)
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(3)