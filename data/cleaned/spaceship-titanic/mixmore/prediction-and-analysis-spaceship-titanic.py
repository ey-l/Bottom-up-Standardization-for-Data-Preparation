import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train_data.head()
test_data.head()
sample_submission.head()
train_data.shape
train_data.info()
train_data.columns
train_data.tail()
print(train_data.HomePlanet.unique())
print(train_data.Cabin.unique())
print(train_data.Destination.unique())
train_data.nunique()

print('TRAIN SET MISSING VALUES:')
print(train_data.isna().sum())
print('')
print('TEST SET MISSING VALUES:')
print(test_data.isna().sum())
print(f'Duplicates in train set: {train_data.duplicated().sum()}, ({np.round(100 * train_data.duplicated().sum() / len(train_data), 1)}%)')
print('')
print(f'Duplicates in test set: {test_data.duplicated().sum()}, ({np.round(100 * test_data.duplicated().sum() / len(test_data), 1)}%)')
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=train_data.isnull(), cbar=False, cmap='viridis')
sns.heatmap(ax=axes[1], yticklabels=False, data=test_data.isnull(), cbar=False, cmap='tab20c')
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')

import missingno
missingno.matrix(train_data)
missingno.matrix(test_data)
data_df = pd.concat([train_data, test_data], axis=0)
data_df
idCol = test_data.PassengerId.to_numpy()
train_data.set_index('PassengerId', inplace=True)
test_data.set_index('PassengerId', inplace=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
test_data = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns, index=test_data.index)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
missingno.matrix(train_data)
missingno.matrix(test_data)
plt.figure(figsize=(17, 8))
plt.plot(train_data.isna().sum() / len(train_data) * 100)
plt.plot(test_data.isna().sum() / len(train_data) * 100)
plt.legend(['train data', 'test data'])
plt.xlabel('Columns')
plt.ylabel('% of Null')

plt.figure(figsize=(6, 6))
train_data['Transported'].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True, textprops={'fontsize': 16}).set_title('Target distribution')
plt.figure(figsize=(10, 4))
sns.histplot(data=train_data, x='Age', hue='Transported', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)')
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
fig = plt.figure(figsize=(10, 20))
for (i, var_name) in enumerate(exp_feats):
    ax = fig.add_subplot(5, 2, 2 * i + 1)
    sns.histplot(data=train_data, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
    ax.set_title(var_name)
    ax = fig.add_subplot(5, 2, 2 * i + 2)
    sns.histplot(data=train_data, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0, 100])
    ax.set_title(var_name)
fig.tight_layout()

cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
fig = plt.figure(figsize=(10, 16))
for (i, var_name) in enumerate(cat_feats):
    ax = fig.add_subplot(4, 1, i + 1)
    sns.countplot(data=train_data, x=var_name, axes=ax, hue='Transported')
    ax.set_title(var_name)
fig.tight_layout()

train_data['Expenditure'] = train_data[exp_feats].sum(axis=1)
train_data['No_spending'] = (train_data['Expenditure'] == 0).astype(int)
test_data['Expenditure'] = test_data[exp_feats].sum(axis=1)
test_data['No_spending'] = (test_data['Expenditure'] == 0).astype(int)
fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=train_data, x='Expenditure', hue='Transported', bins=200)
plt.title('Total expenditure (truncated)')
plt.ylim([0, 200])
plt.xlim([0, 20000])
plt.subplot(1, 2, 2)
sns.countplot(data=train_data, x='No_spending', hue='Transported')
plt.title('No spending indicator')
fig.tight_layout()
train_data.Transported = train_data.Transported.astype('int')
train_data.VIP = train_data.VIP.astype('int')
train_data.CryoSleep = train_data.CryoSleep.astype('int')
train_data.drop(columns=['Cabin', 'Name'], inplace=True)
test_data.drop(columns=['Cabin', 'Name'], inplace=True)
train_data.head()
train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'CryoSleep', 'Destination'])
test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'CryoSleep', 'Destination'])
train_data.head()
y_Train = train_data.pop('Transported').to_numpy()
x_Train = train_data.to_numpy()
x_Test = test_data.to_numpy()
y_Test = test_data.to_numpy()
(x_Train.shape, y_Train.shape, x_Test.shape, y_Test.shape)
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(3)