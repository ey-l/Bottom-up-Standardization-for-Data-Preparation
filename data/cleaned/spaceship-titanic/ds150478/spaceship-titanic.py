import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_data.head()
col_list = list(train_data.columns)
col_list
train_data.describe()
train_data.isnull().mean() * 100
train_data.replace({False: 0, True: 1}, inplace=True)
train_data.Transported.value_counts()
train_data.dtypes
train_data['HomePlanet'].value_counts()
categorical_var = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
fig = plt.figure(figsize=(6, 14))
for (i, var) in enumerate(categorical_var):
    ax = fig.add_subplot(4, 1, i + 1)
    sns.countplot(data=train_data, x=var, axes=ax, hue='Transported')
    ax.set_title(var)
fig.tight_layout()

plt.figure(figsize=(10, 4))
sns.histplot(data=train_data, x='Age', hue='Transported', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)')
train_data.Age.unique()
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
fig = plt.figure(figsize=(10, 20))
for (i, var_name) in enumerate(exp_feats):
    ax = fig.add_subplot(5, 2, 2 * i + 1)
    sns.histplot(data=train_data, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0, 100])
    ax.set_title(var_name)
fig.tight_layout()

train_data.drop(['VIP'], axis=1, inplace=True)
train_data.columns
df_nan = train_data.copy()
df_nan['Nan'] = train_data[['CryoSleep', 'Cabin', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name']].isnull().apply(lambda x: all(x), axis=1)
len(df_nan[df_nan['Nan'] == False])
Q1 = train_data.quantile(0.25)
Q3 = train_data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
outliers = pd.DataFrame(train_data < Q1 - 1.5 * IQR) | (train_data > Q3 + 1.5 * IQR)
outliers.head()
outliers.replace({False: 0, True: 1}, inplace=True)
outliers['Number of outliers'] = outliers.sum(axis=1)
outliers.head()
outliers[outliers['Number of outliers'] == 5]
train_data['HomePlanet'].fillna('Earth', inplace=True)
train_data['Destination'].fillna('TRAPPIST-1e', inplace=True)
train_data['CryoSleep'].fillna(0, inplace=True)
train_data['Age'].fillna(int(np.mean(train_data['Age'])), inplace=True)
train_data['RoomService'].fillna(0, inplace=True)
train_data['FoodCourt'].fillna(0, inplace=True)
train_data['ShoppingMall'].fillna(0, inplace=True)
train_data['Spa'].fillna(0, inplace=True)
train_data['VRDeck'].fillna(0, inplace=True)
train_data['Expenditure'] = np.nan
train_data['Age group'] = np.nan
train_data['Expenditure'] = train_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
train_data['No_spending'] = (train_data['Expenditure'] == 0).astype(int)
train_data.loc[(train_data['Age'] > 0) & (train_data['Age'] <= 18), 'Age group'] = 1
train_data.loc[(train_data['Age'] > 18) & (train_data['Age'] <= 45), 'Age group'] = 2
train_data.loc[train_data['Age'] > 45, 'Age group'] = 3
train_data.head()
sns.countplot(data=train_data, x='Age group', hue='Transported')
train_data['Cabin'].fillna('Z/9999/Z', inplace=True)
train_data['Deck'] = train_data['Cabin'].apply(lambda x: x.split('/')[0])
train_data['Num'] = train_data['Cabin'].apply(lambda x: x.split('/')[1])
train_data['Side'] = train_data['Cabin'].apply(lambda x: x.split('/')[2])
train_data.drop(['Cabin'], axis=1, inplace=True)
train_data.head()
ig = plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
sns.countplot(data=train_data, x='Deck', hue='Transported', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
plt.title('Cabin deck')
plt.subplot(3, 2, 2)
sns.countplot(data=train_data, x='Side', hue='Transported')
plt.title('Cabin Side')
train_data.head(2)
sns.histplot(data=train_data, x='Age', hue='CryoSleep', kde=True)
train_data['CryoSleep'].value_counts()
train_data.isnull().mean() * 100