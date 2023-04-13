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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
col_list = list(_input1.columns)
col_list
_input1.describe()
_input1.isnull().mean() * 100
_input1 = _input1.replace({False: 0, True: 1}, inplace=False)
_input1.Transported.value_counts()
_input1.dtypes
_input1['HomePlanet'].value_counts()
categorical_var = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
fig = plt.figure(figsize=(6, 14))
for (i, var) in enumerate(categorical_var):
    ax = fig.add_subplot(4, 1, i + 1)
    sns.countplot(data=_input1, x=var, axes=ax, hue='Transported')
    ax.set_title(var)
fig.tight_layout()
plt.figure(figsize=(10, 4))
sns.histplot(data=_input1, x='Age', hue='Transported', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)')
_input1.Age.unique()
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
fig = plt.figure(figsize=(10, 20))
for (i, var_name) in enumerate(exp_feats):
    ax = fig.add_subplot(5, 2, 2 * i + 1)
    sns.histplot(data=_input1, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0, 100])
    ax.set_title(var_name)
fig.tight_layout()
_input1 = _input1.drop(['VIP'], axis=1, inplace=False)
_input1.columns
df_nan = _input1.copy()
df_nan['Nan'] = _input1[['CryoSleep', 'Cabin', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name']].isnull().apply(lambda x: all(x), axis=1)
len(df_nan[df_nan['Nan'] == False])
Q1 = _input1.quantile(0.25)
Q3 = _input1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
outliers = pd.DataFrame(_input1 < Q1 - 1.5 * IQR) | (_input1 > Q3 + 1.5 * IQR)
outliers.head()
outliers = outliers.replace({False: 0, True: 1}, inplace=False)
outliers['Number of outliers'] = outliers.sum(axis=1)
outliers.head()
outliers[outliers['Number of outliers'] == 5]
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(0, inplace=False)
_input1['Age'] = _input1['Age'].fillna(int(np.mean(_input1['Age'])), inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(0, inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0, inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0, inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(0, inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0, inplace=False)
_input1['Expenditure'] = np.nan
_input1['Age group'] = np.nan
_input1['Expenditure'] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
_input1['No_spending'] = (_input1['Expenditure'] == 0).astype(int)
_input1.loc[(_input1['Age'] > 0) & (_input1['Age'] <= 18), 'Age group'] = 1
_input1.loc[(_input1['Age'] > 18) & (_input1['Age'] <= 45), 'Age group'] = 2
_input1.loc[_input1['Age'] > 45, 'Age group'] = 3
_input1.head()
sns.countplot(data=_input1, x='Age group', hue='Transported')
_input1['Cabin'] = _input1['Cabin'].fillna('Z/9999/Z', inplace=False)
_input1['Deck'] = _input1['Cabin'].apply(lambda x: x.split('/')[0])
_input1['Num'] = _input1['Cabin'].apply(lambda x: x.split('/')[1])
_input1['Side'] = _input1['Cabin'].apply(lambda x: x.split('/')[2])
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
_input1.head()
ig = plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
sns.countplot(data=_input1, x='Deck', hue='Transported', order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
plt.title('Cabin deck')
plt.subplot(3, 2, 2)
sns.countplot(data=_input1, x='Side', hue='Transported')
plt.title('Cabin Side')
_input1.head(2)
sns.histplot(data=_input1, x='Age', hue='CryoSleep', kde=True)
_input1['CryoSleep'].value_counts()
_input1.isnull().mean() * 100