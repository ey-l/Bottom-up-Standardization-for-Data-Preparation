import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_theme()
sns.set_style('ticks')
sns.despine()

DATA_DIR = 'data/input/spaceship-titanic'

def get_data_path(filename):
    return os.path.join(DATA_DIR, filename)
filepath = get_data_path('train.csv')
df = pd.read_csv(filepath)
df.head()
df.info()
df['PassengerId'].head()
split_id = df['PassengerId'].str.split('_', expand=True)
split_id.head()
df['GroupId'] = split_id[0]
df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
df.head()
split_cabin = df['Cabin'].str.split('/', expand=True)
split_cabin.head()
df['CabinDeck'] = split_cabin[0]
df.head()
df['CabinId'] = split_cabin[0] + split_cabin[1]
df.head()
df['CabinSide'] = split_cabin[2]
df.head()
expenditure_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalExpense'] = df[expenditure_cols].sum(axis=1)
df.head()
transported = df['Transported']
df = df.drop('Transported', axis=1)
df['Transported'] = transported
df.head()
df['HomePlanet'].value_counts()
df['CryoSleep'].value_counts()
df['Destination'].value_counts()
df['VIP'].value_counts()
df['GroupId'].value_counts()
df['GroupSize'].value_counts()
df['CabinDeck'].value_counts()
df['CabinId'].value_counts()
df['CabinSide'].value_counts()
df['Transported'].value_counts()
df['Age'].isna().sum() / len(df) * 100
age_df = df[df['Age'].isna() == True]
age_df.head()
age_df['VIP'].value_counts()
age_df['TotalExpense'].describe()
sns.histplot(x=age_df['TotalExpense'])
age_df[age_df['TotalExpense'] == 22261]
names = df[['PassengerId', 'Name', 'VIP']]
names[names['Name'].str.contains('Unhaftimle') == True]
names[names['PassengerId'].str.contains('6348')]
age_df[age_df['VIP'] == True]
df['Age'].describe()
df[df['Age'] >= 65].head()
len(df[df['Age'] >= 65]) / len(df) * 100
df[df['Age'] == 0].head()
df[df['Age'] == 0]['GroupSize'].value_counts()
g = df[df['GroupSize'] == 3].groupby('GroupId').filter(lambda x: x['Age'].eq(0).any())
g = g.groupby('GroupId')
g.groups
g.get_group('0067')
sns.kdeplot(data=df, x='Age', fill=True)
sns.kdeplot(data=df, x='Age', hue='Transported', fill=True)
sns.histplot(data=df, x='Age', hue='Transported')
sns.countplot(data=df[df['Age'].isna()], x='Transported')
expenditure_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpense']
df[expenditure_cols].isna().sum() / len(df) * 100
df[expenditure_cols].describe()
df[df['TotalExpense'] > 35000]
df[df['Spa'] == df['Spa'].max()]
(fig, axes) = plt.subplots(3, 2, figsize=(10, 10))
for (name, ax) in zip(expenditure_cols, axes.flatten()):
    sns.kdeplot(data=df, x=name, fill=True, ax=ax)
plt.tight_layout()
(fig, axes) = plt.subplots(3, 2, figsize=(10, 10))
for (name, ax) in zip(expenditure_cols, axes.flatten()):
    sns.kdeplot(data=df, x=name, fill=True, hue='Transported', ax=ax)
plt.tight_layout()
false_std = df[df['Transported'] == False]['TotalExpense'].std()
true_std = df[df['Transported'] == True]['TotalExpense'].std()
(false_std, true_std)
(fig, axes) = plt.subplots(2, 3, figsize=(13, 10))
for (name, ax) in zip(expenditure_cols, axes.flatten()):
    if name == 'TotalExpense':
        continue
    sns.countplot(data=df[df[name].isna()], x='Transported', ax=ax)
    ax.set_title(name)
fig.delaxes(axes.flatten()[-1])
plt.tight_layout()
df[df['Spa'].isna()]['Transported'].value_counts()
df[df['VRDeck'].isna()]['Transported'].value_counts()
df['CabinDeck'].isna().sum() / len(df) * 100
df['CabinDeck'].describe()
sns.catplot(data=df, x='CabinDeck', hue='Transported', kind='count')
df['GroupSize'] = df['GroupSize'].astype('category')
df['GroupSize'].isna().sum() / len(df) * 100
df['GroupSize'].describe()
sns.catplot(data=df, x='GroupSize', hue='Transported', kind='count')
df['CryoSleep'].isna().sum() / len(df) * 100
df[df['CryoSleep'].isna()]['VIP'].value_counts()
df['CryoSleep'].describe()
df.groupby('CryoSleep')['TotalExpense'].describe()
sns.catplot(data=df, x='CryoSleep', kind='count', hue='Transported')
len(df[df['CryoSleep'] & ~df['Transported']])
sns.countplot(data=df[df['CryoSleep'].isna()], x='Transported')
df[df['CryoSleep'].isna()]['Transported'].value_counts()
df['HomePlanet'].isna().sum() / len(df) * 100
df['HomePlanet'].describe()
sns.catplot(data=df, x='HomePlanet', hue='Transported', kind='count')
sns.countplot(data=df[df['HomePlanet'].isna()], x='Transported')
df[df['HomePlanet'].isna()]['Transported'].value_counts()
df['Destination'].isna().sum() / len(df) * 100
df['Destination'].describe()
sns.catplot(data=df, x='Destination', hue='Transported', kind='count')
sns.countplot(data=df[df['Destination'].isna()], x='Transported')
df['CabinSide'].isna().sum() / len(df) * 100
df['CabinSide'].describe()
sns.catplot(data=df, x='CabinSide', hue='Transported', kind='count')
sns.countplot(data=df[df['CabinSide'].isna()], x='Transported')
df[df['CabinSide'].isna()]['Transported'].value_counts()
df['VIP'].isna().sum() / len(df) * 100
df['VIP'].describe()
sns.catplot(data=df, x='VIP', hue='Transported', kind='count')
sns.countplot(data=df[df['VIP'].isna()], x='Transported')
df[df['VIP'].isna()]['Transported'].value_counts()
df['GroupId'].isna().sum() / len(df) * 100
df['GroupId'].astype('category').describe()
df['CabinId'].isna().sum() / len(df) * 100
df['CabinId'].describe()
df['CabinOccupancy'] = df.groupby('CabinId')['CabinId'].transform('count')
df['CabinOccupancy'] = df['CabinOccupancy'].astype('category')
df['CabinOccupancy'].value_counts()
sns.catplot(data=df, x='CabinOccupancy', hue='Transported', kind='count')
corr = df.drop('Transported', axis=1).corr()
mask = np.zeros_like(corr.values)
mask[np.triu_indices_from(mask)] = True
(f, ax) = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, mask=mask, annot=True, linewidth=0.5, square=True)
correlated = ['FoodCourt', 'Spa', 'VRDeck']
(fig, axes) = plt.subplots(1, 3, figsize=(15, 5))
for (name, ax) in zip(correlated, axes.flatten()):
    sns.scatterplot(data=df, x='TotalExpense', y=name, hue='Transported', ax=ax)
plt.tight_layout()
(fig, axes) = plt.subplots(3, 2, figsize=(13, 10))
for (name, ax) in zip(expenditure_cols, axes.flatten()):
    sns.scatterplot(data=df, x='Age', y=name, hue='Transported', ax=ax)
plt.tight_layout()
(f, ax) = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='VIP', y='Age', hue='Transported')
(f, ax) = plt.subplots(figsize=(10, 7))
sns.boxplot(data=df, x='GroupSize', y='Age', hue='Transported')
(f, ax) = plt.subplots(figsize=(10, 7))
sns.boxplot(data=df, x='CabinOccupancy', y='Age', hue='Transported')
df[df['CabinOccupancy'] == 11].groupby('Transported')['Age'].describe()
(f, ax) = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='CabinDeck', y='Age', hue='Transported')
(f, ax) = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x='CryoSleep', y='Age', hue='Transported')
(f, ax) = plt.subplots(figsize=(7, 7))
sns.boxplot(data=df, x='CabinDeck', y='TotalExpense', hue='Transported')
order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
g = sns.FacetGrid(data=df[df['CabinDeck'].notna()], col='CabinDeck', hue='Transported', col_wrap=4, col_order=order)
g.map(sns.kdeplot, 'TotalExpense', fill=True)
g.add_legend()
df.groupby('CabinDeck').agg({'TotalExpense': ['mean', 'median']})
(fig, axes) = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
for (name, ax) in zip(expenditure_cols, axes.flatten()):
    sns.boxplot(data=df, x='Destination', y=name, hue='Transported', ax=ax)
plt.tight_layout()
(fig, axes) = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
for (name, ax) in zip(expenditure_cols, axes.flatten()):
    sns.boxplot(data=df, x='HomePlanet', y=name, hue='Transported', ax=ax)
plt.tight_layout()