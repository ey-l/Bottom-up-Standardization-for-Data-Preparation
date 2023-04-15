import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
ss = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
df.shape
df_test.shape
df.head()
type(df)
df.index
df
df.loc[0:4, ['Cabin', 'Age']]
df.iloc[0:4, 0:14]
df.loc[:, 'Transported'].value_counts(normalize=True)
df.shape[0] == len(df)
df.groupby(['Transported', 'VIP'])['PassengerId'].count()
df.groupby('Transported')['Age'].median()
age = df.loc[:, 'Age']
type(age)
age.sum()
age.median()
type(df.loc[:, ['Age']])
sns.histplot(df, x='Age', bins=30, kde=True, color='g')
sns.boxplot(data=df, y='Age')
df.Age.describe()
sns.histplot(data=df, x='Age', hue='Transported', stat='frequency', bins=10, palette='bright', element='step')
df.Age.isnull().sum()
df.head()
df.loc[:, ['lux_spending']] = df.RoomService + df.FoodCourt + df.ShoppingMall + df.Spa + df.VRDeck
sns.relplot(x='Age', y='lux_spending', data=df)
sns.histplot(data=df, x='lux_spending', hue='Transported', stat='percent', bins=10, palette='bright', element='step')
df.HomePlanet.value_counts(normalize=True)
sns.countplot(x='HomePlanet', hue='Transported', data=df)
sns.pointplot(x='HomePlanet', y='Transported', data=df)
sns.pointplot(x='HomePlanet', y='Age', data=df)
pd.crosstab(index=df.HomePlanet, columns=df.Transported, values=df.PassengerId, aggfunc='count', margins=True, normalize=True)