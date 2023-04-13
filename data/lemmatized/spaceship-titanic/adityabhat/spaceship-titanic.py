import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.shape
_input0.shape
_input1.head()
type(_input1)
_input1.index
_input1
_input1.loc[0:4, ['Cabin', 'Age']]
_input1.iloc[0:4, 0:14]
_input1.loc[:, 'Transported'].value_counts(normalize=True)
_input1.shape[0] == len(_input1)
_input1.groupby(['Transported', 'VIP'])['PassengerId'].count()
_input1.groupby('Transported')['Age'].median()
age = _input1.loc[:, 'Age']
type(age)
age.sum()
age.median()
type(_input1.loc[:, ['Age']])
sns.histplot(_input1, x='Age', bins=30, kde=True, color='g')
sns.boxplot(data=_input1, y='Age')
_input1.Age.describe()
sns.histplot(data=_input1, x='Age', hue='Transported', stat='frequency', bins=10, palette='bright', element='step')
_input1.Age.isnull().sum()
_input1.head()
_input1.loc[:, ['lux_spending']] = _input1.RoomService + _input1.FoodCourt + _input1.ShoppingMall + _input1.Spa + _input1.VRDeck
sns.relplot(x='Age', y='lux_spending', data=_input1)
sns.histplot(data=_input1, x='lux_spending', hue='Transported', stat='percent', bins=10, palette='bright', element='step')
_input1.HomePlanet.value_counts(normalize=True)
sns.countplot(x='HomePlanet', hue='Transported', data=_input1)
sns.pointplot(x='HomePlanet', y='Transported', data=_input1)
sns.pointplot(x='HomePlanet', y='Age', data=_input1)
pd.crosstab(index=_input1.HomePlanet, columns=_input1.Transported, values=_input1.PassengerId, aggfunc='count', margins=True, normalize=True)