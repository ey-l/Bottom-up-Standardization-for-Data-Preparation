import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input1.shape
_input1.columns
_input1['Sex'].value_counts()
_input1['Died'] = 1 - _input1['Survived']
_input1.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(10, 5), stacked=True)
figure = plt.figure(figsize=(16, 7))
plt.hist([_input1[_input1['Survived'] == 1]['Fare'], _input1[_input1['Survived'] == 0]['Fare']], stacked=True, bins=50, label=['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
df1 = _input1.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Died'], axis=1)
df1.head(10)
df1.isnull().sum()
df1.Sex = df1.Sex.map({'female': 0, 'male': 1})
df1.Embarked = df1.Embarked.map({'S': 0, 'C': 1, 'Q': 2, 'nan': 'NaN'})
df1.head()
mean_age_men = df1[df1['Sex'] == 1]['Age'].mean()
mean_age_women = df1[df1['Sex'] == 0]['Age'].mean()
df1.loc[df1.Age.isnull() & (df1['Sex'] == 0), 'Age'] = mean_age_women
df1.loc[df1.Age.isnull() & (df1['Sex'] == 1), 'Age'] = mean_age_men
df1.isnull().sum()
df1 = df1.dropna(inplace=False)
df1.isnull().sum()
df1.Age = (df1.Age - min(df1.Age)) / (max(df1.Age) - min(df1.Age))
df1.Fare = (df1.Fare - min(df1.Fare)) / (max(df1.Fare) - min(df1.Fare))
df1.describe()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(df1.drop(['Survived'], axis=1), df1.Survived, test_size=0.2, random_state=0, stratify=df1.Survived)
from sklearn.linear_model import LogisticRegression
lrmod = LogisticRegression()