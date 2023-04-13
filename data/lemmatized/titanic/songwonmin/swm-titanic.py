import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input1.info()
_input1 = _input1.set_index('PassengerId')
_input0 = _input0.set_index('PassengerId')
sns.set(rc={'figure.figsize': (10, 5)})
sns.barplot(x='Pclass', y='Survived', data=_input1)
_input1['Name'].head()
_input1['Name_2'] = _input1['Name'].str.split(',').str[1].str.split('.').str[0].str[1:]
_input0['Name_2'] = _input0['Name'].str.split(',').str[1].str.split('.').str[0].str[1:]
_input1['last_name'] = _input1['Name'].str.split(',').str[0]
_input0['last_name'] = _input0['Name'].str.split(',').str[0]
_input1['last_name'].nunique()
imsi_df = pd.concat([_input1, _input0], axis=0)
_input1['last_name_sum'] = _input1['last_name'].map(dict(imsi_df['last_name'].value_counts()))
_input0['last_name_sum'] = _input0['last_name'].map(dict(imsi_df['last_name'].value_counts()))
_input1 = _input1.drop('last_name', axis=1)
_input0 = _input0.drop('last_name', axis=1)
sns.set(rc={'figure.figsize': (20, 10)})
sns.barplot(x='Name_2', y='Survived', data=_input1)
_input1[_input1['Name_2'] == 'Rev']
_input1[_input1['Name_2'] == 'Don']
_input1[_input1['Name_2'] == 'Capt']
_input1[_input1['Name_2'] == 'Sir']
_input1[_input1['Name_2'] == 'Lady']
_input1[_input1['Name_2'] == 'Ms']
_input1[_input1['Name_2'] == 'Mme']
_input1[_input1['Name_2'] == 'Mlle']
_input1[_input1['Name_2'] == 'Jonkheer']
_input1 = _input1.drop('Name', axis=1)
_input0 = _input0.drop('Name', axis=1)
sns.set(rc={'figure.figsize': (10, 5)})
sns.barplot(x='Sex', y='Survived', data=_input1)
_input1['Age'].unique()[:50]
sns.set(rc={'figure.figsize': (10, 5)})
sns.barplot(x='SibSp', y='Survived', data=_input1)
_input1[_input1['SibSp'] >= 5].sort_values('SibSp')
_input0[_input0['SibSp'] >= 5]
_input1[(_input1['SibSp'] == 3) | (_input1['SibSp'] == 4)]['Survived'].sum() / len(_input1[(_input1['SibSp'] == 3) | (_input1['SibSp'] == 4)])
sns.set(rc={'figure.figsize': (10, 5)})
sns.barplot(x='Parch', y='Survived', data=_input1)
_input1[(_input1['Parch'] == 4) | (_input1['Parch'] == 6)]
_input1[_input1['Parch'] == 5]
_input1[_input1['Parch'] == 3]
len(_input1[_input1['Parch'] == 2])
len(_input1[_input1['Parch'] == 1])
len(_input1[_input1['Parch'] == 0])
_input1['Ticket'].unique()[:50]
_input1 = _input1.drop('Ticket', axis=1)
_input0 = _input0.drop('Ticket', axis=1)
_input1['Fare'].unique()[:50]
_input1['Cabin'] = _input1['Cabin'].str[0]
_input0['Cabin'] = _input0['Cabin'].str[0]
sns.set(rc={'figure.figsize': (10, 5)})
sns.barplot(x='Cabin', y='Survived', data=_input1)
sns.set(rc={'figure.figsize': (10, 5)})
sns.barplot(x='Embarked', y='Survived', data=_input1)
_input1.head()
sex_mapping = {'male': 0, 'female': 1}
_input1['Sex'] = _input1['Sex'].map(sex_mapping)
_input0['Sex'] = _input0['Sex'].map(sex_mapping)
_input0['Fare'] = _input0['Fare'].fillna(_input0['Fare'].mean())
_input1['Embarked'] = _input1['Embarked'].fillna(_input1['Embarked'].mode()[0])
_input1 = pd.get_dummies(_input1, columns=['SibSp', 'Parch', 'Name_2', 'Embarked'])
_input0 = pd.get_dummies(_input0, columns=['SibSp', 'Parch', 'Name_2', 'Embarked'])
_input1 = pd.get_dummies(_input1, columns=['Cabin'], dummy_na=True)
_input0 = pd.get_dummies(_input0, columns=['Cabin'], dummy_na=True)
_input1.info()
_input0.info()
_input1['Parch_9'] = 0
_input0['Name_2_Capt'] = 0
_input0['Name_2_Jonkheer'] = 0
_input0['Name_2_Lady'] = 0
_input0['Name_2_Major'] = 0
_input0['Name_2_Mlle'] = 0
_input0['Name_2_Mme'] = 0
_input0['Name_2_Sir'] = 0
_input0['Name_2_the Countess'] = 0
_input0['Cabin_T'] = 0
_input0 = _input0.rename(columns={'Name_2_Dona': 'Name_2_Don'})
len(_input1.columns)
len(_input0.columns)
age_fill_df = _input1.dropna()
X = age_fill_df.drop(['Survived', 'Age'], axis=1)
y = age_fill_df['Age']
age_fill_df = _input0.dropna()
X = pd.concat([X, age_fill_df.drop(['Age'], axis=1)], axis=0)
y = pd.concat([y, age_fill_df['Age']], axis=0)
from catboost import CatBoostRegressor
cat_cols = list(X.columns)
cat_cols.remove('Fare')
cat_cols.remove('last_name_sum')
model = CatBoostRegressor(cat_features=cat_cols)