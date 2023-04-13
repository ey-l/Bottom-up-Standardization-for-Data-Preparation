import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.head()
_input1.dtypes
_input1.info()
_input0.info()
_input1.info()
all_df = pd.concat([_input1, _input0], sort=False)
all_df.reset_index(drop=True)
all_df['Sex'].value_counts()
all_df['Embarked'].value_counts()
all_df['Name'].value_counts()
all_df['Ticket'].value_counts()
all_df.describe(include=['O'])
all_df['Pclass'].value_counts()
all_df['SibSp'].value_counts()
all_df[all_df['SibSp'] > 4]
all_df['Parch'].value_counts()
all_df[all_df['Parch'] > 4]
all_df['Cabin'].value_counts()
all_df[all_df['Name'] == 'Kelly, Mr. James']
all_df[all_df['Name'] == 'Connolly, Miss. Kate']
all_df[all_df['Cabin'] == 'C23 C25 C27']
all_df[all_df['Cabin'] == 'G6']
all_df[all_df['Cabin'] == 'B57 B59 B63 B66']
Cabin_str = all_df['Cabin'].str[0]
Cabin_str.value_counts()
name = all_df['Name'].str.split('[, .]', expand=True)
name
all_df['Honorific'] = name[2]
all_df['Honorific'].value_counts()
name = all_df['Honorific'].copy()
name[:] = 'X'
name[all_df['Honorific'] == 'Mr'] = 'Mr'
name[all_df['Honorific'] == 'Mrs'] = 'Mrs'
name[all_df['Honorific'] == 'Miss'] = 'Miss'
name[all_df['Honorific'] == 'Master'] = 'Master'
all_df['Honorific'] = name
name
all_df.loc[(all_df['Honorific'] == 'X') & (all_df['Sex'] == 'male'), 'Honorific'] = 'Mr'
all_df.loc[(all_df['Honorific'] == 'X') & (all_df['Sex'] == 'female'), 'Honorific'] = 'Miss'
all_df['Honorific'].value_counts()
name = all_df['Name'].str.split('[, ]', expand=True)
name
name[0].value_counts()
all_df['Cabin_ini'] = all_df['Cabin'].str[0]
all_df['Cabin_ini']
all_df['Cabin_ini'] = all_df['Cabin_ini'].mask((all_df['Cabin_ini'] == 'G') | (all_df['Cabin_ini'] == 'T'), 'Another', inplace=False)
all_df = all_df.fillna(value={'Cabin_ini': 'Another'}, inplace=False)
all_df['Cabin_ini'].value_counts()
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean(), inplace=False)
all_df['Fare'] = all_df['Fare'].fillna(all_df['Fare'].mean(), inplace=False)
all_df['Pclass'] = all_df['Pclass'].fillna(3, inplace=False)
all_df['Embarked'] = all_df['Embarked'].fillna('S', inplace=False)
all_df.isna().sum()
all_df_re = all_df.drop(['Cabin', 'Name', 'Sex', 'Ticket'], axis=1)
all_df_re = pd.get_dummies(all_df_re, drop_first=True)
train_df_re = all_df_re[:len(_input1)]
test_df_re = all_df_re[len(_input1):]
train_X = train_df_re.drop(['Survived', 'PassengerId'], axis=1)
test_X = test_df_re.drop(['Survived', 'PassengerId'], axis=1)
train_Y = train_df_re['Survived']
train_df_re
from sklearn.ensemble import RandomForestClassifier
SEED = 7017
model = RandomForestClassifier(n_estimators=1500, max_depth=500, random_state=SEED)