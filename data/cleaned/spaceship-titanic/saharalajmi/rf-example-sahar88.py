import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test[df_test['Cabin'] == 'G/1496/S']
df_test[df_test['PassengerId'].map(lambda elem: elem.split('_')[0]) == '9266']
df['duplicated_id'] = df['PassengerId'].map(lambda elem: elem.split('_')[0]).duplicated(keep=False)
df_test['duplicated_id'] = df_test['PassengerId'].map(lambda elem: elem.split('_')[0]).duplicated(keep=False)
df['Cabin'].fillna('empty/empty/empty', inplace=True)
df_test['Cabin'].fillna('empty/empty/empty', inplace=True)
df = pd.merge(left=df, right=df['Cabin'].str.split('/', expand=True).rename(columns={0: 'cabin_first', 1: 'cabin_second', 2: 'cabin_third'}), left_index=True, right_index=True)
df_test = pd.merge(left=df_test, right=df_test['Cabin'].str.split('/', expand=True).rename(columns={0: 'cabin_first', 1: 'cabin_second', 2: 'cabin_third'}), left_index=True, right_index=True)
df['cabin_second'].replace({'empty': -1}, inplace=True)
df_test['cabin_second'].replace({'empty': -1}, inplace=True)
from sklearn.preprocessing import OneHotEncoder
categorical_columns = ['HomePlanet', 'cabin_first', 'cabin_third', 'Destination']
encoder = OneHotEncoder(drop='first', sparse=False)