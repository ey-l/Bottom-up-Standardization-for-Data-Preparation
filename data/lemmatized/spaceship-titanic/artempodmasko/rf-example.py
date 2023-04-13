import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input0[_input0['Cabin'] == 'G/1496/S']
_input0[_input0['PassengerId'].map(lambda elem: elem.split('_')[0]) == '9266']
_input1['duplicated_id'] = _input1['PassengerId'].map(lambda elem: elem.split('_')[0]).duplicated(keep=False)
_input0['duplicated_id'] = _input0['PassengerId'].map(lambda elem: elem.split('_')[0]).duplicated(keep=False)
_input1['Cabin'] = _input1['Cabin'].fillna('empty/empty/empty', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('empty/empty/empty', inplace=False)
_input1 = pd.merge(left=_input1, right=_input1['Cabin'].str.split('/', expand=True).rename(columns={0: 'cabin_first', 1: 'cabin_second', 2: 'cabin_third'}), left_index=True, right_index=True)
_input0 = pd.merge(left=_input0, right=_input0['Cabin'].str.split('/', expand=True).rename(columns={0: 'cabin_first', 1: 'cabin_second', 2: 'cabin_third'}), left_index=True, right_index=True)
_input1['cabin_second'] = _input1['cabin_second'].replace({'empty': -1}, inplace=False)
_input0['cabin_second'] = _input0['cabin_second'].replace({'empty': -1}, inplace=False)
from sklearn.preprocessing import OneHotEncoder
categorical_columns = ['HomePlanet', 'cabin_first', 'cabin_third', 'Destination']
encoder = OneHotEncoder(drop='first', sparse=False)