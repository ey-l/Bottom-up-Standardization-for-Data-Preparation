import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input2.head()
_input1.head()
_input0.head()
pid = _input0['PassengerId']
_input1 = _input1.drop(columns=['PassengerId', 'Name'], axis=1)
_input0 = _input0.drop(columns=['PassengerId', 'Name'], axis=1)
_input1.head()
_input1.isna().sum()
_input0.isna().sum()
print(_input1['CryoSleep'].value_counts())
print(_input1['CryoSleep'].mode())
print(_input1['VIP'].value_counts())
print(_input1['VIP'].mode())
print(_input1['HomePlanet'].value_counts())
print(_input1['HomePlanet'].mode())
print(_input1['Destination'].value_counts())
print(_input1['Destination'].mode())

def clean_cat(data):
    data['HomePlanet'] = data['HomePlanet'].fillna('Earth')
    data['HomePlanet'] = data['HomePlanet'].map({'Europa': 0, 'Earth': 1, 'Mars': 2})
    data['Destination'] = data['Destination'].fillna('TRAPPIST-1e')
    data['Destination'] = data['Destination'].map({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
    data['VIP'] = data['VIP'].fillna(False).astype(int)
    data['CryoSleep'] = data['CryoSleep'].fillna(False).astype(int)
    return data

def clean_num(data):
    for col in data.columns:
        if data[col].dtype == 'float64':
            data[col] = data[col].fillna(data[col].mean())
    return data
_input1 = clean_cat(_input1)
_input1 = clean_num(_input1)
_input0 = clean_cat(_input0)
_input0 = clean_num(_input0)
_input1.head()

def cabin(cabin):
    try:
        return cabin.split('/')[0]
    except:
        return np.NaN
_input1['Cabin'] = _input1['Cabin'].fillna('F', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('F', inplace=False)
_input1[_input1['Cabin'] == 'F']
_input1['Cabin'] = _input1['Cabin'].apply(lambda x: cabin(x))
_input0['Cabin'] = _input0['Cabin'].apply(lambda x: cabin(x))
_input1['Cabin'] = _input1['Cabin'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'T': 6, 'U': 7})
_input0['Cabin'] = _input0['Cabin'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'T': 6, 'U': 7})
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
y_train = _input1['Transported']
x_train = _input1.drop(['Transported'], axis=1)
x_test = _input0
x_train['Cabin'] = x_train['Cabin'].fillna(5.0)
x_test['Cabin'] = x_test['Cabin'].fillna(5.0)