import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_sub = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
df_sub.head()
df_train.head()
df_test.head()
pid = df_test['PassengerId']
df_train = df_train.drop(columns=['PassengerId', 'Name'], axis=1)
df_test = df_test.drop(columns=['PassengerId', 'Name'], axis=1)
df_train.head()
df_train.isna().sum()
df_test.isna().sum()
print(df_train['CryoSleep'].value_counts())
print(df_train['CryoSleep'].mode())
print(df_train['VIP'].value_counts())
print(df_train['VIP'].mode())
print(df_train['HomePlanet'].value_counts())
print(df_train['HomePlanet'].mode())
print(df_train['Destination'].value_counts())
print(df_train['Destination'].mode())

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
df_train = clean_cat(df_train)
df_train = clean_num(df_train)
df_test = clean_cat(df_test)
df_test = clean_num(df_test)
df_train.head()

def cabin(cabin):
    try:
        return cabin.split('/')[0]
    except:
        return np.NaN
df_train['Cabin'].fillna('F', inplace=True)
df_test['Cabin'].fillna('F', inplace=True)
df_train[df_train['Cabin'] == 'F']
df_train['Cabin'] = df_train['Cabin'].apply(lambda x: cabin(x))
df_test['Cabin'] = df_test['Cabin'].apply(lambda x: cabin(x))
df_train['Cabin'] = df_train['Cabin'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'T': 6, 'U': 7})
df_test['Cabin'] = df_test['Cabin'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'T': 6, 'U': 7})
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
y_train = df_train['Transported']
x_train = df_train.drop(['Transported'], axis=1)
x_test = df_test
x_train['Cabin'] = x_train['Cabin'].fillna(5.0)
x_test['Cabin'] = x_test['Cabin'].fillna(5.0)