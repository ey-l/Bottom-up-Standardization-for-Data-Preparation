import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
df.head()
test.head(5)
data.head(2)
for i in data.columns:
    print({i: data[i].unique()})
data = data.replace({False: 0, True: 1, 'Europa': 0, 'Earth': 1, 'Mars': 2})
test = test.replace({False: 0, True: 1, 'Europa': 0, 'Earth': 1, 'Mars': 2})
data.head()
data.columns
data.drop(columns=['PassengerId', 'Cabin', 'Destination', 'Name'], axis=1, inplace=True)
t = test['PassengerId']
test.drop(columns=['PassengerId', 'Cabin', 'Destination', 'Name'], axis=1, inplace=True)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
data.head()
test.head()
data = data.fillna(3)
y = data['Transported']
x = data.drop(columns=['Transported'], axis=1)
x.head()
test.head()
y = data['Transported']