import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import random
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
Y = df['Transported']
Y
X = df.drop(columns=['Name', 'Transported', 'Cabin'])
X.isnull().sum()
X['HomePlanet'] = X['HomePlanet'].replace('Europa', '0')
X['HomePlanet'] = X['HomePlanet'].replace('Earth', '1')
X['HomePlanet'] = X['HomePlanet'].replace('Mars', '2')
X.loc[X['HomePlanet'].isnull(), ['HomePlanet']] = random.randrange(3)
X['CryoSleep'] = X['CryoSleep'].replace('True', '0')
X['CryoSleep'] = X['CryoSleep'].replace('False', '1')
X.loc[X['CryoSleep'].isnull(), ['CryoSleep']] = random.randrange(2)
X['Destination'] = X['Destination'].replace('TRAPPIST-1e', '0')
X['Destination'] = X['Destination'].replace('55 Cancri e', '1')
X['Destination'] = X['Destination'].replace('PSO J318.5-22', '2')
X.loc[X['Destination'].isnull(), ['Destination']] = random.randrange(3)
mean = X.mean()
X = X.fillna(mean)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)