import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as ply
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head(5)
df = df.drop(['Name', 'Cabin', 'CryoSleep', 'Destination', 'HomePlanet'], axis='columns')
df.head(5)
df.isna().sum()
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.isna().sum()
df = df.dropna()
df.shape
df['VIP'] = df['VIP'].apply(lambda x: 1 if x == True else 0)
df.head(5)
target = df['Transported'].apply(lambda x: 1 if x == True else 0)
df = df.drop(['Transported'], axis='columns')
df = df.drop(['PassengerId'], axis='columns')
df.head(5)
from sklearn.svm import SVC
model1 = SVC()