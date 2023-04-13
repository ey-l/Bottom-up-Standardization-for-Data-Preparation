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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(5)
_input1 = _input1.drop(['Name', 'Cabin', 'CryoSleep', 'Destination', 'HomePlanet'], axis='columns')
_input1.head(5)
_input1.isna().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean())
_input1.isna().sum()
_input1 = _input1.dropna()
_input1.shape
_input1['VIP'] = _input1['VIP'].apply(lambda x: 1 if x == True else 0)
_input1.head(5)
target = _input1['Transported'].apply(lambda x: 1 if x == True else 0)
_input1 = _input1.drop(['Transported'], axis='columns')
_input1 = _input1.drop(['PassengerId'], axis='columns')
_input1.head(5)
from sklearn.svm import SVC
model1 = SVC()