import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1 = _input1.replace({'HomePlanet': {'Earth': 0, 'Europa': 1, 'Mars': 2}, 'Destination': {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}}, inplace=False)
_input1
_input1.isnull().sum()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode()[0], inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode()[0], inplace=False)
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0], inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode()[0], inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mode()[0], inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mode()[0], inplace=False)
df = _input1.drop(columns=['Cabin', 'RoomService', 'FoodCourt', 'Spa', 'Name', 'PassengerId'])
df.isnull().sum()
df['Destination'].value_counts()
x = df.drop(columns=['Transported'])
y = df['Transported']
(X_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
y_test
model_2 = LogisticRegression()