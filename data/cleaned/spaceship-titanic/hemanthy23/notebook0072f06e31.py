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
space_titanic_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
space_titanic_data.head()
space_titanic_data.replace({'HomePlanet': {'Earth': 0, 'Europa': 1, 'Mars': 2}, 'Destination': {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}}, inplace=True)
space_titanic_data
space_titanic_data.isnull().sum()
space_titanic_data['HomePlanet'].fillna(space_titanic_data['HomePlanet'].mode()[0], inplace=True)
space_titanic_data['CryoSleep'].fillna(space_titanic_data['CryoSleep'].mode()[0], inplace=True)
space_titanic_data['Destination'].fillna(space_titanic_data['Destination'].mode()[0], inplace=True)
space_titanic_data['Age'].fillna(space_titanic_data['Age'].mean(), inplace=True)
space_titanic_data['VIP'].fillna(space_titanic_data['VIP'].mode()[0], inplace=True)
space_titanic_data['ShoppingMall'].fillna(space_titanic_data['ShoppingMall'].mode()[0], inplace=True)
space_titanic_data['VRDeck'].fillna(space_titanic_data['VRDeck'].mode()[0], inplace=True)
df = space_titanic_data.drop(columns=['Cabin', 'RoomService', 'FoodCourt', 'Spa', 'Name', 'PassengerId'])
df.isnull().sum()
df['Destination'].value_counts()
x = df.drop(columns=['Transported'])
y = df['Transported']
(X_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
y_test
model_2 = LogisticRegression()