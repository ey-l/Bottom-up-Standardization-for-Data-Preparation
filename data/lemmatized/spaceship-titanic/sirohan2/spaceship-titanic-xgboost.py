import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.head()
_input1.info()
_input1.describe()
_input1['CryoSleep'].value_counts()
_input1['Transported'].value_counts()
_input1['HomePlanet'].value_counts()
_input1['VIP'].value_counts()
cb = _input1['Cabin'].value_counts()
plt.plot(range(len(cb[:50])), cb[:50])
_input1['Cabin'] = _input1['Cabin'].apply(lambda s: s if str(s) in cb[:50] else 'others')
cb
_input1['Cabin'].value_counts()
sns.countplot(x='HomePlanet', data=_input1, hue='Transported')
sns.countplot(x='CryoSleep', data=_input1, hue='Transported')
sns.countplot(x='HomePlanet', data=_input1, hue='CryoSleep')
sns.countplot(x='HomePlanet', data=_input1, hue='Destination')
sns.countplot(x='Destination', data=_input1, hue='Transported')
from sklearn.preprocessing import StandardScaler
_input1.columns
X_num = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']]
X_cat = _input1[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
y = _input1['Transported']
X_cat = pd.get_dummies(X_cat)
scaler = StandardScaler()