import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.info()
_input1.isnull().sum()
_input1['HomePlanet'].unique()
_input1['HomePlanet'].value_counts()
_input1['VIP'].value_counts()
_input1['VIP'].unique()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
_input1['PassengerId'] = l1.fit_transform(_input1['PassengerId'])
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input1['HomePlanet'] = l1.fit_transform(_input1['HomePlanet'])
_input1['CryoSleep'] = _input1['CryoSleep'].astype(str)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('False')
_input1['CryoSleep'] = l1.fit_transform(_input1['CryoSleep'])
_input1['Cabin'] = _input1['Cabin'].fillna('A/0/S')
_input1['Cabin'] = l1.fit_transform(_input1['Cabin'])
_input1['Destination'] = _input1['Destination'].fillna('PSO J318.5-22')
_input1['Destination'] = l1.fit_transform(_input1['Destination'])
ab = _input1['Age'].mean()
ab = round(ab)
_input1['Age'] = _input1['Age'].fillna(ab)
_input1['VIP'] = _input1['VIP'].astype(str)
_input1['VIP'] = _input1['VIP'].fillna('False')
_input1['VIP'] = l1.fit_transform(_input1['VIP'])
ab1 = _input1['RoomService'].mean()
ab1 = round(ab1)
_input1['RoomService'] = _input1['RoomService'].fillna(ab1)
ab2 = _input1['FoodCourt'].mean()
ab2 = round(ab2)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(ab2)
ab3 = _input1['ShoppingMall'].mean()
ab3 = round(ab3)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(ab3)
ab4 = _input1['Spa'].mean()
ab4 = round(ab4)
_input1['Spa'] = _input1['Spa'].fillna(ab4)
ab5 = _input1['VRDeck'].mean()
ab5 = round(ab5)
_input1['VRDeck'] = _input1['VRDeck'].fillna(ab5)
_input1['Name'] = _input1['Name'].fillna('Juanna Vines')
_input1['Name'] = l1.fit_transform(_input1['Name'])
_input1['Transported'] = _input1['Transported'].astype(str)
_input1['Transported'] = l1.fit_transform(_input1['Transported'])
_input1.info()
_input1.isnull().sum()
_input1
_input1.corr()
import seaborn as sns
sns.heatmap(data=_input1.corr(), annot=True)
import matplotlib.pyplot as plt
plt.hist(_input1['Age'], bins=20, rwidth=0.8)
plt.xlabel('Age')
plt.ylabel('Count')
_input1.head()
ab6 = _input1['FoodCourt'].mean()
ab6 = round(ab6)
_input1['FoodCourt'] = _input1['FoodCourt'].replace(0.0, ab6)
_input1['FoodCourt'].value_counts()
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
plt.hist(_input1['Age'], bins=20, rwidth=0.8, density=True)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
rng = np.arange(_input1['Age'].min(), _input1['Age'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, _input1['Age'].mean(), _input1['Age'].std()))
_input1['Age'].value_counts()
plt.hist(_input1['FoodCourt'], bins=20, rwidth=0.8, density=True)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
rng = np.arange(_input1['FoodCourt'].min(), _input1['FoodCourt'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, _input1['FoodCourt'].mean(), _input1['FoodCourt'].std()))
_input1['FoodCourt'].value_counts()
_input1
x = _input1.iloc[:, :-1].values
y = _input1.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x = std.fit_transform(x)
x1 = x.mean()
x1 = round(x1)
x1
x2 = x.var()
x2 = round(x2)
x2
x
_input1['Transported'].value_counts()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=11)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()