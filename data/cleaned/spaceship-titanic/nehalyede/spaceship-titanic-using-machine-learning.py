import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataset
dataset.info()
dataset.isnull().sum()
dataset['HomePlanet'].unique()
dataset['HomePlanet'].value_counts()
dataset['VIP'].value_counts()
dataset['VIP'].unique()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
dataset['PassengerId'] = l1.fit_transform(dataset['PassengerId'])
dataset['HomePlanet'] = dataset['HomePlanet'].fillna('Earth')
dataset['HomePlanet'] = l1.fit_transform(dataset['HomePlanet'])
dataset['CryoSleep'] = dataset['CryoSleep'].astype(str)
dataset['CryoSleep'] = dataset['CryoSleep'].fillna('False')
dataset['CryoSleep'] = l1.fit_transform(dataset['CryoSleep'])
dataset['Cabin'] = dataset['Cabin'].fillna('A/0/S')
dataset['Cabin'] = l1.fit_transform(dataset['Cabin'])
dataset['Destination'] = dataset['Destination'].fillna('PSO J318.5-22')
dataset['Destination'] = l1.fit_transform(dataset['Destination'])
ab = dataset['Age'].mean()
ab = round(ab)
dataset['Age'] = dataset['Age'].fillna(ab)
dataset['VIP'] = dataset['VIP'].astype(str)
dataset['VIP'] = dataset['VIP'].fillna('False')
dataset['VIP'] = l1.fit_transform(dataset['VIP'])
ab1 = dataset['RoomService'].mean()
ab1 = round(ab1)
dataset['RoomService'] = dataset['RoomService'].fillna(ab1)
ab2 = dataset['FoodCourt'].mean()
ab2 = round(ab2)
dataset['FoodCourt'] = dataset['FoodCourt'].fillna(ab2)
ab3 = dataset['ShoppingMall'].mean()
ab3 = round(ab3)
dataset['ShoppingMall'] = dataset['ShoppingMall'].fillna(ab3)
ab4 = dataset['Spa'].mean()
ab4 = round(ab4)
dataset['Spa'] = dataset['Spa'].fillna(ab4)
ab5 = dataset['VRDeck'].mean()
ab5 = round(ab5)
dataset['VRDeck'] = dataset['VRDeck'].fillna(ab5)
dataset['Name'] = dataset['Name'].fillna('Juanna Vines')
dataset['Name'] = l1.fit_transform(dataset['Name'])
dataset['Transported'] = dataset['Transported'].astype(str)
dataset['Transported'] = l1.fit_transform(dataset['Transported'])
dataset.info()
dataset.isnull().sum()
dataset
dataset.corr()
import seaborn as sns
sns.heatmap(data=dataset.corr(), annot=True)
import matplotlib.pyplot as plt
plt.hist(dataset['Age'], bins=20, rwidth=0.8)
plt.xlabel('Age')
plt.ylabel('Count')

dataset.head()
ab6 = dataset['FoodCourt'].mean()
ab6 = round(ab6)
dataset['FoodCourt'] = dataset['FoodCourt'].replace(0.0, ab6)
dataset['FoodCourt'].value_counts()
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
plt.hist(dataset['Age'], bins=20, rwidth=0.8, density=True)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
rng = np.arange(dataset['Age'].min(), dataset['Age'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, dataset['Age'].mean(), dataset['Age'].std()))
dataset['Age'].value_counts()
plt.hist(dataset['FoodCourt'], bins=20, rwidth=0.8, density=True)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
rng = np.arange(dataset['FoodCourt'].min(), dataset['FoodCourt'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, dataset['FoodCourt'].mean(), dataset['FoodCourt'].std()))
dataset['FoodCourt'].value_counts()
dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
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
dataset['Transported'].value_counts()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=11)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()