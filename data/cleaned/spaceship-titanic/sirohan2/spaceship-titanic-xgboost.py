import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
df.head()
df.info()
df.describe()
df['CryoSleep'].value_counts()
df['Transported'].value_counts()
df['HomePlanet'].value_counts()
df['VIP'].value_counts()
cb = df['Cabin'].value_counts()
plt.plot(range(len(cb[:50])), cb[:50])
df['Cabin'] = df['Cabin'].apply(lambda s: s if str(s) in cb[:50] else 'others')
cb
df['Cabin'].value_counts()
sns.countplot(x='HomePlanet', data=df, hue='Transported')
sns.countplot(x='CryoSleep', data=df, hue='Transported')
sns.countplot(x='HomePlanet', data=df, hue='CryoSleep')
sns.countplot(x='HomePlanet', data=df, hue='Destination')
sns.countplot(x='Destination', data=df, hue='Transported')
from sklearn.preprocessing import StandardScaler
df.columns
X_num = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']]
X_cat = df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']]
y = df['Transported']
X_cat = pd.get_dummies(X_cat)
scaler = StandardScaler()