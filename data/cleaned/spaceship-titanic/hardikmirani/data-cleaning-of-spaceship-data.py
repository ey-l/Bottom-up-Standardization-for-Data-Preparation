import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
df.describe()
df.info()
df.drop(['Cabin', 'Name'], axis=1, inplace=True)
df.head()
print(df['HomePlanet'].unique())
print(df['CryoSleep'].unique())
print(df['Destination'].unique())
total_vasi = df.groupby(['HomePlanet']).count()
import matplotlib.pyplot as plt
total_vasi['PassengerId'].plot.pie(title='Rehvasi', figsize=(5, 5))
df.plot.scatter(x='Age', y='Spa')

df.columns
print('Destination = ', df['Destination'].unique())
df['VRDeck']
new_df = df.dropna()
new_df
new_df.info()
new_df['Total_Spend'] = new_df.iloc[:, -6:-1].sum(axis=1)
new_df.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=True)
new_df
newdf = pd.get_dummies(new_df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
print(newdf)
newdf
x = newdf.drop(['PassengerId', 'Transported'], axis=1)
y = newdf['Transported']
print(x.head())
print('-----')
print(y.head())
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state=0)