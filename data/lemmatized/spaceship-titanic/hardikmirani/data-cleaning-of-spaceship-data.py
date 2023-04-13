import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.describe()
_input1.info()
_input1 = _input1.drop(['Cabin', 'Name'], axis=1, inplace=False)
_input1.head()
print(_input1['HomePlanet'].unique())
print(_input1['CryoSleep'].unique())
print(_input1['Destination'].unique())
total_vasi = _input1.groupby(['HomePlanet']).count()
import matplotlib.pyplot as plt
total_vasi['PassengerId'].plot.pie(title='Rehvasi', figsize=(5, 5))
_input1.plot.scatter(x='Age', y='Spa')
_input1.columns
print('Destination = ', _input1['Destination'].unique())
_input1['VRDeck']
new_df = _input1.dropna()
new_df
new_df.info()
new_df['Total_Spend'] = new_df.iloc[:, -6:-1].sum(axis=1)
new_df = new_df.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=False)
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