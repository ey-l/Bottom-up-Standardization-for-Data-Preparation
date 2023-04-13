import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
all_df = pd.concat([_input1, _input0], axis=0).reset_index()
all_df.isnull().sum()
all_df['HomePlanet'].value_counts(dropna=False)
HomePlanet_df = all_df[['HomePlanet', 'Transported', 'PassengerId']].dropna().groupby(['HomePlanet', 'Transported']).count().unstack()
HomePlanet_df.plot.bar(stacked=True)
CryoSleep_df = all_df[['CryoSleep', 'Transported', 'PassengerId']].dropna().groupby(['CryoSleep', 'Transported']).count().unstack()
CryoSleep_df.plot.bar(stacked=True)
CryoSleep_df = all_df[['CryoSleep', 'HomePlanet', 'PassengerId']].dropna().groupby(['CryoSleep', 'HomePlanet']).count().unstack()
CryoSleep_df.plot.bar(stacked=True)
pd.DataFrame(all_df['Age'].value_counts(dropna=False).sort_index())
all_df['Age'].describe()
import matplotlib.pyplot as plt
plt.plot(all_df['Age'].value_counts().sort_index())
Age_df = all_df[['Age', 'Transported', 'PassengerId']].dropna().groupby(['Age', 'Transported']).count().unstack()
Age_df.plot.bar(stacked=True)
all_df['RoomService'].describe()
all_df['RoomService'].value_counts(dropna=False).sort_index()
all_df['RoomService'].value_counts().sort_index()[[0]]
RoomService_df = all_df[['RoomService', 'Transported', 'PassengerId']].dropna().groupby(['RoomService', 'Transported']).count().unstack()
RoomService_df
RoomService_Zero_df = RoomService_df.iloc[0]
RoomService_Zero_df.plot.bar(stacked=False)
RoomService_NotZero_df = RoomService_df.iloc[1:].sum()
RoomService_NotZero_df
RoomService_NotZero_df.plot.bar(stacked=True)
RoomService_Full_df = pd.concat([RoomService_Zero_df, RoomService_NotZero_df])
RoomService_Full_df.plot.bar(stacked=True)
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
non_df = all_df.dropna()
from sklearn.linear_model import LinearRegression
x = non_df.loc[:, num_columns]
x_Age = x.iloc[:, 1:6]
x_Age
t_Age = x[['Age']]
model_Age = LinearRegression()