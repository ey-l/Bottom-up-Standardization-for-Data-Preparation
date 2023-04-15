import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
ss = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head()
test.head()
ss.head()
print('Dimensions of train: {}'.format(train.shape))
print('Dimensions of test: {}'.format(test.shape))
print('Dimensions of sample submission: {}'.format(ss.shape))
train.describe()
train.info()
num_values = ['ShoppingMall', 'Age', 'FoodCourt', 'VRDeck', 'RoomService', 'Spa']
for num in num_values:
    train[num] = train[num].fillna(0)
for num in num_values:
    test[num] = test[num].fillna(0)
train.isna().sum()
HomePlanet_pivot = train.pivot_table(index='HomePlanet', values='Transported')
HomePlanet_pivot.plot.bar()

Destination_pivot = train.pivot_table(index='Destination', values='Transported')
Destination_pivot.plot.bar()

CryoSleep_pivot = train.pivot_table(index='CryoSleep', values='Transported')
CryoSleep_pivot.plot.bar()

random_forest = RandomForestClassifier()