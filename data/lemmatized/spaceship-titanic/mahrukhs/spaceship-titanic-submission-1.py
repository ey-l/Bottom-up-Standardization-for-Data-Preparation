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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input0.head()
_input2.head()
print('Dimensions of train: {}'.format(_input1.shape))
print('Dimensions of test: {}'.format(_input0.shape))
print('Dimensions of sample submission: {}'.format(_input2.shape))
_input1.describe()
_input1.info()
num_values = ['ShoppingMall', 'Age', 'FoodCourt', 'VRDeck', 'RoomService', 'Spa']
for num in num_values:
    _input1[num] = _input1[num].fillna(0)
for num in num_values:
    _input0[num] = _input0[num].fillna(0)
_input1.isna().sum()
HomePlanet_pivot = _input1.pivot_table(index='HomePlanet', values='Transported')
HomePlanet_pivot.plot.bar()
Destination_pivot = _input1.pivot_table(index='Destination', values='Transported')
Destination_pivot.plot.bar()
CryoSleep_pivot = _input1.pivot_table(index='CryoSleep', values='Transported')
CryoSleep_pivot.plot.bar()
random_forest = RandomForestClassifier()