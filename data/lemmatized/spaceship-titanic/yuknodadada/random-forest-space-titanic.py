import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data2 = _input0
_input0.head()
_input0.info()
_input1 = _input1.drop(['Name'], axis=1)
_input1.head()
nans = _input1.isna().sum().sort_values(ascending=False)
missing_data = pd.concat([nans], axis=1, keys=['Total'])
missing_data
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('other')
_input1['VIP'] = _input1['VIP'].fillna('other')
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('other')
_input1['Cabin'] = _input1['Cabin'].fillna('other')
_input1['Destination'] = _input1['Destination'].fillna('other')
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='mean')