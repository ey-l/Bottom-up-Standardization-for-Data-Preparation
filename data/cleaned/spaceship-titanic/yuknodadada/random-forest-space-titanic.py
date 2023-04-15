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
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data2 = test_data
test_data.head()
test_data.info()
train_data = train_data.drop(['Name'], axis=1)
train_data.head()
nans = train_data.isna().sum().sort_values(ascending=False)
missing_data = pd.concat([nans], axis=1, keys=['Total'])
missing_data
train_data['CryoSleep'] = train_data['CryoSleep'].fillna('other')
train_data['VIP'] = train_data['VIP'].fillna('other')
train_data['HomePlanet'] = train_data['HomePlanet'].fillna('other')
train_data['Cabin'] = train_data['Cabin'].fillna('other')
train_data['Destination'] = train_data['Destination'].fillna('other')
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='mean')