import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
features = ['HomePlanet', 'CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_dummied = pd.get_dummies(_input1[features], columns=['HomePlanet', 'CryoSleep', 'VIP'])
test_dummied = pd.get_dummies(_input0[features], columns=['HomePlanet', 'CryoSleep', 'VIP'])
my_imputer = SimpleImputer()
train_imputed = pd.DataFrame(my_imputer.fit_transform(train_dummied))
train_imputed.columns = train_dummied.columns
test_imputed = pd.DataFrame(my_imputer.fit_transform(test_dummied))
test_imputed.columns = test_dummied.columns
y = _input1.Transported.astype(int)
(train_X, val_X, train_y, val_y) = train_test_split(train_imputed, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1)