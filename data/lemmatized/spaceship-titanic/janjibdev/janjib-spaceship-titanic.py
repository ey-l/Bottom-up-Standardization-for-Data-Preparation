import pandas as pd
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
(_input1.shape, _input1.info(), _input1.describe())
'\nPassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. \nPeople in a group are often family members, but not always.\n'
_input1['Group'] = _input1['PassengerId'].apply(lambda x: int(str(x)[:4]))
_input1.head()
del _input1['PassengerId']
_input1.head()
'\nHomePlanet - The planet the passenger departed from, typically their planet of permanent residence.\n'
q = pd.Categorical(_input1['HomePlanet'])
q.describe()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
q = pd.Categorical(_input1['HomePlanet'])
q.describe()
'\nCryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. \nPassengers in cryosleep are confined to their cabins.\n'
q = pd.Categorical(_input1['CryoSleep'])
q.describe()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
q = pd.Categorical(_input1['CryoSleep'])
q.describe()
_input1['CryoSleep'] = _input1['CryoSleep'].apply(lambda x: int(x))
_input1['CryoSleep']
test = np.where(_input1['Cabin'].isnull())
len(test[0])
q = _input1['Cabin'].apply(lambda x: str(x)[0])
c = pd.Categorical(q)
c.describe()
q = np.where(q == 'n', 'T', q)
c = pd.Categorical(q)
c.describe()
_input1['Deck'] = q
_input1['Deck']
q = _input1['Cabin'].apply(lambda x: str(x)[2])
c = pd.Categorical(q)
q = np.where(q == 'n', '0', q)
c = pd.Categorical(q)
q = np.array(list(map(int, q)))
_input1['Cnum'] = q
q = _input1['Cabin'].apply(lambda x: str(x)[-1])
c = pd.Categorical(q)
q = np.where(q == 'n', 'S', q)
c = pd.Categorical(q)
_input1['Port'] = q
_input1.head()
del _input1['Cabin']
q = pd.Categorical(_input1['VIP'])
q.describe()
_input1['VIP'] = _input1['VIP'].fillna(False)
_input1['VIP'] = _input1['VIP'].apply(lambda x: int(x))
_input1['VIP']
_input1['RoomService'] = _input1['RoomService'].fillna(0)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0)
_input1['Spa'] = _input1['Spa'].fillna(0)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0)
del _input1['Name']
y = _input1['Transported']
del _input1['Transported']
_input1
y
cleaned_df = _input1
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(cleaned_df, y, random_state=42)
enc_column = ['HomePlanet', 'Destination', 'Deck', 'Port']
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(enc.fit_transform(X_train[enc_column]))
OH_cols_valid = pd.DataFrame(enc.transform(X_valid[enc_column]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
num_x_train = X_train.drop(enc_column, axis=1)
num_x_valid = X_valid.drop(enc_column, axis=1)
OH_X_train = pd.concat([num_x_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_x_valid, OH_cols_valid], axis=1)
OH_X_train
y_train
import math
mean_age = math.floor(OH_X_train['Age'].mean())
OH_X_train['Age'] = OH_X_train['Age'].fillna(mean_age)
OH_X_valid['Age'] = OH_X_valid['Age'].fillna(mean_age)
OH_X_train
from sklearn.preprocessing import StandardScaler
col_names = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
scaled_train = OH_X_train.copy()
scaled_valid = OH_X_valid.copy()
features_train = scaled_train[col_names]
features_valid = scaled_valid[col_names]