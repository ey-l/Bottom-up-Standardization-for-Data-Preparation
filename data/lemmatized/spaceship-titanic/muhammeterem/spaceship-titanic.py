import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input0.head()
_input1.pop('Name')
_input0.pop('Name')
encoder = LabelEncoder()
scaler = StandardScaler()
_input1['Cabin'] = _input1['Cabin'].fillna('B/0/S')
_input1['Cabin_Side'] = _input1['Cabin'].str.split('/').str[2]
_input1['Cabin_Deck'] = _input1['Cabin'].str.split('/').str[0]
_input1['HomePlanet'] = encoder.fit_transform(_input1['HomePlanet'])
_input1['CryoSleep'] = encoder.fit_transform(_input1['CryoSleep'])
_input1['Destination'] = encoder.fit_transform(_input1['Destination'])
_input1['VIP'] = encoder.fit_transform(_input1['VIP'])
_input1['Cabin_Side'] = encoder.fit_transform(_input1['Cabin_Side'])
_input1['Cabin_Deck'] = encoder.fit_transform(_input1['Cabin_Deck'])
_input1 = _input1.drop('Cabin', axis=1)
food_court_median_value = _input1['FoodCourt'].median()
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(food_court_median_value)
room_service_median_value = _input1['RoomService'].median()
_input1['RoomService'] = _input1['RoomService'].fillna(room_service_median_value)
shopping_mall_median_value = _input1['ShoppingMall'].median()
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(shopping_mall_median_value)
spa_median_value = _input1['Spa'].median()
_input1['Spa'] = _input1['Spa'].fillna(spa_median_value)
vrdeck_median_value = _input1['VRDeck'].median()
_input1['VRDeck'] = _input1['VRDeck'].fillna(vrdeck_median_value)
cabindeck_median_value = _input1['Cabin_Deck'].median()
_input1['Cabin_Deck'] = _input1['Cabin_Deck'].fillna(cabindeck_median_value)
cabinside_median_value = _input1['Cabin_Side'].median()
_input1['Cabin_Side'] = _input1['Cabin_Side'].fillna(cabinside_median_value)
age_median_value = _input1['Age'].median()
_input1['Age'] = _input1['Age'].fillna(age_median_value)
_input1.info()
_input1['Age'] = scaler.fit_transform(_input1[['Age']])
_input1['RoomService'] = scaler.fit_transform(_input1[['RoomService']])
_input1['FoodCourt'] = scaler.fit_transform(_input1[['FoodCourt']])
_input1['ShoppingMall'] = scaler.fit_transform(_input1[['ShoppingMall']])
_input1['VRDeck'] = scaler.fit_transform(_input1[['VRDeck']])
_input1['Spa'] = scaler.fit_transform(_input1[['Spa']])
_input1
_input0.info()
_input0['Cabin'] = _input0['Cabin'].fillna('B/0/S')
_input0['Cabin_Side'] = _input0['Cabin'].str.split('/').str[2]
_input0['Cabin_Deck'] = _input0['Cabin'].str.split('/').str[0]
_input0['HomePlanet'] = encoder.fit_transform(_input0['HomePlanet'])
_input0['CryoSleep'] = encoder.fit_transform(_input0['CryoSleep'])
_input0['Destination'] = encoder.fit_transform(_input0['Destination'])
_input0['VIP'] = encoder.fit_transform(_input0['VIP'])
_input0['Cabin_Side'] = encoder.fit_transform(_input0['Cabin_Side'])
_input0['Cabin_Deck'] = encoder.fit_transform(_input0['Cabin_Deck'])
_input0 = _input0.drop('Cabin', axis=1)
food_court_median_value = _input0['FoodCourt'].median()
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(food_court_median_value)
room_service_median_value = _input0['RoomService'].median()
_input0['RoomService'] = _input0['RoomService'].fillna(room_service_median_value)
shopping_mall_median_value = _input0['ShoppingMall'].median()
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(shopping_mall_median_value)
spa_median_value = _input0['Spa'].median()
_input0['Spa'] = _input0['Spa'].fillna(spa_median_value)
vrdeck_median_value = _input0['VRDeck'].median()
_input0['VRDeck'] = _input0['VRDeck'].fillna(vrdeck_median_value)
cabindeck_median_value = _input0['Cabin_Deck'].median()
_input0['Cabin_Deck'] = _input0['Cabin_Deck'].fillna(cabindeck_median_value)
cabinside_median_value = _input0['Cabin_Side'].median()
_input0['Cabin_Side'] = _input0['Cabin_Side'].fillna(cabinside_median_value)
age_median_value = _input0['Age'].median()
_input0['Age'] = _input0['Age'].fillna(age_median_value)
_input0['Age'] = scaler.fit_transform(_input0[['Age']])
_input0['RoomService'] = scaler.fit_transform(_input0[['RoomService']])
_input0['FoodCourt'] = scaler.fit_transform(_input0[['FoodCourt']])
_input0['ShoppingMall'] = scaler.fit_transform(_input0[['ShoppingMall']])
_input0['VRDeck'] = scaler.fit_transform(_input0[['VRDeck']])
_input0['Spa'] = scaler.fit_transform(_input0[['Spa']])
_input1
sns.barplot(x=_input1['HomePlanet'], y=_input1['Transported'])
sns.barplot(x=_input1['CryoSleep'], y=_input1['Transported'])
sns.barplot(x=_input1['Destination'], y=_input1['Transported'])
sns.barplot(x=_input1['VIP'], y=_input1['Transported'])
sns.barplot(x=_input1['Cabin_Side'], y=_input1['Transported'])
sns.barplot(x=_input1['Cabin_Deck'], y=_input1['Transported'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_important_features(model, X_train):
    feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
    return feature_importances
X = _input1.drop('Transported', axis=1)
y = _input1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=1200, max_depth=15, random_state=1)