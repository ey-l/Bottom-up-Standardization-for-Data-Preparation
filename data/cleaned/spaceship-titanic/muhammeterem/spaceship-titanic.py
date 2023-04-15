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
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
test_data.head()
train_data.pop('Name')
test_data.pop('Name')
encoder = LabelEncoder()
scaler = StandardScaler()
train_data['Cabin'] = train_data['Cabin'].fillna('B/0/S')
train_data['Cabin_Side'] = train_data['Cabin'].str.split('/').str[2]
train_data['Cabin_Deck'] = train_data['Cabin'].str.split('/').str[0]
train_data['HomePlanet'] = encoder.fit_transform(train_data['HomePlanet'])
train_data['CryoSleep'] = encoder.fit_transform(train_data['CryoSleep'])
train_data['Destination'] = encoder.fit_transform(train_data['Destination'])
train_data['VIP'] = encoder.fit_transform(train_data['VIP'])
train_data['Cabin_Side'] = encoder.fit_transform(train_data['Cabin_Side'])
train_data['Cabin_Deck'] = encoder.fit_transform(train_data['Cabin_Deck'])
train_data = train_data.drop('Cabin', axis=1)
food_court_median_value = train_data['FoodCourt'].median()
train_data['FoodCourt'] = train_data['FoodCourt'].fillna(food_court_median_value)
room_service_median_value = train_data['RoomService'].median()
train_data['RoomService'] = train_data['RoomService'].fillna(room_service_median_value)
shopping_mall_median_value = train_data['ShoppingMall'].median()
train_data['ShoppingMall'] = train_data['ShoppingMall'].fillna(shopping_mall_median_value)
spa_median_value = train_data['Spa'].median()
train_data['Spa'] = train_data['Spa'].fillna(spa_median_value)
vrdeck_median_value = train_data['VRDeck'].median()
train_data['VRDeck'] = train_data['VRDeck'].fillna(vrdeck_median_value)
cabindeck_median_value = train_data['Cabin_Deck'].median()
train_data['Cabin_Deck'] = train_data['Cabin_Deck'].fillna(cabindeck_median_value)
cabinside_median_value = train_data['Cabin_Side'].median()
train_data['Cabin_Side'] = train_data['Cabin_Side'].fillna(cabinside_median_value)
age_median_value = train_data['Age'].median()
train_data['Age'] = train_data['Age'].fillna(age_median_value)
train_data.info()
train_data['Age'] = scaler.fit_transform(train_data[['Age']])
train_data['RoomService'] = scaler.fit_transform(train_data[['RoomService']])
train_data['FoodCourt'] = scaler.fit_transform(train_data[['FoodCourt']])
train_data['ShoppingMall'] = scaler.fit_transform(train_data[['ShoppingMall']])
train_data['VRDeck'] = scaler.fit_transform(train_data[['VRDeck']])
train_data['Spa'] = scaler.fit_transform(train_data[['Spa']])
train_data
test_data.info()
test_data['Cabin'] = test_data['Cabin'].fillna('B/0/S')
test_data['Cabin_Side'] = test_data['Cabin'].str.split('/').str[2]
test_data['Cabin_Deck'] = test_data['Cabin'].str.split('/').str[0]
test_data['HomePlanet'] = encoder.fit_transform(test_data['HomePlanet'])
test_data['CryoSleep'] = encoder.fit_transform(test_data['CryoSleep'])
test_data['Destination'] = encoder.fit_transform(test_data['Destination'])
test_data['VIP'] = encoder.fit_transform(test_data['VIP'])
test_data['Cabin_Side'] = encoder.fit_transform(test_data['Cabin_Side'])
test_data['Cabin_Deck'] = encoder.fit_transform(test_data['Cabin_Deck'])
test_data = test_data.drop('Cabin', axis=1)
food_court_median_value = test_data['FoodCourt'].median()
test_data['FoodCourt'] = test_data['FoodCourt'].fillna(food_court_median_value)
room_service_median_value = test_data['RoomService'].median()
test_data['RoomService'] = test_data['RoomService'].fillna(room_service_median_value)
shopping_mall_median_value = test_data['ShoppingMall'].median()
test_data['ShoppingMall'] = test_data['ShoppingMall'].fillna(shopping_mall_median_value)
spa_median_value = test_data['Spa'].median()
test_data['Spa'] = test_data['Spa'].fillna(spa_median_value)
vrdeck_median_value = test_data['VRDeck'].median()
test_data['VRDeck'] = test_data['VRDeck'].fillna(vrdeck_median_value)
cabindeck_median_value = test_data['Cabin_Deck'].median()
test_data['Cabin_Deck'] = test_data['Cabin_Deck'].fillna(cabindeck_median_value)
cabinside_median_value = test_data['Cabin_Side'].median()
test_data['Cabin_Side'] = test_data['Cabin_Side'].fillna(cabinside_median_value)
age_median_value = test_data['Age'].median()
test_data['Age'] = test_data['Age'].fillna(age_median_value)
test_data['Age'] = scaler.fit_transform(test_data[['Age']])
test_data['RoomService'] = scaler.fit_transform(test_data[['RoomService']])
test_data['FoodCourt'] = scaler.fit_transform(test_data[['FoodCourt']])
test_data['ShoppingMall'] = scaler.fit_transform(test_data[['ShoppingMall']])
test_data['VRDeck'] = scaler.fit_transform(test_data[['VRDeck']])
test_data['Spa'] = scaler.fit_transform(test_data[['Spa']])
train_data
sns.barplot(x=train_data['HomePlanet'], y=train_data['Transported'])
sns.barplot(x=train_data['CryoSleep'], y=train_data['Transported'])
sns.barplot(x=train_data['Destination'], y=train_data['Transported'])
sns.barplot(x=train_data['VIP'], y=train_data['Transported'])
sns.barplot(x=train_data['Cabin_Side'], y=train_data['Transported'])
sns.barplot(x=train_data['Cabin_Deck'], y=train_data['Transported'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_important_features(model, X_train):
    feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
    return feature_importances
X = train_data.drop('Transported', axis=1)
y = train_data['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=1200, max_depth=15, random_state=1)