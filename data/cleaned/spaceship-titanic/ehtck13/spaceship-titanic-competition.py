import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.head()
import seaborn as sns
sns.heatmap(train_data.corr())
train_data.describe()
train_data['Age'] = train_data['Age'].fillna(train_data.Age.mean())
train_data['RoomService'] = train_data['RoomService'].fillna(train_data.RoomService.median())
train_data['FoodCourt'] = train_data['FoodCourt'].fillna(train_data.FoodCourt.median())
train_data['ShoppingMall'] = train_data['ShoppingMall'].fillna(train_data.ShoppingMall.median())
train_data['Spa'] = train_data['Spa'].fillna(train_data.Spa.median())
train_data['VRDeck'] = train_data['VRDeck'].fillna(train_data.VRDeck.median())
test_data['Age'] = test_data['Age'].fillna(test_data.Age.mean())
test_data['RoomService'] = test_data['RoomService'].fillna(test_data.RoomService.median())
test_data['FoodCourt'] = test_data['FoodCourt'].fillna(test_data.FoodCourt.median())
test_data['ShoppingMall'] = test_data['ShoppingMall'].fillna(test_data.ShoppingMall.median())
test_data['Spa'] = test_data['Spa'].fillna(test_data.Spa.median())
test_data['VRDeck'] = test_data['VRDeck'].fillna(test_data.VRDeck.median())
train_data.VIP[train_data.VIP == True] = 1
train_data.VIP[train_data.VIP == False] = 0
test_data.VIP[test_data.VIP == True] = 1
test_data.VIP[test_data.VIP == False] = 0
train_data['VIP'] = train_data['VIP'].fillna(train_data.VIP.mean())
test_data['VIP'] = test_data['VIP'].fillna(test_data.VIP.mean())
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
num_columns = train_data.select_dtypes(include=np.number).columns.tolist()
num_columns
X_train_val = train_data[num_columns]
X_test = test_data[num_columns]
y_train_val = train_data['Transported'].astype(int)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
(X_train, X_val, y_train, y_val) = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=1)
PassengerID = test_data.PassengerId
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()