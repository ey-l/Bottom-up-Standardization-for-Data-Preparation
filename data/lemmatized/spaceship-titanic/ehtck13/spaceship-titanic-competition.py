import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
import seaborn as sns
sns.heatmap(_input1.corr())
_input1.describe()
_input1['Age'] = _input1['Age'].fillna(_input1.Age.mean())
_input1['RoomService'] = _input1['RoomService'].fillna(_input1.RoomService.median())
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1.FoodCourt.median())
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1.ShoppingMall.median())
_input1['Spa'] = _input1['Spa'].fillna(_input1.Spa.median())
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1.VRDeck.median())
_input0['Age'] = _input0['Age'].fillna(_input0.Age.mean())
_input0['RoomService'] = _input0['RoomService'].fillna(_input0.RoomService.median())
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(_input0.FoodCourt.median())
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(_input0.ShoppingMall.median())
_input0['Spa'] = _input0['Spa'].fillna(_input0.Spa.median())
_input0['VRDeck'] = _input0['VRDeck'].fillna(_input0.VRDeck.median())
_input1.VIP[_input1.VIP == True] = 1
_input1.VIP[_input1.VIP == False] = 0
_input0.VIP[_input0.VIP == True] = 1
_input0.VIP[_input0.VIP == False] = 0
_input1['VIP'] = _input1['VIP'].fillna(_input1.VIP.mean())
_input0['VIP'] = _input0['VIP'].fillna(_input0.VIP.mean())
_input1 = _input1.fillna(0)
_input0 = _input0.fillna(0)
num_columns = _input1.select_dtypes(include=np.number).columns.tolist()
num_columns
X_train_val = _input1[num_columns]
X_test = _input0[num_columns]
y_train_val = _input1['Transported'].astype(int)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
(X_train, X_val, y_train, y_val) = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=1)
PassengerID = _input0.PassengerId
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()