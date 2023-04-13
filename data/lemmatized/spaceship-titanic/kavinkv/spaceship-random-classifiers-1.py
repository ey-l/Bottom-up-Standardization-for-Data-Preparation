import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.info()
_input1 = _input1.drop(_input1.iloc[:, 12:13], axis=1, inplace=False)
_input1
_input1 = _input1.drop(_input1.iloc[:, 0:2], axis=1, inplace=False)
_input1
_input1 = _input1.drop(_input1.iloc[:, 1:3], axis=1, inplace=False)
_input1
_input1 = _input1.replace(to_replace=False, value=0)
_input1 = _input1.replace(to_replace=True, value=1)
_input1
_input1.isnull().sum()
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mean(), inplace=False)
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mean(), inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(_input1['RoomService'].mean(), inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(_input1['FoodCourt'].mean(), inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(_input1['ShoppingMall'].mean(), inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(_input1['Spa'].mean(), inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(_input1['VRDeck'].mean(), inplace=False)
_input1
x = _input1.iloc[:, 0:8]
y = _input1.Transported
x
std = x.iloc[:, 3:]
std
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()