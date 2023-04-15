import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
data_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
from sklearn.impute import KNNImputer
data_train['HomePlanet'].fillna(method='bfill', inplace=True)
data_train['CryoSleep'].fillna(method='bfill', inplace=True)
data_train['Cabin'].fillna(method='bfill', inplace=True)
data_train['Destination'].fillna(method='bfill', inplace=True)
data_train['Age'].fillna(value=data_train['Age'].mode()[0], inplace=True)
data_train['VIP'].fillna(method='bfill', inplace=True)
data_train['RoomService'].fillna(method='bfill', inplace=True)
data_train['FoodCourt'].fillna(method='bfill', inplace=True)
data_train['ShoppingMall'].fillna(method='bfill', inplace=True)
data_train['Spa'].fillna(method='bfill', inplace=True)
data_train['VRDeck'].fillna(method='bfill', inplace=True)
data_train['Name'].fillna('null', inplace=True)
print(data_train.isnull().sum())
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
group = []
group_member = []
for i in range(data_train['PassengerId'].size):
    str_left = data_train['PassengerId'][i].split('_')[0]
    group.append(str_left)
    str_right = data_train['PassengerId'][i].split('_')[1]
    group_member.append(str_right)
group = pd.DataFrame(group, columns=['group'], dtype=np.float)
group_member = pd.DataFrame(group_member, columns=['group_member'], dtype=np.float)