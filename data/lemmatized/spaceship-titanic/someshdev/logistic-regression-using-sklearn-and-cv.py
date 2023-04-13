import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input1.isna().sum()
_input1 = _input1.drop(['Name'], axis=1)
_input0 = _input0.drop(['Name'], axis=1)
space_num = _input1.select_dtypes(include=np.number)
space_num.columns
import seaborn as sns
import warnings
for x in space_num.columns:
    plt.hist(space_num[x], bins=100)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input0['Age'] = _input0['Age'].fillna(_input1['Age'].median())
temp = _input1[_input1['CryoSleep'] == True]
temp[temp['RoomService'] > 0]
_input1['RoomService'] = _input1.apply(lambda row: 0 if row['CryoSleep'] == True else row['RoomService'], axis=1)
_input1['FoodCourt'] = _input1.apply(lambda row: 0 if row['CryoSleep'] == True else row['FoodCourt'], axis=1)
_input1['ShoppingMall'] = _input1.apply(lambda row: 0 if row['CryoSleep'] == True else row['ShoppingMall'], axis=1)
_input1['Spa'] = _input1.apply(lambda row: 0 if row['CryoSleep'] == True else row['Spa'], axis=1)
_input1['VRDeck'] = _input1.apply(lambda row: 0 if row['CryoSleep'] == True else row['VRDeck'], axis=1)
_input1.isna().sum()
_input1.describe()
_input1['RoomService'] = _input1['RoomService'].fillna(0)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0)
_input1['Spa'] = _input1['Spa'].fillna(0)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0)
_input0['RoomService'] = _input0['RoomService'].fillna(0)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(0)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(0)
_input0['Spa'] = _input0['Spa'].fillna(0)
_input0['VRDeck'] = _input0['VRDeck'].fillna(0)
_input1.isna().sum()
_input1['HomePlanet'].value_counts()
space_temp = _input1[_input1['HomePlanet'].isna() == False]
space_temp.head(10)
_input1['id_1'] = _input1['PassengerId'].str[:4]
_input1['id_2'] = _input1['PassengerId'].str[5:7]
space_temp = _input1[_input1['HomePlanet'].isna() == False]
space_temp.head()
space_dict = _input1[_input1['HomePlanet'].isna() == False]
key = space_dict['id_1']
value = space_dict['HomePlanet']
final_dict = dict(zip(key, value))
freq_dict = dict(_input1['id_1'].value_counts())

def method(id):
    if freq_dict[id] > 1:
        if id in final_dict:
            return final_dict[id]
_input1['HomePlanet'] = _input1.apply(lambda row: method(row['id_1']) if pd.isnull(row['HomePlanet']) == True else row['HomePlanet'], axis=1)
_input1['HomePlanet'].value_counts()
_input1.info()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input1.info()
_input0['id_1'] = _input0['PassengerId'].str[:4]
_input0['id_2'] = _input0['PassengerId'].str[5:7]
space_temp = _input0[_input0['HomePlanet'].isna() == False]
space_dict = _input0[_input0['HomePlanet'].isna() == False]
key = space_dict['id_1']
value = space_dict['HomePlanet']
final_dict = dict(zip(key, value))
freq_dict = dict(_input0['id_1'].value_counts())
_input0['HomePlanet'] = _input0.apply(lambda row: method(row['id_1']) if pd.isnull(row['HomePlanet']) == True else row['HomePlanet'], axis=1)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input0.info()
_input1 = pd.get_dummies(_input1, prefix=['HomePlanet'], columns=['HomePlanet'], drop_first=True)
_input0 = pd.get_dummies(_input0, prefix=['HomePlanet'], columns=['HomePlanet'], drop_first=True)
_input1.head()
space_trial = _input1[_input1['CryoSleep'] == False]
space_trial[space_trial['RoomService'] == 0]
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1['CryoSleep'] = _input1['CryoSleep'].replace([True, False], [1, 0])
_input0['CryoSleep'] = _input0['CryoSleep'].replace([True, False], [1, 0])
_input1['Cabin'].value_counts()
_input1['cabin_deck'] = _input1['Cabin'].str[:1]
_input1['cabin_side'] = _input1['Cabin'].str[-1]
_input1['cabin_deck'] = _input1['cabin_deck'].fillna('U')
_input1['cabin_side'] = _input1['cabin_side'].fillna('U')
_input1['cabin_missing'] = np.where(_input1['cabin_side'] == 'U', 1, 0)
_input0['cabin_deck'] = _input0['Cabin'].str[:1]
_input0['cabin_side'] = _input0['Cabin'].str[-1]
_input0['cabin_deck'] = _input0['cabin_deck'].fillna('U')
_input0['cabin_side'] = _input0['cabin_side'].fillna('U')
_input0['cabin_missing'] = np.where(_input0['cabin_side'] == 'U', 1, 0)
import category_encoders as ce
features = ['cabin_deck', 'cabin_side']
count_encoder = ce.CountEncoder(cols=features)