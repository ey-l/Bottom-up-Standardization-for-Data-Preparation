import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
space = pd.read_csv('data/input/spaceship-titanic/train.csv')
space_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
space.info()
space.isna().sum()
space = space.drop(['Name'], axis=1)
space_test = space_test.drop(['Name'], axis=1)
space_num = space.select_dtypes(include=np.number)
space_num.columns
import seaborn as sns
import warnings
for x in space_num.columns:
    plt.hist(space_num[x], bins=100)

space['Age'] = space['Age'].fillna(space['Age'].median())
space_test['Age'] = space_test['Age'].fillna(space['Age'].median())
temp = space[space['CryoSleep'] == True]
temp[temp['RoomService'] > 0]
space['RoomService'] = space.apply(lambda row: 0 if row['CryoSleep'] == True else row['RoomService'], axis=1)
space['FoodCourt'] = space.apply(lambda row: 0 if row['CryoSleep'] == True else row['FoodCourt'], axis=1)
space['ShoppingMall'] = space.apply(lambda row: 0 if row['CryoSleep'] == True else row['ShoppingMall'], axis=1)
space['Spa'] = space.apply(lambda row: 0 if row['CryoSleep'] == True else row['Spa'], axis=1)
space['VRDeck'] = space.apply(lambda row: 0 if row['CryoSleep'] == True else row['VRDeck'], axis=1)
space.isna().sum()
space.describe()
space['RoomService'] = space['RoomService'].fillna(0)
space['FoodCourt'] = space['FoodCourt'].fillna(0)
space['ShoppingMall'] = space['ShoppingMall'].fillna(0)
space['Spa'] = space['Spa'].fillna(0)
space['VRDeck'] = space['VRDeck'].fillna(0)
space_test['RoomService'] = space_test['RoomService'].fillna(0)
space_test['FoodCourt'] = space_test['FoodCourt'].fillna(0)
space_test['ShoppingMall'] = space_test['ShoppingMall'].fillna(0)
space_test['Spa'] = space_test['Spa'].fillna(0)
space_test['VRDeck'] = space_test['VRDeck'].fillna(0)
space.isna().sum()
space['HomePlanet'].value_counts()
space_temp = space[space['HomePlanet'].isna() == False]
space_temp.head(10)
space['id_1'] = space['PassengerId'].str[:4]
space['id_2'] = space['PassengerId'].str[5:7]
space_temp = space[space['HomePlanet'].isna() == False]
space_temp.head()
space_dict = space[space['HomePlanet'].isna() == False]
key = space_dict['id_1']
value = space_dict['HomePlanet']
final_dict = dict(zip(key, value))
freq_dict = dict(space['id_1'].value_counts())

def method(id):
    if freq_dict[id] > 1:
        if id in final_dict:
            return final_dict[id]
space['HomePlanet'] = space.apply(lambda row: method(row['id_1']) if pd.isnull(row['HomePlanet']) == True else row['HomePlanet'], axis=1)
space['HomePlanet'].value_counts()
space.info()
space['HomePlanet'] = space['HomePlanet'].fillna('Earth')
space.info()
space_test['id_1'] = space_test['PassengerId'].str[:4]
space_test['id_2'] = space_test['PassengerId'].str[5:7]
space_temp = space_test[space_test['HomePlanet'].isna() == False]
space_dict = space_test[space_test['HomePlanet'].isna() == False]
key = space_dict['id_1']
value = space_dict['HomePlanet']
final_dict = dict(zip(key, value))
freq_dict = dict(space_test['id_1'].value_counts())
space_test['HomePlanet'] = space_test.apply(lambda row: method(row['id_1']) if pd.isnull(row['HomePlanet']) == True else row['HomePlanet'], axis=1)
space_test['HomePlanet'] = space_test['HomePlanet'].fillna('Earth')
space_test.info()
space = pd.get_dummies(space, prefix=['HomePlanet'], columns=['HomePlanet'], drop_first=True)
space_test = pd.get_dummies(space_test, prefix=['HomePlanet'], columns=['HomePlanet'], drop_first=True)
space.head()
space_trial = space[space['CryoSleep'] == False]
space_trial[space_trial['RoomService'] == 0]
space['CryoSleep'] = space['CryoSleep'].fillna(False)
space_test['CryoSleep'] = space_test['CryoSleep'].fillna(False)
space['CryoSleep'] = space['CryoSleep'].replace([True, False], [1, 0])
space_test['CryoSleep'] = space_test['CryoSleep'].replace([True, False], [1, 0])
space['Cabin'].value_counts()
space['cabin_deck'] = space['Cabin'].str[:1]
space['cabin_side'] = space['Cabin'].str[-1]
space['cabin_deck'] = space['cabin_deck'].fillna('U')
space['cabin_side'] = space['cabin_side'].fillna('U')
space['cabin_missing'] = np.where(space['cabin_side'] == 'U', 1, 0)
space_test['cabin_deck'] = space_test['Cabin'].str[:1]
space_test['cabin_side'] = space_test['Cabin'].str[-1]
space_test['cabin_deck'] = space_test['cabin_deck'].fillna('U')
space_test['cabin_side'] = space_test['cabin_side'].fillna('U')
space_test['cabin_missing'] = np.where(space_test['cabin_side'] == 'U', 1, 0)
import category_encoders as ce
features = ['cabin_deck', 'cabin_side']
count_encoder = ce.CountEncoder(cols=features)