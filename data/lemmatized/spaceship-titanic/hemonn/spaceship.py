import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1['dataversion'] = 'train'
_input0['dataversion'] = 'test'
full = pd.concat([_input1, _input0])
full
_input1.info()
_input0.info()
full.describe()
not_billed_data = full[(full['RoomService'] == 0) & (full['Spa'] == 0) & (full['FoodCourt'] == 0) & (full['ShoppingMall'] == 0) & (full['VRDeck'] == 0)]
mean_age = not_billed_data['Age'].mean().round()
print('There are {0} passengers on the board of the spaceship who have not spent money, their average age is {1}'.format(not_billed_data.shape[0], mean_age))
full.describe(include='object')
_input1.shape
_input0.shape
_input1['HomePlanet'].value_counts()
_input1['Destination'].value_counts()
_input1['Transported'].value_counts()
_input1.isna().sum()
_input0.isna().sum()
_input1.isna().sum() / _input1.shape[0]
_input0.isna().sum() / _input0.shape[0]
full.isna().value_counts().head(20)
import missingno as msno
msno.matrix(full)
_input0.isna().value_counts().head(20)
full['Groupnum'] = full['PassengerId'].str.split('_', expand=True)[0].tolist()
full['Familyname'] = full['Name'].str.split(' ', expand=True)[1].tolist()
full['Cabinnum'] = full['Cabin'].str.split('/', expand=True)[0].tolist()
full
from sklearn.preprocessing import LabelEncoder
columns = ['HomePlanet', 'Destination', 'Groupnum', 'Familyname', 'Cabinnum']
names = ['HomePlanet_label', 'Destination_label', 'Groupnum_label', 'Familyname_label', 'Cabinnum_label']
for (i, j) in zip(columns, names):
    full[j] = LabelEncoder().fit_transform(full[i]).tolist()
full
from sklearn.ensemble import RandomForestClassifier as RFC
feature = ['Destination_label', 'Groupnum_label', 'Familyname_label', 'Cabinnum_label']
label = ['HomePlanet_label']
Home_traindf = full[full['HomePlanet'].notna()]
Home_testdf = full[full['HomePlanet'].isna()]
Home_testdf