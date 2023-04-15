import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train['dataversion'] = 'train'
test['dataversion'] = 'test'
full = pd.concat([train, test])
full
train.info()
test.info()
full.describe()
not_billed_data = full[(full['RoomService'] == 0) & (full['Spa'] == 0) & (full['FoodCourt'] == 0) & (full['ShoppingMall'] == 0) & (full['VRDeck'] == 0)]
mean_age = not_billed_data['Age'].mean().round()
print('There are {0} passengers on the board of the spaceship who have not spent money, their average age is {1}'.format(not_billed_data.shape[0], mean_age))
full.describe(include='object')
train.shape
test.shape
train['HomePlanet'].value_counts()
train['Destination'].value_counts()
train['Transported'].value_counts()
train.isna().sum()
test.isna().sum()
train.isna().sum() / train.shape[0]
test.isna().sum() / test.shape[0]
full.isna().value_counts().head(20)
import missingno as msno
msno.matrix(full)
test.isna().value_counts().head(20)
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