import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
(r1, c1) = _input1.shape
print('The training data has {} rows and {} columns'.format(r1, c1))
(r2, c2) = _input0.shape
print('The validation data has {} rows and {} columns'.format(r2, c2))
print('MISSING VALUES IN TRAINING DATASET:')
print(_input1.isna().sum().nlargest(c1))
print('')
print('MISSING VALUES IN VALIDATION DATASET:')
print(_input0.isna().sum().nlargest(c2))
_input1 = _input1.set_index('PassengerId', inplace=False)
_input0 = _input0.set_index('PassengerId', inplace=False)
_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = _input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].median())
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].median())
_input1['VIP'] = _input1['VIP'].fillna(False)
_input0['VIP'] = _input0['VIP'].fillna(False)
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Mars')
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Mars')
_input1['Destination'] = _input1['Destination'].fillna('PSO J318.5-22')
_input0['Destination'] = _input0['Destination'].fillna('PSO J318.5-22')
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1['Cabin'] = _input1['Cabin'].fillna('T/0/P')
_input0['Cabin'] = _input0['Cabin'].fillna('T/0/P')
plt.figure(figsize=(15, 18))
sns.heatmap(_input1.corr(), annot=True)
_input1[['Deck', 'Num', 'Side']] = _input1.Cabin.str.split('/', expand=True)
_input0[['Deck', 'Num', 'Side']] = _input0.Cabin.str.split('/', expand=True)
_input1['total_spent'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['total_spent'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1['AgeGroup'] = 0
for i in range(6):
    _input1.loc[(_input1.Age >= 10 * i) & (_input1.Age < 10 * (i + 1)), 'AgeGroup'] = i
_input0['AgeGroup'] = 0
for i in range(6):
    _input0.loc[(_input0.Age >= 10 * i) & (_input0.Age < 10 * (i + 1)), 'AgeGroup'] = i
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Num']
for i in categorical_cols:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate((_input1[i], _input0[i])).astype(str)