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
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
(r1, c1) = df_train.shape
print('The training data has {} rows and {} columns'.format(r1, c1))
(r2, c2) = df_test.shape
print('The validation data has {} rows and {} columns'.format(r2, c2))
print('MISSING VALUES IN TRAINING DATASET:')
print(df_train.isna().sum().nlargest(c1))
print('')
print('MISSING VALUES IN VALIDATION DATASET:')
print(df_test.isna().sum().nlargest(c2))
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)
df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
df_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_train['VIP'] = df_train['VIP'].fillna(False)
df_test['VIP'] = df_test['VIP'].fillna(False)
df_train['HomePlanet'] = df_train['HomePlanet'].fillna('Mars')
df_test['HomePlanet'] = df_test['HomePlanet'].fillna('Mars')
df_train['Destination'] = df_train['Destination'].fillna('PSO J318.5-22')
df_test['Destination'] = df_test['Destination'].fillna('PSO J318.5-22')
df_train['CryoSleep'] = df_train['CryoSleep'].fillna(False)
df_test['CryoSleep'] = df_test['CryoSleep'].fillna(False)
df_train['Cabin'] = df_train['Cabin'].fillna('T/0/P')
df_test['Cabin'] = df_test['Cabin'].fillna('T/0/P')
plt.figure(figsize=(15, 18))
sns.heatmap(df_train.corr(), annot=True)
df_train[['Deck', 'Num', 'Side']] = df_train.Cabin.str.split('/', expand=True)
df_test[['Deck', 'Num', 'Side']] = df_test.Cabin.str.split('/', expand=True)
df_train['total_spent'] = df_train['RoomService'] + df_train['FoodCourt'] + df_train['ShoppingMall'] + df_train['Spa'] + df_train['VRDeck']
df_test['total_spent'] = df_test['RoomService'] + df_test['FoodCourt'] + df_test['ShoppingMall'] + df_test['Spa'] + df_test['VRDeck']
df_train['AgeGroup'] = 0
for i in range(6):
    df_train.loc[(df_train.Age >= 10 * i) & (df_train.Age < 10 * (i + 1)), 'AgeGroup'] = i
df_test['AgeGroup'] = 0
for i in range(6):
    df_test.loc[(df_test.Age >= 10 * i) & (df_test.Age < 10 * (i + 1)), 'AgeGroup'] = i
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Num']
for i in categorical_cols:
    print(i)
    le = LabelEncoder()
    arr = np.concatenate((df_train[i], df_test[i])).astype(str)