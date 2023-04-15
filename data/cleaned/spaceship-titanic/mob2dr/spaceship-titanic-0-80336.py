import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import Pool, CatBoostClassifier, cv
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df.info()
train_df['Cabin'].value_counts()
train_df[['deck', 'num', 'side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['deck', 'num', 'side']] = test_df['Cabin'].str.split('/', expand=True)
train_df.drop(['Cabin'], axis=1, inplace=True)
test_df.drop(['Cabin'], axis=1, inplace=True)
test_df['deck'].value_counts()
train_df['deck'].replace('T', 'F', inplace=True)
test_df['deck'].replace('T', 'F', inplace=True)
train_df['TravelGroup'] = train_df['PassengerId'].str.split('_', expand=True)[0]
train_df['TravelGroupPos'] = train_df['PassengerId'].str.split('_', expand=True)[1]
test_df['TravelGroup'] = test_df['PassengerId'].str.split('_', expand=True)[0]
test_df['TravelGroupPos'] = test_df['PassengerId'].str.split('_', expand=True)[1]
col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_df['SumSpends'] = train_df[col_to_sum].sum(axis=1)
test_df['SumSpends'] = test_df[col_to_sum].sum(axis=1)
object_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' or train_df[col].dtype == 'category']
object_cols
object_cols.remove('num')
object_cols.remove('Name')
object_cols.remove('PassengerId')
object_cols.remove('TravelGroup')
object_cols.remove('TravelGroupPos')
for col in object_cols:
    le = LabelEncoder()