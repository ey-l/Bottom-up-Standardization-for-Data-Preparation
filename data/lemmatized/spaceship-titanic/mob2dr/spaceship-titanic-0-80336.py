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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input1['Cabin'].value_counts()
_input1[['deck', 'num', 'side']] = _input1['Cabin'].str.split('/', expand=True)
_input0[['deck', 'num', 'side']] = _input0['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
_input0 = _input0.drop(['Cabin'], axis=1, inplace=False)
_input0['deck'].value_counts()
_input1['deck'] = _input1['deck'].replace('T', 'F', inplace=False)
_input0['deck'] = _input0['deck'].replace('T', 'F', inplace=False)
_input1['TravelGroup'] = _input1['PassengerId'].str.split('_', expand=True)[0]
_input1['TravelGroupPos'] = _input1['PassengerId'].str.split('_', expand=True)[1]
_input0['TravelGroup'] = _input0['PassengerId'].str.split('_', expand=True)[0]
_input0['TravelGroupPos'] = _input0['PassengerId'].str.split('_', expand=True)[1]
col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1['SumSpends'] = _input1[col_to_sum].sum(axis=1)
_input0['SumSpends'] = _input0[col_to_sum].sum(axis=1)
object_cols = [col for col in _input1.columns if _input1[col].dtype == 'object' or _input1[col].dtype == 'category']
object_cols
object_cols.remove('num')
object_cols.remove('Name')
object_cols.remove('PassengerId')
object_cols.remove('TravelGroup')
object_cols.remove('TravelGroupPos')
for col in object_cols:
    le = LabelEncoder()