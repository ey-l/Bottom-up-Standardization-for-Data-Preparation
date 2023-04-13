import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier
SEED = 31415
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1.head()
_input1.info()
_input1.Transported = _input1.Transported.replace({True: 1, False: 0})
_input1.describe()
_input1.isna().sum()
_input0.isna().sum()
_input1['isTrain'] = 'Yes'
_input0['isTrain'] = 'No'
data = pd.concat([_input1.drop('Transported', axis=1), _input0])
data.head()
data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
data.Age.median()
data.Age = data.Age.fillna(27, inplace=False)
data[['CabinDeck', 'CabinNum', 'CabinSide']] = data['Cabin'].str.split('/', expand=True)
data['Services'] = data['RoomService'] + data['FoodCourt'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa']
data.head()
data.dtypes
categorical = data.select_dtypes('object').columns.to_list()
categorical
numerical = [column for column in data.columns if column not in categorical]
numerical
frquent_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')