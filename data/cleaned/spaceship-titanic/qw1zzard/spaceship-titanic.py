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
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
train_data.head()
train_data.info()
train_data.Transported = train_data.Transported.replace({True: 1, False: 0})
train_data.describe()
train_data.isna().sum()
test.isna().sum()
train_data['isTrain'] = 'Yes'
test['isTrain'] = 'No'
data = pd.concat([train_data.drop('Transported', axis=1), test])
data.head()
data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
data.Age.median()
data.Age.fillna(27, inplace=True)
data[['CabinDeck', 'CabinNum', 'CabinSide']] = data['Cabin'].str.split('/', expand=True)
data['Services'] = data['RoomService'] + data['FoodCourt'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa']
data.head()
data.dtypes
categorical = data.select_dtypes('object').columns.to_list()
categorical
numerical = [column for column in data.columns if column not in categorical]
numerical
frquent_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')