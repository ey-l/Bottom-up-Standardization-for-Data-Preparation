from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize': (11, 8)})
sns.set_style('whitegrid')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.isnull().sum().plot(kind='bar')
_input1.apply(lambda x: x.isnull().sum(), axis=1).value_counts().plot(kind='bar')
_input1['Name'] = _input1['Name'].fillna('JOHN DOE')
_input0['Name'] = _input0['Name'].fillna('JOHN DOE')
_input1.loc[_input1['CryoSleep'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0.0
_input1.loc[_input1['CryoSleep'] == True, 'VIP'] = False
_input0.loc[_input0['CryoSleep'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0.0
_input0.loc[_input0['CryoSleep'] == True, 'VIP'] = False
_input1.loc[(_input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) == 0.0) & _input1['CryoSleep'].isna(), 'CryoSleep'] = True
_input0.loc[(_input0[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) == 0.0) & _input0['CryoSleep'].isna(), 'CryoSleep'] = True
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input1['isna'] = _input1.apply(lambda x: x.isnull().sum(), axis=1)
_input0['isna'] = _input0.apply(lambda x: x.isnull().sum(), axis=1)
_input0['Transported'] = np.nan
full_data = pd.concat([_input1, _input0])
full_data_transported = full_data['Transported']
full_data = full_data.drop('Transported', axis=1)
wmissed = full_data[full_data['isna'] == 0].drop('isna', axis=1).copy(deep=True)
wmissed = wmissed.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
cat = {'HomePlanet', 'Destination', 'VIP', 'Transported'}
cat_missed = {'HomePlanet', 'Destination', 'VIP'}
num_missed = {'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'}
missed = cat_missed.union(num_missed)

def imputation(column, to_predict):
    is_cat = column in cat
    params = {'loss_function': 'MultiClass' if is_cat else 'RMSE', 'verbose': False, 'n_estimators': 1000, 'learning_rate': 0.001, 'random_state': 42}
    if is_cat:
        model = CatBoostClassifier(**params)
    else:
        model = CatBoostRegressor(**params)
    X_train = pd.get_dummies(wmissed.drop(column, axis=1))
    y_train = wmissed[column]