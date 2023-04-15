
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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.isnull().sum().plot(kind='bar')
train.apply(lambda x: x.isnull().sum(), axis=1).value_counts().plot(kind='bar')
train['Name'] = train['Name'].fillna('JOHN DOE')
test['Name'] = test['Name'].fillna('JOHN DOE')
train.loc[train['CryoSleep'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0.0
train.loc[train['CryoSleep'] == True, 'VIP'] = False
test.loc[test['CryoSleep'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0.0
test.loc[test['CryoSleep'] == True, 'VIP'] = False
train.loc[(train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) == 0.0) & train['CryoSleep'].isna(), 'CryoSleep'] = True
test.loc[(test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) == 0.0) & test['CryoSleep'].isna(), 'CryoSleep'] = True
train['CryoSleep'] = train['CryoSleep'].fillna(False)
test['CryoSleep'] = test['CryoSleep'].fillna(False)
train['isna'] = train.apply(lambda x: x.isnull().sum(), axis=1)
test['isna'] = test.apply(lambda x: x.isnull().sum(), axis=1)
test['Transported'] = np.nan
full_data = pd.concat([train, test])
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