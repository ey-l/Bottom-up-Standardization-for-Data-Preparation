import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
numfs = [feature for feature in _input1.columns if _input1[feature].dtype == int or _input1[feature].dtype == float]
catfs = [feature for feature in _input1.columns if _input1[feature].dtype == object]
print('Numerical Features: ', numfs)
print('Categorical Features: ', catfs)
(f, axs) = plt.subplots(2, 2, figsize=(20, 10))
sns.countplot(x='HomePlanet', data=_input1, ax=axs[0][0])
sns.countplot(x='CryoSleep', data=_input1, ax=axs[0][1])
sns.countplot(x='Destination', data=_input1, ax=axs[1][0])
sns.countplot(x='VIP', data=_input1, ax=axs[1][1])

def split_data(series, character):
    h = {}
    flag = True
    for string in series:
        if pd.isna(string):
            for j in range(count):
                h[f'list_{j}'].append(np.nan)
            continue
        list = string.split(character)
        if flag:
            count = len(list)
            for i in range(count):
                h[f'list_{i}'] = []
            flag = False
        for j in range(count):
            h[f'list_{j}'].append(list[j])
    return h
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

class SplitFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        passengerid_data = split_data(X_['PassengerId'], '_')
        X_['GroupId'] = passengerid_data['list_0']
        X_['MemberId'] = passengerid_data['list_1']
        cabin_data = split_data(X_['Cabin'], '/')
        X_['Deck'] = cabin_data['list_0']
        X_['Num'] = cabin_data['list_1']
        X_['Side'] = cabin_data['list_2']
        name_data = split_data(X_['Name'], ' ')
        X_['FirstName'] = name_data['list_0']
        X_['SecondName'] = name_data['list_1']
        X_ = X_.drop(['Cabin', 'Name'], axis=1, inplace=False)
        return X_

class RmvPreZerosTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['GroupId'] = X_['GroupId'].apply(lambda x: x.lstrip('0') if not pd.isna(x) else x)
        X_['MemberId'] = X_['MemberId'].apply(lambda x: x.lstrip('0') if not pd.isna(x) else x)
        return X_

class AddLuxuryTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, flag):
        self.flag = flag

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        luxury = []
        for (index, row) in X_.iterrows():
            sum = row['FoodCourt'] + row['ShoppingMall'] + row['Spa'] + row['VRDeck']
            luxury.append(sum)
        X_['luxury'] = luxury
        if self.flag == True:
            X_ = X_[['GroupId', 'MemberId', 'HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FirstName', 'SecondName', 'luxury', 'Transported']]
        elif self.flag == False:
            X_ = X_[['PassengerId', 'GroupId', 'MemberId', 'HomePlanet', 'CryoSleep', 'Deck', 'Num', 'Side', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FirstName', 'SecondName', 'luxury']]
        return X_

class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
data_prep_tr = make_pipeline(SplitFeatureTransformer(), RmvPreZerosTransformer(), AddLuxuryTransformer(flag=True))
data_prep_te = make_pipeline(SplitFeatureTransformer(), RmvPreZerosTransformer(), AddLuxuryTransformer(flag=False))