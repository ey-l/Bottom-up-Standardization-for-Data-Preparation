import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from scipy.stats import shapiro
from scipy.stats import probplot
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
DATA_ROOT = Path('data/input/spaceship-titanic')
DF_TRAIN = DATA_ROOT / 'train.csv'
DF_TEST = DATA_ROOT / 'test.csv'
df_train = pd.read_csv(DF_TRAIN)
df_test = pd.read_csv(DF_TEST)
df_train.head(10).T
df_train.info()
from sklearn.base import BaseEstimator, TransformerMixin

class DataPipeline:
    """Initial data preparation"""

    def __init__(self):
        """Class parameters"""
        self.homePlanet_mode = None
        self.destination_mode = None
        self.vip_mode = None
        self.cabin_mode = None
        self.age_median = None

    def fit(self, df_train):
        """Saving statistics"""
        self.homePlanet_mode = df_train['HomePlanet'].mode()[0]
        self.destination_mode = df_train['Destination'].mode()[0]
        self.vip_mode = df_train['VIP'].mode()[0]
        self.cabin_mode = df_train['Cabin'].mode()[0]
        self.age_median = df_train['Age'].median()

    def transform(self, df_train):
        df_train[['PassengerIdGroup', 'PassengerIdNum']] = df_train['PassengerId'].str.split(pat='_', expand=True).astype('int')
        df_train = df_train.drop(['PassengerId'], axis=1)
        df_train['PassengerIdNum'] = df_train['PassengerIdNum'].astype('object')
        df_train['PassengerIdGroup'] = df_train['PassengerIdGroup'].astype('object')
        df_train['commonSum'] = df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
        df_train['CryoSleep'] = np.where(df_train['CryoSleep'].isna() & (df_train['commonSum'] > 0), True, df_train['CryoSleep'])
        df_train['CryoSleep'] = np.where(df_train['CryoSleep'].isna() & (df_train['commonSum'] == 0), False, df_train['CryoSleep'])
        df_train['Cabin'] = df_train['Cabin'].fillna(self.cabin_mode)
        df_train[['CabinDeck', 'CabinNum', 'CabinSide']] = df_train['Cabin'].str.split(pat='/', expand=True)
        df_train['HomePlanet'] = df_train['HomePlanet'].fillna(self.homePlanet_mode)
        df_train['commonSum'] = df_train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
        df_train['Destination'] = df_train['Destination'].fillna(self.destination_mode)
        df_train.fillna({'RoomService': 0, 'FoodCourt': 0, 'ShoppingMall': 0, 'Spa': 0, 'VRDeck': 0}, inplace=True)
        df_train.fillna({'commonSum': 0}, inplace=True)
        df_train['VIP'] = df_train['VIP'].fillna(self.vip_mode)
        df_train['VIP'] = df_train['VIP'].astype('object')
        df_train['Age'] = df_train['Age'].fillna(self.age_median)
        df_train = df_train.drop(['Cabin'], axis=1)
        df_train = df_train.drop(['CabinNum'], axis=1)
        df_train = df_train.drop(['Name'], axis=1)
        df_train = df_train.drop(['PassengerIdGroup'], axis=1)
        return df_train
pipe = DataPipeline()