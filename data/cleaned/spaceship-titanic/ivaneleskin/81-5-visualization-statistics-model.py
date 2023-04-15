import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, learning_curve
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from pathlib import Path
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
from scipy.stats import probplot
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, learning_curve
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
DATA_ROOT = Path('data/input/spaceship-titanic')
DF_TRAIN = DATA_ROOT / 'train.csv'
DF_TEST = DATA_ROOT / 'test.csv'
df = pd.read_csv(DF_TRAIN)
df_test = pd.read_csv(DF_TEST)
test_pass_id = df_test.PassengerId.copy()
df_test.info()
df.corr()['Transported'].sort_values(ascending=False)
from sklearn.base import BaseEstimator, TransformerMixin

class DataPipeline:
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.homePlanet_mode = None
        self.destination_mode = None
        self.vip_mode = None
        self.cabin_mode = None
        self.age_median = None

    def fit(self, df_train):
        """Сохранение статистик"""
        self.homePlanet_mode = df['HomePlanet'].mode()[0]
        self.destination_mode = df['Destination'].mode()[0]
        self.vip_mode = df['VIP'].mode()[0]
        self.cabin_mode = df['Cabin'].mode()[0]
        self.age_median = df['Age'].median()

    def transform(self, df):
        df['Pass_group'] = df.PassengerId.str.split('_').str[0]
        df.Pass_group = df.Pass_group.astype(float)
        df['Lastname'] = df.Name.str.split(' ').str[1]
        df[['Deck', 'Cab_num', 'Deck_side']] = df.Cabin.str.split('/', expand=True)
        df.Cab_num = df.Cab_num.astype(float)
        df['Deck'] = df['Deck'].replace('np.nan', np.nan)
        df['Deck'].fillna(df['Deck'].mode()[0], inplace=True)
        df.loc[df['Deck'].isin(['D', 'F']), 'Deck'] = 'F'
        df.loc[df['Deck'].isin(['E', 'T']), 'Deck'] = 'T'
        df.loc[df.RoomService.gt(9000), 'RoomService'] = 9000
        df.loc[df.FoodCourt.gt(22000), 'FoodCourt'] = 22000
        df.loc[df.ShoppingMall.gt(11000), 'ShoppingMall'] = 11000
        df.loc[df.Spa.gt(17000), 'Spa'] = 17000
        df.loc[df.VRDeck.gt(21000), 'VRDeck'] = 21000
        amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df.loc[df.CryoSleep.eq(True), amenities] = 0
        zero_amenities = df[amenities].sum(axis=1).eq(0)
        df.loc[zero_amenities, amenities] = 0
        for i in amenities:
            df.loc[df[i].isna(), i] = df.loc[df[i].gt(0), i].median()
        df['Total_expenses'] = df[amenities].sum(axis=1)
        df.loc[df.CryoSleep.isna() & df.Total_expenses.gt(0), 'CryoSleep'] = False
        df.loc[df.VIP.isna() & (df.Age < 18), 'VIP'] = False
        df.loc[df.VIP.isna() & (df.HomePlanet == 'Earth'), 'VIP'] = False
        df.loc[df.VIP.isna() & df.HomePlanet.eq('Mars') & df.Destination.eq('55 Cancri e'), 'VIP'] = False
        df.loc[df.VIP.isna() & df.Deck.isin(['G', 'T']), 'VIP'] = False
        df.loc[df.VIP.isna() & df.CryoSleep.eq(False) & ~df.Deck.isin(['A', 'B', 'C', 'D']), 'VIP'] = True
        df.loc[df.HomePlanet.isna() & df.VIP.eq(True) & df.Destination.eq('55 Cancri e'), 'HomePlanet'] = 'Europa'
        present_values = ~df.Pass_group.isna() & ~df.HomePlanet.isna()
        group_home_map = df.loc[present_values, ['Pass_group', 'HomePlanet']].set_index('Pass_group').to_dict()['HomePlanet']
        df.loc[df.HomePlanet.isna(), 'HomePlanet'] = df.Pass_group.map(group_home_map)
        df.loc[df.HomePlanet.isna() & df.Deck.isin(['T', 'A', 'B', 'C']), 'HomePlanet'] = 'Europa'
        df.loc[df.HomePlanet.isna() & df.Deck.eq('G'), 'HomePlanet'] = 'Earth'
        present_values = ~df.Lastname.isna() & ~df.HomePlanet.isna()
        lastname_home_map = df.loc[present_values, ['Lastname', 'HomePlanet']].set_index('Lastname').to_dict()['HomePlanet']
        df.loc[df.HomePlanet.isna(), 'HomePlanet'] = df.Lastname.map(lastname_home_map)
        df.loc[(df.VIP == True) & df.Age.isna(), 'Age'] = df.loc[df.VIP == True, 'Age'].median()
        df.loc[df.Age.isna() & df.Total_expenses.gt(0), 'Age'] = df.loc[df.Total_expenses.gt(0), 'Age'].median()
        df.loc[df.Age.isna() & df.Total_expenses.eq(0) & df.CryoSleep.eq(False), 'Age'] = df.loc[df.Total_expenses.eq(0) & df.CryoSleep.eq(False), 'Age'].median()
        df.Age.fillna(df.Age.median(), inplace=True)
        df.Cab_num.fillna(df.Cab_num.median(), inplace=True)
        Group_members = df.Pass_group.value_counts().to_dict()
        df['Group_members'] = df.Pass_group.map(Group_members)
        Cabin_members = df.Cabin.value_counts().to_dict()
        df['Cabin_members'] = df.Cabin.map(Cabin_members)
        df.Cabin_members.fillna(df.Cabin_members.mean(), inplace=True)
        df.loc[df['HomePlanet'].isin(['Earth', 'Europa']), 'HomePlanet'] = 'Europa'
        df.loc[df['Destination'].isin(['55 Cancri e', 'PSO J318.5-22']), 'Destination'] = 'PSO J318.5-22'
        return df
df_train = df.copy()
pipe = DataPipeline()