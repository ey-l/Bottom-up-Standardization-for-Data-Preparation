import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier, Pool
from sklearn.feature_selection import RFECV
import category_encoders as ce
import joblib
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0

def category_comp_plot(df, by, forCol):
    col_series = _input1[by].value_counts().index
    (fig, axs) = plt.subplots(1, len(col_series), figsize=(12, 5))
    plt.subplots_adjust(right=1.5)
    for (i, category) in enumerate(col_series):
        plt.subplot(1, len(col_series), i + 1)
        sns.countplot(data=_input1.loc[_input1[by] == category], x=forCol)
        plt.title(category)

def numeric_comp_plot(df, by, forCol, binwidth, xlim=None, ylim=None):
    my_df = _input1.copy()
    my_df[by] = my_df[by].fillna(my_df[by].median(), inplace=False)
    my_df[by] = my_df[by].astype('int')
    sns.histplot(data=my_df, x=by, hue=forCol, binwidth=binwidth)
    plt.xlim(xlim)
    plt.ylim(ylim)
category_comp_plot(_input1, 'VIP', 'Transported')
category_comp_plot(_input1, 'HomePlanet', 'Transported')
category_comp_plot(_input1, 'CryoSleep', 'Transported')
_input1['deck'] = _input1['Cabin'].str.split('/', expand=True)[0]
category_comp_plot(_input1, 'deck', 'Transported')
_input1['Cabin_Num'] = _input1['Cabin'].str.split('/', expand=True)[1]
numeric_comp_plot(_input1, 'Cabin_Num', forCol='Transported', binwidth=100, xlim=[0, 2000])
_input1['Side'] = _input1['Cabin'].str[-1]
category_comp_plot(_input1, 'Side', 'Transported')
category_comp_plot(_input1, 'Destination', 'Transported')
numeric_comp_plot(_input1, 'RoomService', forCol='Transported', binwidth=1000, xlim=[0, 10000], ylim=[0, 4000])
numeric_comp_plot(_input1, 'ShoppingMall', forCol='Transported', binwidth=2000, xlim=[0, 10000], ylim=[0, 200])
numeric_comp_plot(_input1, 'ShoppingMall', forCol='Transported', binwidth=2000, xlim=[0, 2000], ylim=[0, 5000])
numeric_comp_plot(_input1, 'Spa', forCol='Transported', binwidth=2000, xlim=[0, 25000], ylim=[0, 4000])
numeric_comp_plot(_input1, 'VRDeck', forCol='Transported', binwidth=2000, xlim=[0, 25000], ylim=[0, 25])
numeric_comp_plot(_input1, 'FoodCourt', forCol='Transported', binwidth=2000, ylim=[0, 25])
spentNadf = _input1.dropna(subset=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP'])
spentNadf['TotalSpent'] = spentNadf['RoomService'] + spentNadf['FoodCourt'] + spentNadf['ShoppingMall'] + spentNadf['Spa'] + spentNadf['VRDeck']
spentNadf['spent?'] = spentNadf['TotalSpent'].apply(lambda x: 1 if x > 0 else 0)
numeric_comp_plot(spentNadf, 'Age', 'spent?', binwidth=5, xlim=[0, 100], ylim=[0, 1000])
numeric_comp_plot(spentNadf, 'Age', 'CryoSleep', binwidth=5, xlim=[0, 100], ylim=[0, 1000])
_input1[['P_ID1', 'P_ID2']] = _input1['PassengerId'].str.split('_', expand=True)[[0, 1]]
_input1['groupSize'] = _input1.groupby(['P_ID1'])['P_ID1'].transform('count')
numeric_comp_plot(_input1, 'groupSize', forCol='Transported', binwidth=1, ylim=[0, 5000])
_input1[['P_ID1', 'P_ID2']] = _input1['PassengerId'].str.split('_', expand=True)[[0, 1]]
_input1['P_ID1'] = _input1['P_ID1'].astype('int')
numeric_comp_plot(_input1, 'P_ID1', forCol='Transported', binwidth=100, xlim=[0, 10000], ylim=[0, 75])
_input1[['P_ID1', 'P_ID2']] = _input1['PassengerId'].str.split('_', expand=True)[[0, 1]]
category_comp_plot(_input1, 'P_ID2', 'Transported')
numeric_comp_plot(_input1, 'Age', 'Transported', binwidth=6, xlim=[0, 100], ylim=[0, 1000])
_input1['lastName'] = _input1['Name'].dropna().str.split(' ').apply(lambda x: x[1])
_input1['vipInFamily'] = _input1.groupby(['lastName'])['VIP'].transform(lambda x: True if x.sum() > 0 else False)
category_comp_plot(_input1, 'vipInFamily', 'Transported')
_input0['lastName'] = _input0['Name'].str.split(' ', expand=True)[1]
all_last_names = pd.concat([_input1['lastName'], _input0['lastName']]).reset_index()
_input1 = pd.merge(_input1, all_last_names.groupby(['lastName'])['index'].agg('count').reset_index().rename(columns={'index': 'family_size'}), how='left', on='lastName', suffixes=['_train', '_all'])
numeric_comp_plot(_input1, 'family_size', 'Transported', binwidth=1, xlim=[0, 20], ylim=[0, 700])
PlanetDeckPivot = pd.pivot_table(_input1, index='HomePlanet', columns='deck', values='PassengerId', aggfunc=len)
sns.heatmap(PlanetDeckPivot, cmap='crest')
PidPlanetPivot = pd.pivot_table(_input1, index='P_ID1', columns='HomePlanet', values='PassengerId', aggfunc=len).head(30)
sns.heatmap(PidPlanetPivot, cmap='crest')
PidPort = pd.pivot_table(_input1, index='P_ID1', columns='Side', values='PassengerId', aggfunc=len).head(30)
sns.heatmap(PidPort, cmap='crest')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.isna().sum()

class Int_Imputer(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        imputer = SimpleImputer(strategy='median')