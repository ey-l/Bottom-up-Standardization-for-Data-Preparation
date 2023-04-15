import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import copy
original_test_data = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
original_train_data = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
original_train_data.info()
original_train_data.head(4000)
original_train_data.describe()
original_test_data.info()
original_test_data.describe()
original_test_data.head(2000)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=original_train_data.Age)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=original_train_data.RoomService)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=original_train_data.FoodCourt)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=original_train_data.ShoppingMall)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=original_train_data.Spa)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=original_train_data.VRDeck)

def handle_multiple_outliers(df: pd.DataFrame, cols: list):
    new_df = df.copy()
    for col in cols:
        new_df = handle_outliers(new_df, col)
    return new_df

def handle_outliers(df: pd.DataFrame, col: str):
    res_df = df.copy()
    q1 = np.nanpercentile(res_df[col], 25)
    q3 = np.nanpercentile(res_df[col], 75)
    IRQ = q3 - q1
    min_value = q1 - 1.5 * IRQ
    max_value = q3 + 1.5 * IRQ
    median = res_df[col].median()
    print(f'\ncol = {col}; len={len(res_df[col])}')
    print(f'median ={median}, q1={q1}, q1={q3}, IRQ={IRQ}')
    print(f'max={max_value}, min={min_value}')
    print(f'top outliers count = {len(res_df.loc[res_df[col] > max_value, col])}; ')
    print(f'bottom outliers count = {len(res_df.loc[res_df[col] < min_value, col])}; \n')
    res_df.loc[res_df[col] > max_value, col] = max_value
    res_df.loc[res_df[col] < min_value, col] = min_value
    return res_df
train_data = original_train_data.copy()
test_data = original_test_data.copy()
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.Age)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.RoomService)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.FoodCourt)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.ShoppingMall)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.Spa)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.VRDeck)
train_data.VRDeck.hist()

def fill_missed_values(df):
    res_df = df.copy()
    res_df.VIP.fillna(False, inplace=True)
    res_df.CryoSleep.fillna(False, inplace=True)
    res_df.HomePlanet.fillna(res_df.HomePlanet.mode()[0], inplace=True)
    res_df.Destination.fillna(res_df.Destination.mode()[0], inplace=True)
    res_df.Age.fillna(train_data.Age.median(), inplace=True)
    res_df.RoomService.fillna(res_df.RoomService.median(), inplace=True)
    res_df.FoodCourt.fillna(res_df.FoodCourt.median(), inplace=True)
    res_df.ShoppingMall.fillna(res_df.ShoppingMall.median(), inplace=True)
    res_df.Spa.fillna(res_df.Spa.median(), inplace=True)
    res_df.VRDeck.fillna(res_df.VRDeck.median(), inplace=True)
    return res_df
train_data = fill_missed_values(train_data)
test_data = fill_missed_values(test_data)

def change_bool_columns_to_int(df):
    res_df = df.copy()
    res_df.VIP = res_df.VIP.astype(int)
    res_df.CryoSleep = res_df.CryoSleep.astype(int)
    if 'Transported' in res_df.columns:
        res_df.Transported = res_df.Transported.astype(int)
    return res_df
train_data = change_bool_columns_to_int(train_data)
test_data = change_bool_columns_to_int(test_data)

def create_passenger_group_feature(df):
    res_df = df.copy()
    res_df['PassengerGroup'] = res_df.index.str.split('_', 1).str[0]
    return res_df
train_data = create_passenger_group_feature(train_data)
test_data = create_passenger_group_feature(test_data)

def create_features_from_cabin(df):
    res_df = df.copy()
    res_df.Cabin.fillna('F/0/S', inplace=True)
    splits = res_df['Cabin'].str.split('/', 3)
    res_df['CabinDeck'] = splits.str[0]
    res_df['CabinNum'] = splits.str[1].astype(int)
    res_df['CabinSide'] = splits.str[2]
    return res_df
train_data = create_features_from_cabin(train_data)
test_data = create_features_from_cabin(test_data)

def generate_new_features(df):
    res_df = df.copy()
    res_df['TotalSpent'] = res_df['RoomService'] + res_df['Spa'] + res_df['VRDeck']
    res_df['SpentRatio'] = res_df['TotalSpent'] / (res_df['FoodCourt'] + res_df['ShoppingMall'] + 1)
    res_df['TotalSpentLog'] = (res_df['TotalSpent'] + 1).transform(np.log)
    res_df['VIPPlusCryoSleep'] = res_df['VIP'] + res_df['CryoSleep']
    res_df['CabinDeckSide'] = res_df['CabinDeck'] + res_df['CabinSide']
    res_df['HomePlanetDestination'] = res_df['HomePlanet'] + '-' + res_df['Destination']
    res_df['GroupMembersCount'] = res_df.groupby('PassengerGroup')['PassengerGroup'].transform('count')
    return res_df
train_data = generate_new_features(train_data)
test_data = generate_new_features(test_data)

def binning_spent_ratio_feature(df):
    res_df = df.copy()
    bins = [0, 1e-05, 0.1, 5, 100000]
    labels = ['1', '2', '3', '4']
    res_df['SpentRatioBin'] = pd.cut(res_df['SpentRatio'], bins=bins, labels=labels, include_lowest=True).astype(str)
    return res_df
train_data = binning_spent_ratio_feature(train_data)
test_data = binning_spent_ratio_feature(test_data)

def binning_age_feature(df):
    res_df = df.copy()
    bins = [0, 5, 18, 40, 55, 100]
    labels = ['1', '2', '3', '4', '5']
    res_df['AgeBin'] = pd.cut(res_df['Age'], bins=bins, labels=labels, include_lowest=True).astype(str)
    return res_df
train_data = binning_age_feature(train_data)
test_data = binning_age_feature(test_data)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.TotalSpent)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.GroupMembersCount)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.VIPPlusCryoSleep)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.GroupMembersCount)
plt.figure(figsize=(12, 1))
ax = sns.boxplot(x=train_data.TotalSpent)
X_select = train_data.drop('Transported', axis=1)[['GroupMembersCount', 'TotalSpentLog', 'TotalSpent', 'SpentRatio', 'VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt', 'RoomService', 'CryoSleep', 'VIP', 'VIPPlusCryoSleep', 'Age']]
y_select = train_data.Transported.astype(bool)