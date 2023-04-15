import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn import tree
from statistics import mode
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPRegressor
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
sample_submission_df = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train_test = pd.concat([test_df, train_df], axis=0).reset_index()
scoreing = {}

def split_value(df):
    df[['Cabin_Dech', 'Cabin_Num', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
    df[['Group_passenger', 'ID_passenger']] = df['PassengerId'].str.split('_', expand=True)
    df[['First_Name', 'Second_Name']] = df['Name'].str.split(' ', expand=True)
    return df.drop(['Cabin', 'PassengerId', 'Name'], axis=1)

def fill_name(df):
    df['First_Name'] = df['First_Name'].fillna('_')
    df['Second_Name'] = df['Second_Name'].fillna('_')
    return df

def name_len(df):
    df['First_Name_len'] = df['First_Name'].map(lambda x: len(x))
    df['Second_Name_len'] = df['Second_Name'].map(lambda x: len(x))
    return df

def total_exp(df):
    df['Min_Total_exp'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1, min_count=1)
    df['Total_exp'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1, min_count=5)
    return df

def columns_spliting(df):
    df = split_value(df)
    df = fill_name(df)
    df = name_len(df)
    df = total_exp(df)
    return df

def smart_missing_fill(df):
    df.loc[df['CryoSleep'].isnull() & (df['Min_Total_exp'] > 0), 'CryoSleep'] = False
    df.loc[(df['CryoSleep'] == True) & df['Total_exp'].isna(), ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0
    df['Total_exp'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1, min_count=5)
    return df

def must_freq_missing_data(df, col):
    freq_values = {}
    most_freq = df[col].mode().iloc[0]
    freq_values[col] = most_freq
    return df.fillna(value=freq_values)

def missing_value_cat_list_model(df, col):
    if df[col].isna().sum() == 0:
        return df
    else:
        test_feature_data = df[df[col].isna()]
        train_feature_data = df[df[col].isna() == False]
        features = df.columns.drop(col)
        train_y = train_feature_data[col]
        train_x = train_feature_data[features]
        test_x = test_feature_data[features]
        cat_features = train_feature_data[features].select_dtypes(include='object').columns
        enc = OrdinalEncoder()
        le = LabelEncoder()
        train_x.loc[:, cat_features] = enc.fit_transform(train_x[cat_features].astype(str))
        test_x.loc[:, cat_features] = enc.fit_transform(test_x[cat_features].astype(str))
        train_y = le.fit_transform(train_y)
        (x_train, x_val, y_train, y_val) = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(x_train, y_train)
        dval = xgb.DMatrix(x_val, y_val)
        dtest = xgb.DMatrix(test_x[features])
        bst = XGBClassifier(n_estimators=10, max_depth=14, learning_rate=0.1)