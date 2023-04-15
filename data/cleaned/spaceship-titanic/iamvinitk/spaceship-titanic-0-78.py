import pandas as pd
import numpy as np
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(train_df.shape)
print(test_df.shape)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(train_df.corr(), annot=True, fmt='.2f')

train_df.isnull().sum()

def clean_data(df):
    cleaned_df = pd.DataFrame()
    cleaned_df['group'] = df['PassengerId'].apply(lambda x: str(x).split('_')[0])
    cleaned_df['passenger_number'] = df['PassengerId'].apply(lambda x: str(x).split('_')[1])
    cleaned_df['home_planet'] = df['HomePlanet']
    print(cleaned_df['home_planet'].value_counts())
    print(cleaned_df.isnull().sum())
    most_frequent_home_planet = cleaned_df['home_planet'].value_counts().index[0]
    cleaned_df['home_planet'] = cleaned_df['home_planet'].fillna(cleaned_df.groupby('group')['home_planet'].transform(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 1 else most_frequent_home_planet))
    cleaned_df['cryo_sleep'] = df['CryoSleep']
    print(cleaned_df['cryo_sleep'].value_counts())
    print(cleaned_df.isnull().sum())
    most_frequent_cryo_sleep = cleaned_df['cryo_sleep'].value_counts().index[0]
    cleaned_df['cryo_sleep'] = cleaned_df['cryo_sleep'].fillna(cleaned_df.groupby('group')['cryo_sleep'].transform(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 1 else most_frequent_cryo_sleep))
    print(cleaned_df['cryo_sleep'].value_counts())
    print(cleaned_df.isnull().sum())
    print(df['Cabin'].isnull().sum())
    cleaned_df['cabin'] = df['Cabin']
    cleaned_df['cabin'] = cleaned_df['cabin'].fillna('None/-1/M')
    cleaned_df['deck'] = cleaned_df['cabin'].apply(lambda x: str(x).split('/')[0])
    cleaned_df['num'] = cleaned_df['cabin'].apply(lambda x: str(x).split('/')[1])
    cleaned_df['side'] = cleaned_df['cabin'].apply(lambda x: str(x).split('/')[2])
    cleaned_df = cleaned_df.drop(['cabin'], axis=1)
    print(cleaned_df.isnull().sum())
    print(cleaned_df.head(100))
    print(cleaned_df['deck'].value_counts())
    print(cleaned_df['num'].value_counts())
    print(cleaned_df['side'].value_counts())
    print(df['Destination'].isnull().sum())
    cleaned_df['destination'] = df['Destination']
    print(cleaned_df['destination'].value_counts())
    most_frequent_destination = cleaned_df['destination'].value_counts().index[0]
    cleaned_df['destination'] = cleaned_df['destination'].fillna(cleaned_df.groupby('group')['destination'].transform(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 1 else most_frequent_destination))
    print(cleaned_df['destination'].value_counts())
    print(cleaned_df.isnull().sum())
    print(df['Age'].isnull().sum())
    cleaned_df['age'] = df['Age']
    cleaned_df['age'] = cleaned_df['age'].fillna(cleaned_df['age'].median())
    print(cleaned_df['age'].value_counts())
    print(cleaned_df.isnull().sum())
    print(df['VIP'].isnull().sum())
    cleaned_df['vip'] = df['VIP']
    cleaned_df['vip'] = cleaned_df['vip'].fillna(2)
    cleaned_df['vip'] = cleaned_df['vip'].replace({True: 1, False: 0})
    print(cleaned_df['vip'].value_counts())
    print(cleaned_df.isnull().sum())
    print(df['RoomService'].isnull().sum())
    print(df['FoodCourt'].isnull().sum())
    print(df['ShoppingMall'].isnull().sum())
    print(df['Spa'].isnull().sum())
    print(df['VRDeck'].isnull().sum())
    cleaned_df['room_service'] = df['RoomService'].fillna(0)
    cleaned_df['food_court'] = df['FoodCourt'].fillna(0)
    cleaned_df['shopping_mall'] = df['ShoppingMall'].fillna(0)
    cleaned_df['spa'] = df['Spa'].fillna(0)
    cleaned_df['vr_deck'] = df['VRDeck'].fillna(0)
    print(cleaned_df.isnull().sum())
    print(df['Name'].isnull().sum())
    cleaned_df['name'] = df['Name'].fillna('FNU LNU')
    print(cleaned_df['name'].isnull().sum())
    print(cleaned_df['name'].isna().sum())
    cleaned_df['first_name'] = df['Name'].apply(lambda x: str(x).split(' ')[0] if len(str(x).split(' ')) > 1 else 'FNU')
    cleaned_df['last_name'] = df['Name'].apply(lambda x: str(x).split(' ')[1] if len(str(x).split(' ')) > 1 else 'LNU')
    print(cleaned_df.isnull().sum())
    cleaned_df = cleaned_df.drop(['name'], axis=1)
    return cleaned_df
cleaned_train_df = clean_data(train_df)
print(train_df['Transported'].isnull().sum())
cleaned_train_df['transported'] = train_df['Transported']
corr = cleaned_train_df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)

train_df = cleaned_train_df.drop(['transported'], axis=1)
target_df = cleaned_train_df['transported']
print(train_df.head())
print(target_df.head())
train_df = pd.get_dummies(train_df)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def test_for_x_features(train, test, feature_count=10):
    model = ExtraTreesClassifier()