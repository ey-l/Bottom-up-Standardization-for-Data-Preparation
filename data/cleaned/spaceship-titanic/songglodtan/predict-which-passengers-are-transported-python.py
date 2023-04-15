import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_path = 'data/input/spaceship-titanic/train.csv'
test_path = 'data/input/spaceship-titanic/test.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_train.head()
df_train.info()
df_train.describe()
df_train.isna().sum()[df_train.isna().sum() > 0]
df_train.duplicated().sum()
df_train.nunique()
df_test.head()
df_test.info()
df_test.describe()
df_test.isna().sum()[df_test.isna().sum() > 0]
df_test.duplicated().sum()

def split_cabin(df):
    df['Cabin'].fillna('Z/9999/Z', inplace=True)
    cols = ['CabinDeck', 'CabinNum', 'CabinSide']
    df[cols] = df['Cabin'].str.split('/', expand=True)
    df.drop(['CabinNum'], axis=1, inplace=True)
    df.loc[df['CabinDeck'] == 'Z', 'CabinDeck'] = np.nan
    df.loc[df['CabinSide'] == 'Z', 'CabinSide'] = np.nan
    df.drop(['Cabin'], axis=1, inplace=True)
split_cabin(df_train)
split_cabin(df_test)
df_train.head()
df_train.info()
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.concat([df_train, df_test], ignore_index=True)
data.groupby('Destination').size()
plt.pie(data=data.groupby('Destination').size().reset_index(name='Count'), x='Count', labels='Destination', autopct='%.2f%%')

data.groupby(['Destination', 'HomePlanet']).size().unstack()
sns.boxplot(data=data, x='Age', y='HomePlanet')
plt.figure(figsize=(10, 5))
sns.histplot(data=df_train, x='Age', hue='Transported', kde=True, binwidth=1)
plt.figure(figsize=(10, 5))
sns.histplot(data=data, x='Age', kde=True, binwidth=1)
for planet in ['Europa', 'Earth', 'Mars']:
    age_median = data.loc[data['HomePlanet'] == planet, 'Age'].median()
    print('The age of people from', planet, 'is around', age_median)
age_median = {'Europa': 33, 'Earth': 23, 'Mars': 28}
sns.countplot(data=data, x='VIP')
sns.countplot(data=data, x='VIP', hue='Transported')
sns.countplot(data=data, x='HomePlanet', hue='Transported')
sns.countplot(data=data, x='CryoSleep', hue='Transported')
sns.countplot(data=data, x='CabinDeck', hue='Transported')
data.groupby('CabinDeck').size()
data.groupby(['CabinDeck', 'HomePlanet']).size().unstack().fillna(0)
data.groupby(['HomePlanet', 'CabinDeck', 'Destination']).size().unstack().fillna(0)
data.groupby(['CabinSide', 'HomePlanet']).size().unstack()
data.groupby(['CabinDeck', 'CabinSide', 'HomePlanet']).size().unstack().fillna(0)
sns.countplot(data=data, x='CabinSide', hue='Transported')
data.groupby('CabinSide').size()
sns.heatmap(data=df_train.corr(), annot=True, fmt='.2f')
print('train: missing values in CryoSleep column is', df_train.isna().sum()['CryoSleep'])
print('test: missing values in CryoSleep column is', df_test.isna().sum()['CryoSleep'])
df_train[df_train['CryoSleep'] == True].describe()
df_train[df_train['CryoSleep'] == False].describe()

def update_amount_bill(df):
    cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df.loc[df['CryoSleep'] == True, cols] = 0

def update_cryo_sleep(df):
    cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df.loc[(df[cols[0]] == 0) & (df[cols[1]] == 0) & (df[cols[2]] == 0) & (df[cols[3]] == 0) & (df[cols[4]] == 0), 'CryoSleep'] = True
    df.loc[(df[cols[0]] > 0) | (df[cols[1]] > 0) | (df[cols[2]] > 0) | (df[cols[3]] > 0) | (df[cols[4]] > 0), 'CryoSleep'] = False
update_amount_bill(df_train)
update_amount_bill(df_test)
update_cryo_sleep(df_train)
update_cryo_sleep(df_test)
print('train: missing values in CryoSleep column is', df_train.isna().sum()['CryoSleep'])
print('test: missing values in CryoSleep column is', df_test.isna().sum()['CryoSleep'])
from sklearn.impute import SimpleImputer
STRATEGY_NUMERIC = 'median'
exp_cols = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']

def impute_missing_data(df, cols, strategy):
    imputer = SimpleImputer(strategy=strategy)