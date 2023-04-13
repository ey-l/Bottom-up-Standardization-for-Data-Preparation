import numpy as np
import pandas as pd
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
df = pd.concat([_input1, _input0], axis=0)
df.head()
df.info()
df.describe()
df = df.drop(['Name'], axis=1, inplace=False)
df.isnull().sum()

def missing(df1):
    missing_number = df1.isnull().sum().sort_values(ascending=False)
    missing_percent = (df1.isnull().sum() / df1.isnull().count() * 100).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values
missing(df)
df['Age'] = df['Age'].fillna(df['Age'].mean(), inplace=False)
df['RoomService'] = df['RoomService'].fillna(df['RoomService'].mean(), inplace=False)
df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=False)
df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=False)
df['Spa'] = df['Spa'].fillna(df['Spa'].mean(), inplace=False)
df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=False)
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.HomePlanet.value_counts()
df['HomePlanet'] = df['HomePlanet'].replace(to_replace=np.nan, value='Earth', inplace=False)
df['HomePlanet'] = le.fit_transform(df['HomePlanet'])
df.HomePlanet.value_counts()
df.CryoSleep.value_counts()
df.CryoSleep = df.CryoSleep.replace(to_replace=np.nan, value=False, inplace=False)
df.CryoSleep = le.fit_transform(df.CryoSleep)
df.CryoSleep.value_counts()
df.isnull().sum()
df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=False)
df['Cabin'].isnull().sum()
group_id = []
group_num = []
for i in df['PassengerId']:
    group_id.append(i.split('_')[0])
    group_num.append(i.split('_')[1])
df['group_id'] = pd.DataFrame(group_id)
df['group_num'] = pd.DataFrame(group_num)
df['group_id'].value_counts()
deck = []
num = []
side = []
for i in df['Cabin']:
    deck.append(i.split('/')[0])
    num.append(i.split('/')[1])
    side.append(i.split('/')[2])
df['cabin_deck'] = pd.DataFrame(deck)
df['cabin_num'] = pd.DataFrame(num)
df['cabin_side'] = pd.DataFrame(side)
df.head()
df['cabin_deck'].value_counts()
df.cabin_deck = le.fit_transform(df.cabin_deck)
df.cabin_side = le.fit_transform(df.cabin_side)
df.cabin_deck.value_counts()
df.Destination.value_counts()
df.Destination = df.Destination.replace(to_replace=np.nan, value='TRAPPIST-1e', inplace=False)
df.Destination = le.fit_transform(df.Destination)
df.Destination.value_counts()
df.VIP.value_counts()
df.VIP = df.VIP.replace(to_replace=np.nan, value=False, inplace=False)
df.VIP = le.fit_transform(df.VIP)
df.VIP.value_counts()
df = df.drop(['Cabin'], axis=1, inplace=False)
df.isnull().sum()
df.info()
df.head()
df.shape
import seaborn as sns
df['cabin_num'] = pd.to_numeric(df['cabin_num'])
df['group_id'] = pd.to_numeric(df['group_id'])
df['group_num'] = pd.to_numeric(df['group_num'])
df['Transported'] = pd.to_numeric(df['Transported'])
df.info()
sns.kdeplot(x='cabin_num', data=df[df['Transported'].notnull()], hue='Transported')
sns.kdeplot(x='cabin_deck', data=df[df['Transported'].notnull()], hue='Transported')
sns.kdeplot(x='cabin_side', data=df[df['Transported'].notnull()], hue='Transported')
sns.kdeplot(x='group_id', data=df[df['Transported'].notnull()], hue='Transported')
sns.kdeplot(x='group_num', data=df[df['Transported'].notnull()], hue='Transported')
sns.kdeplot(x='Age', data=df[df['Transported'].notnull()], hue='Transported')
df.head(2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df['RoomService'] = sc.fit_transform(df[['RoomService']])
df['Age'] = sc.fit_transform(df[['Age']])
df['FoodCourt'] = sc.fit_transform(df[['FoodCourt']])
df['ShoppingMall'] = sc.fit_transform(df[['ShoppingMall']])
df.head()
df.info()
df.head()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=12, random_state=42)
_input1 = df[df['Transported'].notnull()]
_input0 = df[df['Transported'].isnull()]
_input0 = _input0.drop(['Transported'], axis=1, inplace=False)
_input0.head()
x_train = _input1.drop(['Transported', 'PassengerId', 'group_id', 'cabin_num', 'Spa', 'VRDeck'], axis=1)
y_train = _input1.Transported
x_test = _input0.drop(['PassengerId', 'group_id', 'cabin_num', 'Spa', 'VRDeck'], axis=1)
x_test