import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head()
train.describe()
train.info()
train[train.isna().any(axis=1)]
train['HomePlanet'].unique()
train['HomePlanet']
sns.barplot(data=train, x='HomePlanet', y='Transported')
train['Destination'].unique()
sns.barplot(data=train, x='Destination', y='Transported')
train[['Deck', 'Number', 'Side']] = train['Cabin'].str.split('/', expand=True)
test[['Deck', 'Number', 'Side']] = test['Cabin'].str.split('/', expand=True)
bins = [0, 13, 18, 25, 200]
labels = ['<=13', '13-18', '18-25', '>25']
train['AgeGroup'] = pd.cut(train['Age'], bins=bins, labels=labels, right=False)
test['AgeGroup'] = pd.cut(test['Age'], bins=bins, labels=labels, right=False)
train.columns
train['Name'] = train['Name'].apply(lambda x: str(x).split(' ')[-1])
test['Name'] = test['Name'].apply(lambda x: str(x).split(' ')[-1])
train.isnull().sum()
train[['group_id', 'id_in_group']] = train['PassengerId'].str.split('_', expand=True)
test[['group_id', 'id_in_group']] = test['PassengerId'].str.split('_', expand=True)
train['total_in_group'] = train['group_id'].map(lambda x: pd.concat([train['group_id'], test['group_id']]).value_counts()[x])
test['total_in_group'] = test['group_id'].map(lambda x: pd.concat([train['group_id'], test['group_id']]).value_counts()[x])
Y = train[['PassengerId', 'Transported']]
X = train.drop(['Transported'], axis=1)
total_data = pd.concat([X, test]).reset_index()
total_data[['Deck', 'Number', 'Side']] = total_data.groupby('group_id')[['Deck', 'Number', 'Side']].fillna(method='ffill').fillna(np.nan)
total_data['HomePlanet'] = total_data.groupby('group_id')['HomePlanet'].fillna(method='ffill').fillna(np.nan)
total_data[['Deck', 'Number', 'Side', 'HomePlanet']].groupby(['HomePlanet', 'Deck']).size().reset_index()

def some(x):
    if pd.isnull(x['HomePlanet']):
        if x['Deck'] == 'G':
            return 'Earth'
        elif (x['Deck'] == 'T') | (x['Deck'] == 'A') | (x['Deck'] == 'B') | (x['Deck'] == 'C'):
            return 'Europa'
    else:
        return x['HomePlanet']
total_data['HomePlanet'] = total_data.apply(some, axis=1)
total_data[['HomePlanet', 'VIP']].groupby(['HomePlanet', 'VIP']).size().reset_index()
total_data.loc[total_data['HomePlanet'] == 'Earth', 'VIP'] = total_data.loc[total_data['HomePlanet'] == 'Earth', 'VIP'].fillna(False)
total_data['total_expenses'] = total_data.iloc[:, 7:12].sum(axis=1, skipna=False)
sns.barplot(data=total_data, x='AgeGroup', y='total_expenses')
sns.barplot(data=total_data, x='CryoSleep', y='total_expenses')
cond5 = (total_data['AgeGroup'] == '0-13') | (total_data['CryoSleep'] == True)
total_data.loc[cond5, 'RoomService'] = total_data.loc[cond5, 'RoomService'].fillna(0)
total_data.loc[cond5, 'FoodCourt'] = total_data.loc[cond5, 'FoodCourt'].fillna(0)
total_data.loc[cond5, 'ShoppingMall'] = total_data.loc[cond5, 'ShoppingMall'].fillna(0)
total_data.loc[cond5, 'Spa'] = total_data.loc[cond5, 'Spa'].fillna(0)
total_data.loc[cond5, 'VRDeck'] = train.loc[cond5, 'VRDeck'].fillna(0)
total_data[['VIP', 'AgeGroup']].groupby(['VIP', 'AgeGroup']).size().reset_index()
total_data.isnull().sum()
total_data['Name'] = total_data.groupby('group_id')['Name'].fillna(method='ffill').fillna(np.nan)
total_data[['AgeGroup', 'CryoSleep', 'total_expenses', 'Destination']].groupby(['AgeGroup', 'CryoSleep', 'total_expenses', 'Destination']).size().reset_index().query('total_expenses==0')
cond1 = (total_data['AgeGroup'] != '<=13') & (total_data['CryoSleep'] == False) & (total_data['total_expenses'] == 0)
total_data.loc[cond1, 'Destination'] = total_data.loc[cond1, 'Destination'].fillna('TRAPPIST-1e')
total_data[['HomePlanet', 'AgeGroup', 'Destination']].groupby(['AgeGroup', 'HomePlanet', 'Destination']).size().reset_index()
pd.crosstab(total_data['HomePlanet'], total_data['Destination'])
bins = [0, 400, 800, 100000]
labels = ['0-400', '400-800', '800-1000000']
total_data['expenses_group'] = pd.cut(total_data['total_expenses'], bins=bins, labels=labels, right=False)
x = total_data[total_data['RoomService'].notna()]['RoomService']
(q1, q3) = np.percentile(x, [25, 98])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
qq = np.where((x > upper_bound) | (x < lower_bound))[0]
train.isnull().sum()
total_data.columns
total_data['ShoppingMall'] = total_data['ShoppingMall'].fillna(total_data.groupby('HomePlanet')['ShoppingMall'].transform('median'))
total_data['RoomService'] = total_data['RoomService'].fillna(total_data.groupby('HomePlanet')['RoomService'].transform('median'))
total_data['FoodCourt'] = total_data['FoodCourt'].fillna(total_data.groupby('HomePlanet')['FoodCourt'].transform('median'))
total_data['Spa'] = total_data['Spa'].fillna(total_data.groupby('HomePlanet')['Spa'].transform('median'))
total_data['VRDeck'] = total_data['VRDeck'].fillna(total_data.groupby('HomePlanet')['VRDeck'].transform('median'))
total_data.isnull().sum()
train[['group_id', 'Transported', 'HomePlanet']].groupby(['HomePlanet', 'group_id']).mean().reset_index()
total_data['total_in_group'] = np.where(total_data['total_in_group'] == 1, 0, 1)
total_data['Number'] = total_data['Number'].astype(float)
total_data['group_id'] = total_data['group_id'].astype(float)
total_data['id_in_group'] = total_data['id_in_group'].astype(int)
total_data['CryoSleep'] = total_data['CryoSleep'].astype(bool)
total_data['VIP'] = total_data['VIP'].astype(bool)
plt.figure(figsize=(18, 18))
sns.heatmap(train.corr(), annot=True, cmap='RdYlGn')

sns.histplot(x='Number', data=train, hue='Transported', kde=True, bins=20)
total_data['Number'] = total_data['Number'].apply(lambda x: np.log10(x) if x != 0 else x)
total_data.isnull().sum()

def numerical_(df):
    data_num = df.select_dtypes(['float64', 'int64'])
    cols_num = list(data_num.columns)
    dict_num = {i: cols_num[i] for i in range(len(cols_num))}
    imputer = SimpleImputer(strategy='median')
    d = imputer.fit_transform(data_num)
    temp1 = pd.DataFrame(d, index=df.index)
    temp1 = temp1.rename(columns=dict_num)
    return temp1

def object_(df):
    obj_data = df.select_dtypes(['object', 'category'])
    cols = list(obj_data.columns)
    for col in cols:
        obj_data[col].fillna(obj_data[col].mode()[0], inplace=True)
    for i in cols:
        un = obj_data[i].unique()
        ran = range(1, len(un) + 1)
        obj_data.replace(dict(zip(un, ran)), inplace=True)
    obj_data = pd.get_dummies(obj_data, columns=['HomePlanet', 'Destination'])
    return obj_data

def boolean_(df):
    bool_data = df.select_dtypes(['bool'])
    cols = bool_data.columns
    for i in cols:
        bool_data[i] = LabelEncoder().fit_transform(bool_data[i])
    return bool_data
total_num = numerical_(total_data)
bins = [0, 13, 18, 25, 200]
labels = ['<=13', '13-18', '18-25', '>25']
total_data['AgeGroup'] = pd.cut(total_num['Age'], bins=4, labels=labels, right=False)
test['AgeGroup'] = pd.cut(test['Age'], bins=bins, labels=labels, right=False)
total_data['total_expenses'] = total_num[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
bins = [0, 400, 800, 1200, 100000]
labels = ['0-400', '400-800', '800-1200', '1200-1000000']
total_data['expenses_group'] = pd.cut(total_data['total_expenses'], bins=bins, labels=labels, right=False)
total_cat = object_(total_data)
total_bool = boolean_(total_data)
final_data = pd.concat([total_num, total_cat, total_bool], axis=1)
final_data.columns
final_data['PassengerId'] = total_data['PassengerId']
train_data = pd.merge(final_data, Y, how='inner', on=['PassengerId'])
test_data = pd.merge(final_data, test['PassengerId'], on=['PassengerId'], how='inner')
predictor_cols = final_data.drop(['VIP', 'total_expenses', 'total_in_group', 'Name', 'Age', 'PassengerId', 'Cabin'], axis=1).columns[1:]
X = train_data[predictor_cols]
Y = train_data['Transported']
(x_train, x_val, y_train, y_val) = train_test_split(X, Y, test_size=0.2, random_state=60, shuffle=True)
params = {'n_estimators': [i for i in range(100, 600, 50)], 'max_depth': [i for i in range(6, 20, 2)], 'min_samples_leaf': [i for i in range(15, 40, 3)], 'min_samples_split': [i for i in range(5, 240, 10)]}
params = {'n_estimators': [i for i in range(200, 500, 50)]}
rfc = RandomForestClassifier()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1001)
grid = GridSearchCV(rfc, params, cv=skf, scoring='accuracy', return_train_score=False, verbose=1)