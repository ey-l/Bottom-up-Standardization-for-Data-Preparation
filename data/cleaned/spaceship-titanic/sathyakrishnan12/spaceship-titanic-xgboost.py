import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
url = 'data/input/spaceship-titanic/'
df = pd.read_csv(url + 'train.csv')
test = pd.read_csv(url + 'test.csv')
df.head()
df.info()
df.isnull().sum()
df.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
df.HomePlanet.value_counts()

def count_plot(data):
    cat_cols = data.select_dtypes(include='object').columns
    nrows = int(np.ceil(len(cat_cols) / 2))
    (fig, axes) = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 15))
    if len(cat_cols) % 2 == 1:
        axes[nrows - 1][1].axis('off')
    i = 0
    for j in range(nrows):
        for k in range(2):
            if i == len(cat_cols):
                break
            sns.countplot(data=data, x=cat_cols[i], ax=axes[j][k])
            i += 1
        if i == len(cat_cols):
            break
count_plot(df.drop(['PassengerId', 'Cabin'], axis=1))

def hist_plot(data, kde=False, include_int=False, hue=None, bins=10):
    if include_int:
        cols = list(data.select_dtypes(include=['float', 'int']).columns)
    else:
        cols = list(data.select_dtypes(include='float').columns)
    nrows = int(np.ceil(len(cols) / 2))
    (fig, axes) = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 15))
    if len(cols) % 2 == 1:
        axes[nrows - 1][1].axis('off')
    i = 0
    for j in range(nrows):
        for k in range(2):
            if i == len(cols):
                break
            sns.histplot(data=data, x=cols[i], ax=axes[j][k], kde=kde, hue=hue, bins=bins)
            i += 1
        if i == len(cols):
            break
hist_plot(df, kde=True, hue='HomePlanet')
hist_plot(df, kde=True, hue='Transported', bins=3)

def scatter_plot(data, x_axis, hue=None):
    cols = list(data.select_dtypes(include=['float', 'int']).columns)
    cols.remove(x_axis)
    nrows = int(np.ceil(len(cols) / 2))
    (fig, axes) = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 15))
    if len(cols) % 2 == 1:
        axes[nrows - 1][1].axis('off')
    i = 0
    for j in range(nrows):
        for k in range(2):
            if i == len(cols):
                break
            sns.scatterplot(data=data, x=x_axis, y=cols[i], ax=axes[j][k], hue=hue)
            i += 1
        if i == len(cols):
            break
scatter_plot(df, x_axis='Spa', hue='Transported')
scatter_plot(df, x_axis='FoodCourt', hue='Transported')
scatter_plot(df, 'VRDeck', hue='Transported')
scatter_plot(df, 'RoomService', hue='Transported')
df['group'] = np.zeros(shape=(len(df), 1))
test['group'] = np.zeros(shape=(len(test), 1))
(df.columns, df.shape, test.columns, test.shape)
for i in range(len(df)):
    df.iat[i, 13] = int(df.iloc[i].PassengerId.split('_')[0])
for i in range(len(test)):
    test.iat[i, 12] = int(test.iloc[i].PassengerId.split('_')[0])
df.head(5)
test.head(5)
df.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)
train = df.drop('Transported', axis=1)
train = pd.concat([train, test], axis=0)
train.shape
age_null_grp = set(train[train['Age'].isnull()].group.value_counts().index)
grp_with_more_than_one = set(train.group.value_counts()[train.group.value_counts() > 1].index)
common_grps = list(age_null_grp.intersection(grp_with_more_than_one))
train_temp = train[train['group'] == common_grps[0]]
train_temp[train_temp.Age.isnull()].index
ind_age = {}
for i in common_grps:
    train_temp = train[train['group'] == i]
    temp_mean = train_temp.Age.mean()
    age_null_ind = train_temp[train_temp.Age.isnull()].index[0]
    if not pd.isna(temp_mean):
        ind_age[age_null_ind] = int(np.round(temp_mean, 0))
    else:
        ind_age[age_null_ind] = int(np.round(train.Age.mean(), 0))
ind_age
train.columns
for i in ind_age.keys():
    train.at[i, 'Age'] = ind_age[i]
train.Age.isnull().sum()
planet_null_grps = set(train[train['HomePlanet'].isnull()].group.value_counts().index)
common_grps = list(planet_null_grps.intersection(grp_with_more_than_one))
ind_planet = {}
for i in common_grps:
    train_temp = train[train['group'] == i]
    if len(train_temp.HomePlanet.mode().values) > 0:
        temp_planet = train_temp.HomePlanet.mode().values[0]
        planet_null_ind = train_temp[train_temp.HomePlanet.isnull()].index[0]
        ind_planet[planet_null_ind] = temp_planet
ind_planet
for i in ind_planet.keys():
    train.at[i, 'HomePlanet'] = ind_planet[i]
train.HomePlanet.isnull().sum()
train.isnull().sum()
mean_planet_age = train.groupby('HomePlanet').mean()['Age'].to_dict()

def fill_age(x, y):
    if pd.isna(x):
        if pd.isna(y):
            return int(train.Age.mean())
        else:
            return int(mean_planet_age[y])
    return x
train['Age'] = train.apply(lambda a: fill_age(a['Age'], a['HomePlanet']), axis=1)
train.isnull().sum()
cryosleep_null = set(train[train['CryoSleep'].isnull()].group.value_counts().index)
train[train['CryoSleep'].isnull()].group
common_grps = list(cryosleep_null.intersection(grp_with_more_than_one))
ind_cs = {}
for i in common_grps:
    train_temp = train[train['group'] == i]
    if len(train_temp.CryoSleep.mode().values) > 0:
        temp_cs = train_temp.CryoSleep.mode().values[0]
        cs_null_ind = train_temp[train_temp.CryoSleep.isnull()].index[0]
        ind_cs[cs_null_ind] = temp_cs
len(ind_cs)
train.columns
for i in ind_cs.keys():
    train.at[i, 'CryoSleep'] = ind_cs[i]
train.CryoSleep.isnull().sum()
train.isnull().sum()
cabin_null = list(train[train['Cabin'].isnull()].group.value_counts().index)

def fill_nas(column):
    print('Before filling: ', train[column].isnull().sum())
    col_null = set(train[train[column].isnull()].group.value_counts().index)
    common_grps = list(col_null.intersection(grp_with_more_than_one))
    ind_nulls = {}
    for i in common_grps:
        train_temp = train[train['group'] == i]
        if len(train_temp[column].mode().values) > 0:
            temp_val = train_temp[column].mode().values[0]
            temp_null_ind = train_temp[train_temp[column].isnull()].index[0]
            ind_nulls[temp_null_ind] = temp_val
    for i in ind_nulls.keys():
        train.at[i, column] = ind_nulls[i]
    print('After filling: ', train[column].isnull().sum())
train.isnull().sum()
fill_nas('VIP')
fill_nas('Cabin')
fill_nas('Destination')
train['RoomService'] = train['RoomService'].fillna(0)
train['VRDeck'] = train['VRDeck'].fillna(0)
train['Spa'] = train['Spa'].fillna(0)
train['ShoppingMall'] = train['ShoppingMall'].fillna(0)

def fill_foodcourt(x, y):
    if pd.isna(x):
        if y in ['Earth', 'Mars']:
            return 0.0
        else:
            return 11.0
    return x
train['FoodCourt'] = train.apply(lambda a: fill_foodcourt(a['FoodCourt'], a['HomePlanet']), axis=1)
train.groupby('HomePlanet').describe().transpose().loc['RoomService':'VRDeck']

def fill_planet(rs, fc, planet):
    if pd.isna(planet):
        if rs < 22 and fc < 5:
            return 'Earth'
        elif rs < 22 and fc > 5:
            return 'Europa'
        else:
            return 'Mars'
    return planet
train['HomePlanet'] = train.apply(lambda a: fill_planet(a['RoomService'], a['FoodCourt'], a['HomePlanet']), axis=1)
train.isnull().sum()

def start_dest(x, y):
    if pd.isna(y):
        return np.nan
    else:
        return x + '_' + y
train['Start_Destination'] = train.apply(lambda a: start_dest(a['HomePlanet'], a['Destination']), axis=1)
train['Start_Destination'].value_counts()
train.groupby('Start_Destination').describe().T
train.groupby('Destination').describe().T
train.groupby('Destination').describe().T.loc[('VRDeck', ['mean', 'std']), :]

def fill_dest(x, y):
    if pd.isna(x):
        if y in range(200, 300):
            return 'TRAPPIST-1e'
        elif y in range(130, 180):
            return 'PSO J318.5-22'
        else:
            return '55 Cancri e'
    return x
train['Destination'] = train.apply(lambda a: fill_dest(a['Destination'], a['VRDeck']), axis=1)
train.isnull().sum()
train['Start_Destination'] = train.apply(lambda a: start_dest(a['HomePlanet'], a['Destination']), axis=1)

def deck(x):
    if not pd.isna(x):
        return x.split('/')[0]
    return np.nan

def num(x):
    if not pd.isna(x):
        return x.split('/')[1]
    return np.nan

def side(x):
    if not pd.isna(x):
        return x.split('/')[2]
    return np.nan
train['Deck'] = train['Cabin'].apply(deck)
train['Num'] = train['Cabin'].apply(num)
train['Side'] = train['Cabin'].apply(side)
train['Deck'].value_counts()
train.drop('Cabin', axis=1, inplace=True)
train.drop('Num', axis=1, inplace=True)
train.groupby('Deck').describe().T.loc[(slice(None), ['mean', 'std', '50%']), :]
train['TotalSum'] = train.RoomService.fillna(0) + train.ShoppingMall.fillna(0) + train.Spa.fillna(0) + train.VRDeck.fillna(0) + train.FoodCourt.fillna(0)
train.isnull().sum()
missing_cols = ['CryoSleep', 'VIP', 'Deck', 'Side']
missing_cols[1:]
df_train = train.drop(missing_cols[1:], axis=1).copy()
(df_train.shape, df_train.columns)
from sklearn.preprocessing import StandardScaler, LabelEncoder
df_train.drop(['HomePlanet', 'Destination'], axis=1, inplace=True)
one_hot_cols = ['Start_Destination']
scale_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSum']
encode_cols = ['Age', 'group']
df_train['CryoSleep'] = df_train['CryoSleep'].map({True: 1, False: 0})
for col in one_hot_cols:
    temp = pd.get_dummies(df_train[col], drop_first=True, prefix=col)
    df_train.drop(col, axis=1, inplace=True)
    df_train = pd.concat([df_train, temp], axis=1)
for col in scale_cols:
    sc = StandardScaler()
    x = sc.fit_transform(df_train[col].values.reshape(-1, 1))
    df_train.drop(col, axis=1, inplace=True)
    df_train[col] = x
for col in encode_cols:
    lab_enc = LabelEncoder()
    x = lab_enc.fit_transform(df_train[col].values.reshape(-1, 1))
    df_train.drop(col, axis=1, inplace=True)
    df_train[col] = x
cryosleep_test = df_train[df_train['CryoSleep'].isna()].copy()
cryosleep_test.drop('CryoSleep', axis=1, inplace=True)
cryosleep_test
cryosleep_train = df_train.dropna().copy()
cryosleep_train.shape
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def impute_using_model(train_data, test_data, train_cols, target_col):
    (X_train, X_test, y_train, y_test) = train_test_split(train_data[train_cols], train_data[target_col], test_size=0.2, random_state=42, stratify=train_data[target_col])
    rf = RandomForestClassifier(n_estimators=200, max_depth=10)