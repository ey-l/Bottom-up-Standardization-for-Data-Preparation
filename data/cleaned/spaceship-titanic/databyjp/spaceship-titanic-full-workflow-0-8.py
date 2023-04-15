import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from pathlib import Path
path = Path('data/input/spaceship-titanic/')
import pandas as pd
import numpy as np
df = pd.read_csv(path / 'train.csv')
test_df = pd.read_csv(path / 'test.csv')
df
df.describe(include=np.number)
df.describe(include=object)
df.isna().sum()
df.isna().sum(axis=1).describe()
df[['PassengerId', 'Cabin']]

def proc_passenger_id(df_in):
    df_in['passenger_group'] = df_in['PassengerId'].str.split('_', expand=True)[0].astype(int)
    df_in['passenger_group_size'] = df_in.groupby('passenger_group').transform('count')['PassengerId']
    df_in['passenger_group_id'] = df_in['PassengerId'].str.split('_', expand=True)[1].astype(int)
    return df_in
df = proc_passenger_id(df)
test_df = proc_passenger_id(test_df)

def proc_cabin(df_in):
    df_in['cabin_deck'] = df_in['Cabin'].str.split('/', expand=True)[0]
    df_in['cabin_num'] = df_in['Cabin'].str.split('/', expand=True)[1]
    df_in['cabin_side'] = df_in['Cabin'].str.split('/', expand=True)[2]
    return df_in
df = proc_cabin(df)
test_df = proc_cabin(test_df)

def sep_names(df_in):
    df_in['first_name'] = df_in['Name'].str.split(' ', expand=True)[0]
    df_in['last_name'] = df_in['Name'].str.split(' ', expand=True)[1]
    return df_in
df = sep_names(df)
test_df = sep_names(test_df)
df.head()
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
inferred_lnames = df[df.Name.isna()].apply(lambda x: df[df['passenger_group'] == x['passenger_group']].iloc[0]['last_name'], axis=1)
df.loc[df.Name.isna(), 'last_name'] = inferred_lnames
test_inferred_lnames = test_df[test_df.Name.isna()].apply(lambda x: test_df[test_df['passenger_group'] == x['passenger_group']].iloc[0]['last_name'], axis=1)
test_df.loc[test_df.Name.isna(), 'last_name'] = test_inferred_lnames
df['last_name'] = df['last_name'].fillna('UNK')
df['first_name'] = df['first_name'].fillna('UNK')
test_df['last_name'] = test_df['last_name'].fillna('UNK')
test_df['first_name'] = test_df['first_name'].fillna('UNK')
df.groupby('passenger_group_size').mean()['Age']
import seaborn as sns
import matplotlib.pyplot as plt
(fig, axs) = plt.subplots(1, 2)
df['solo_or_couple'] = df['passenger_group_size'] <= 2
sns.histplot(df[df['solo_or_couple']], x='Age', ax=axs[0], bins=20)
sns.histplot(df[~df['solo_or_couple']], x='Age', ax=axs[1], bins=20)
df[df['solo_or_couple']]['Age'].mode()

def fill_age(df_in):
    df_in.loc[df['solo_or_couple'] & df_in['Age'].isna(), 'Age'] = 24
    return df_in
df = fill_age(df)
test_df = fill_age(test_df)
df['Age'].isna().sum()
sns.histplot(df[~df['solo_or_couple']], x='Age', bins=20)
adult_count = 0
child_count = 0
for (i, row) in df[df['Age'].isna()].iterrows():
    fdf = df[df['passenger_group'] == row.passenger_group]
    if fdf['Age'].min() <= 18:
        df.loc[row.name, 'Age'] = 24
        adult_count += 1
    else:
        df.loc[row.name, 'Age'] = 0
        child_count += 1
print(f'Filled {adult_count} ages with 24, {child_count} ages with 0.')
tdf = df[df.HomePlanet.notna()]
gdf = tdf.groupby(['passenger_group', 'HomePlanet']).count().reset_index()
gdf = gdf[gdf['PassengerId'] > 0]
ggdf = gdf.groupby('passenger_group').count()['HomePlanet']


def get_homeplanet(df_in, row):
    tdf = df_in[df_in.HomePlanet.notna()]
    gdf = tdf.groupby(['passenger_group', 'HomePlanet']).count().reset_index()
    gdf = gdf[gdf['PassengerId'] > 0]
    match = gdf[gdf['passenger_group'] == row['passenger_group']]
    if len(match) > 0:
        return match.iloc[0]['HomePlanet']
    else:
        return None
print(df['HomePlanet'].isna().sum())
df.loc[df.HomePlanet.isna(), 'HomePlanet'] = df.loc[df.HomePlanet.isna()].apply(lambda x: get_homeplanet(df, x), axis=1)
print(df['HomePlanet'].isna().sum())
test_df.loc[test_df.HomePlanet.isna(), 'HomePlanet'] = test_df.loc[test_df.HomePlanet.isna()].apply(lambda x: get_homeplanet(test_df, x), axis=1)
df[df['Cabin'].isna() & (df['passenger_group_size'] > 1)]
inferred_cabins = df[df.Cabin.isna()].apply(lambda x: df[df['passenger_group'] == x['passenger_group']].iloc[0]['Cabin'], axis=1)
df.loc[df.Cabin.isna(), 'Cabin'] = inferred_cabins
test_inferred_cabins = test_df[test_df.Cabin.isna()].apply(lambda x: test_df[test_df['passenger_group'] == x['passenger_group']].iloc[0]['Cabin'], axis=1)
test_df.loc[test_df.Cabin.isna(), 'Cabin'] = test_inferred_cabins
print(sum(df['Cabin'].isna()))
df = proc_cabin(df)
test_df = proc_cabin(test_df)
df['spent_money'] = True
df.loc[df[spend_cols].sum(axis=1) == 0, 'spent_money'] = False
df.groupby(['spent_money', 'CryoSleep']).count()
df.groupby(['Destination', 'HomePlanet', 'CryoSleep']).count()['PassengerId']

def fill_cryo(df_in):
    df_in.loc[df_in.CryoSleep.isna() & (df['spent_money'] == True), 'CryoSleep'] = False
    df_in['CryoSleep'] = df_in['CryoSleep'].fillna(True)
    return df_in
df = fill_cryo(df)
test_df = fill_cryo(test_df)
df.isna().sum()
cats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabin_deck', 'cabin_side']
for c in cats:
    df[c] = pd.Categorical(df[c])
    df[c] = df[c].cat.codes
    test_df[c] = pd.Categorical(test_df[c])
    test_df[c] = test_df[c].cat.codes
modes = df.mode().iloc[0]
df.fillna(modes, inplace=True)
test_df.fillna(modes, inplace=True)
df.isna().sum()
df['passenger_group'] = df['passenger_group'].astype(int)
df['passenger_group_size'] = df['passenger_group_size'].astype(int)
test_df['passenger_group'] = test_df['passenger_group'].astype(int)
test_df['passenger_group_size'] = test_df['passenger_group_size'].astype(int)
df['cabin_num'] = df['cabin_num'].astype(int)
test_df['cabin_num'] = test_df['cabin_num'].astype(int)
(fig, axs) = plt.subplots(1, 2)
sns.histplot(df[df['Transported']], x='cabin_num', ax=axs[0], bins=20, hue='Transported')
sns.histplot(df[~df['Transported']], x='cabin_num', ax=axs[1], bins=20, hue='Transported')
sns.histplot(df, x='cabin_num', binwidth=100, hue='Transported')
sns.histplot(df, x='passenger_group', binwidth=250, hue='Transported')
nbins = 10
df['cabin_num_grp'] = pd.qcut(df['cabin_num'], q=nbins, labels=range(nbins)).astype(int)
test_df['cabin_num_grp'] = pd.qcut(test_df['cabin_num'], q=nbins, labels=range(nbins)).astype(int)
nbins = 5
df['passenger_group_grp'] = pd.qcut(df['passenger_group'], q=nbins, labels=range(nbins)).astype(int)
test_df['passenger_group_grp'] = pd.qcut(test_df['passenger_group'], q=nbins, labels=range(nbins)).astype(int)
df['age_bins'] = df['Age'] // 9
test_df['age_bins'] = df['Age'] // 9
ohe_cols = ['passenger_group_size', 'cabin_num_grp', 'passenger_group_grp', 'age_bins']
ohe_df = pd.get_dummies(df[ohe_cols].astype(str), prefix=ohe_cols)
test_ohe_df = pd.get_dummies(test_df[ohe_cols].astype(str), prefix=ohe_cols)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
yvar = 'Transported'
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
used_cols = cats + spend_cols
proc_df = df[used_cols + [yvar]].join(ohe_df)
proc_test_df = test_df[used_cols].join(test_ohe_df)
(df_train, df_val) = train_test_split(proc_df, train_size=0.75, random_state=0)
X_train = df_train.drop(yvar, axis=1)
y_train = df_train[yvar]
X_val = df_val.drop(yvar, axis=1)
y_val = df_val[yvar]
rfr = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0)