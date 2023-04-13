import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
warnings.filterwarnings(action='ignore')
submission_id = _input0['PassengerId']
submission_id
print('train_shape : ', _input1.shape)
df = pd.merge(_input1, _input0, how='outer')
full_index = df['PassengerId']
df.info()
df.head()
for _ in df.columns.tolist():
    if df[_].dtype == 'object':
        print(df[_].value_counts())
        print('*' * 40)
df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0})
df['VIP'] = df['VIP'].map({True: 1, False: 0})
df['Transported'] = df['Transported'].map({True: 1, False: 0})
df
df['Cabin']
df['cabin'] = df['Cabin'].str[0]
df['side'] = df['Cabin'].str[-1]
df = df.drop('Cabin', axis=1)
df
df['Name'] = df['Name'].str.split(' ').str[1]
df.head()
df['id'] = [i[0] for i in df['PassengerId'].str.split('_')]
df['member'] = [i[1] for i in df['PassengerId'].str.split('_')]
gggg = df.groupby('id').count()['member'].index.tolist()
pp = df.groupby('id').count()['member'].values.tolist()
df['FamilySize'] = 0
for (ID, member) in zip(gggg, pp):
    df.loc[df['id'] == ID, 'FamilySize'] = member
df
df = df.drop(['PassengerId', 'id', 'member'], axis=1)
df
df.isna().sum()
plt.figure(figsize=[7, 7])
sns.set_palette('Set3')
sns.set_style('darkgrid')
sns.countplot(data=df, x='HomePlanet')
plt.title('Countplot of HomePlanet')
df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
df.isna().sum()
plt.figure(figsize=[7, 10])
sns.countplot(data=df, x='Destination')
plt.title('Countplot of Destination')
df['Destination'] = df['Destination'].fillna('TRAPPIST-1e')
df.isna().sum()
plt.figure(figsize=[7, 7])
sns.countplot(data=df, x='VIP')
plt.title('Countplot of VIP')
df['VIP'] = df['VIP'].fillna(0)
df.isna().sum()
sns.kdeplot(data=df, x='Age', hue='VIP', multiple='stack')
plt.axvline(x=df['Age'].mean(), ls='--', c='dodgerblue', label='mean')
plt.axvline(x=df['Age'].median(), ls='--', c='purple', label='median')
plt.legend()
plt.title('Kdeplot of Age according to VIP')
plt.figure(figsize=[7, 7])
sns.violinplot(data=df, x='VIP', y='Age')
plt.axhline(y=df.loc[df['VIP'] == 0]['Age'].median(), ls='--', c='dodgerblue', label='Not VIP')
plt.axhline(y=df.loc[df['VIP'] == 1]['Age'].median(), ls='--', c='purple', label='VIP')
plt.legend(prop={'size': 15})
plt.title('median age according to VIP presence')
df.loc[df['VIP'] == 0, 'Age'] = df.loc[df['VIP'] == 0, 'Age'].fillna(df.loc[df['VIP'] == 0, 'Age'].median())
df.loc[df['VIP'] == 1, 'Age'] = df.loc[df['VIP'] == 1, 'Age'].fillna(df.loc[df['VIP'] == 1, 'Age'].median())
df.isna().sum()
numeric_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
plt.figure(figsize=[15, 15])
plt.subplots_adjust(wspace=1)
for (i, var) in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)
    sns.barplot(data=df, x='VIP', y=var)
    plt.axhline(y=df.loc[df['VIP'] == 0][var].mean(), ls='--', c='dodgerblue', label='Not VIP')
    plt.axhline(y=df.loc[df['VIP'] == 1][var].mean(), ls='--', c='purple', label='VIP')
    plt.legend(loc=9)
    plt.title(f'violineplot of {var} according to VIP')
for var in numeric_cols:
    df.loc[df['VIP'] == 0, var] = df.loc[df['VIP'] == 0, var].fillna(df.loc[df['VIP'] == 0, var].mean())
    df.loc[df['VIP'] == 1, var] = df.loc[df['VIP'] == 1, var].fillna(df.loc[df['VIP'] == 1, var].mean())
df.isna().sum()
plt.figure(figsize=[7, 7])
sns.countplot(data=df, x='VIP', hue='CryoSleep')
plt.title('Countplot of CyroSleep according to VIP')
df['CryoSleep'] = df['CryoSleep'].fillna(0)
df.isna().sum()
obj_cols = ['cabin', 'side']
plt.subplots_adjust(hspace=1)
plt.figure(figsize=[10, 10])
for (i, var) in enumerate(obj_cols):
    plt.subplot(2, 1, i + 1)
    sns.countplot(data=df, x=var, hue='HomePlanet')
    plt.title(f'Countplot of {var} according to HomePlanet')
na_cond = df['cabin'].isna()
df.loc[na_cond & (df['HomePlanet'] == 'Earth'), 'cabin'] = df[na_cond & (df['HomePlanet'] == 'Earth')].fillna('G')
df.loc[na_cond & (df['HomePlanet'] == 'Europa'), 'cabin'] = df[na_cond & (df['HomePlanet'] == 'Europa')].fillna('B')
df.loc[na_cond & (df['HomePlanet'] == 'Mars'), 'cabin'] = df[na_cond & (df['HomePlanet'] == 'Mars')].fillna('F')
df.isna().sum()
df.loc[df['side'].isna(), 'side'][:150] = df.loc[df['side'].isna(), 'side'][:150].fillna('P')
df.loc[df['side'].isna(), 'side'] = df.loc[df['side'].isna(), 'side'].fillna('S')
df.isna().sum()
df['Name'] = df['Name'].fillna('Unknown')
df.isna().sum()
name_count = df['Name'].value_counts()
Rare_name = name_count[name_count.values < name_count.mean()].index.tolist()
uncommon_name = name_count[(name_count.mean() < name_count.values) & (name_count.values <= 8)].index.tolist()
common_name = name_count[(8 < name_count.values) & (name_count.index != 'Unknown')].index.tolist()
unknown_name = name_count[name_count.index == 'Unknown'].index.tolist()
df.loc[df['Name'].isin(common_name), 'Name'] = 0
df.loc[df['Name'].isin(uncommon_name), 'Name'] = 1
df.loc[df['Name'].isin(Rare_name), 'Name'] = 2
df.loc[df['Name'].isin(unknown_name), 'Name'] = 3
df.head()
df['passenger_id'] = full_index
_input0 = df.loc[df['passenger_id'].isin(submission_id)].drop('passenger_id', axis=1)
_input1 = df.drop(index=_input0.index, axis=0).drop('passenger_id', axis=1)
plt.figure(figsize=[10, 5])
sns.kdeplot(data=_input1, x='Age', hue='Transported', palette=['black', 'purple'], multiple='stack')
plt.title('Kdeplot of Age according to Transported')
obj_cols = ['HomePlanet', 'Destination', 'cabin', 'side', 'FamilySize', 'Name']
plt.figure(figsize=[10, 16])
plt.subplots_adjust(hspace=0.5)
for (i, var) in enumerate(obj_cols):
    plt.subplot(6, 1, i + 1)
    sns.countplot(data=_input1, x=var, hue='Transported', palette=['black', 'purple'])
    plt.title(f'Countplot of {var} according to Transported')
numeric_cols = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numeric_cols
plt.figure(figsize=[10, 16])
plt.subplots_adjust(hspace=0.5)
for (i, var) in enumerate(numeric_cols):
    plt.subplot(4, 2, i + 1)
    sns.histplot(data=_input1, x=var, hue='Transported', palette=['black', 'purple'], kde=True, multiple='dodge', bins=50)
    plt.title(f'histplot of {var} according to Transported')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score

def encoder(df):
    obj_cols = df.describe(include='O').columns.tolist()
    for (i, var) in enumerate(obj_cols):
        dummy = df[var].unique()
        num = list(range(len(dummy)))
        for _ in range(len(dummy)):
            df.loc[df[var] == dummy[_], var] = num[_]
    return df
_input1 = encoder(_input1)
_input1
cv = KFold(n_splits=10)
max_depth = 20
avg_score = []
for depth in range(1, max_depth + 1):
    score = []
    model = tree.DecisionTreeClassifier(max_depth=depth)
    for (v_train, v_test) in cv.split(_input1):
        v_train_feature = _input1.iloc[v_train].drop('Transported', axis=1)
        v_train_target = _input1.iloc[v_train]['Transported']
        v_test_feature = _input1.iloc[v_test].drop('Transported', axis=1)
        v_test_target = _input1.iloc[v_test]['Transported']