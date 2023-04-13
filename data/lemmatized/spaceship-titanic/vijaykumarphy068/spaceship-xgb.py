import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1
_input0
from pandas_profiling import ProfileReport
ProfileReport(_input1, title='Pandas Profiling Report')
ProfileReport(_input0, title='Pandas Profiling Report')
_input1.columns
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1)
for (i, col) in enumerate(cat_cols):
    train_cat_col = _input1[col].value_counts().to_frame() / len(_input1)
    test_cat_col = _input0[col].value_counts().to_frame() / len(_input0)
    fig.add_trace(go.Bar(x=train_cat_col.index, y=train_cat_col[col], name='train' + ' ' + col, legendgroup=i), row=i + 1, col=1)
    fig.add_annotation(xref='x domain', yref='y domain', x=0.5, y=1.2, text=col, row=i + 1, col=1)
    fig.add_trace(go.Bar(x=test_cat_col.index, y=test_cat_col[col], name='test' + ' ' + col, legendgroup=i), row=i + 1, col=1)
    fig.add_annotation(xref='x domain', yref='y domain', x=0.5, y=1.2, text=col, row=i + 1, col=1)
fig.update_layout(height=1600, width=1000, title_text='plots of categorial data from train and test', legend_tracegroupgap=340)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set(style='darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (15, 9)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
_input1.columns
_input1.describe()
_input0.describe()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
('train:', _input1[num_cols].isnull().sum() / len(_input1[num_cols]), '       ', 'test:', _input0[num_cols].isnull().sum() / len(_input0[num_cols]))
_input1[(_input1['CryoSleep'] == True) & (_input1['ShoppingMall'] > 0)]
_input1['RoomService'] = _input1['RoomService'].fillna(0, inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0, inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0, inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0, inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(0, inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(0, inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(0, inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(0, inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(0, inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(0, inplace=False)
sns.histplot(data=_input1['Age'], kde=True, bins=40)
sns.histplot(data=_input0['Age'], kde=True, bins=40)
sns.distplot(_input1['Age'], kde=True, bins=40)
sns.distplot(_input0['Age'], kde=True, bins=40)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input1['Age'].mean(), inplace=False)
(fig, axs) = plt.subplots(nrows=5, ncols=1, figsize=(15, 25))
hist_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'deeppink', 'plum', 'peru']
plt.figure(figsize=(1000, 600))
for (i, col) in enumerate(num_cols[1:]):
    train_num_col = np.log(_input1[_input1[col] > 0][col])
    test_num_col = np.log(_input0[_input0[col] > 0][col])
    axs[i].hist(train_num_col, edgecolor='black', color=hist_color[i], label='train ')
    axs[i].legend(loc='upper right')
    axs[i].hist(test_num_col, edgecolor='black', color=hist_color[i + 5], label='test')
    axs[i].legend(loc='upper right')
    axs[i].title.set_text(col)
for (i, col) in enumerate(cat_cols):
    _input1[col] = _input1[col].fillna(_input1[col].mode()[0], inplace=False)
    _input0[col] = _input0[col].fillna(_input0[col].mode()[0], inplace=False)
_input1.isnull().sum()
_input0.isnull().sum()
_input1['Deck'] = _input1['Cabin'].str.split('/').str[0]
_input1['Num'] = _input1['Cabin'].str.split('/').str[1]
_input1['Side'] = _input1['Cabin'].str.split('/').str[2]
_input0['Deck'] = _input0['Cabin'].str.split('/').str[0]
_input0['Num'] = _input0['Cabin'].str.split('/').str[1]
_input0['Side'] = _input0['Cabin'].str.split('/').str[2]
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input1.isnull().sum()
_input1['Deck'].value_counts(dropna=False) / len(_input1)
_input0['Deck'].value_counts(dropna=False) / len(_input0)
_input1['Side'].value_counts(dropna=False) / len(_input1)
X_train = _input1.loc[:, _input1.columns != 'Transported']
Y_train = _input1['Transported']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
(fig, axes) = plt.subplots(nrows=6, ncols=1, figsize=(15, 35))
for (i, col) in enumerate(cat_cols):
    sns.countplot(x=col, hue='Transported', data=_input1, ax=axes[i], palette='dark', saturation=0.5, order=_input1[col].value_counts().index)
Y_train = Y_train.map({False: 0, True: 1})
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df = np.log(X_train[num_cols])
df['Transported'] = Y_train.copy()
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, vmin=-1)
(_input1.isnull().sum(), _input0.isnull().sum())
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input1
_input1.groupby('Deck')['CryoSleep'].sum().sort_values(ascending=False)
deck_planet = _input1.groupby('HomePlanet')['Deck'].value_counts().reset_index(name='Counts').sort_values('Counts', ascending=False)
deck_planet
sns.barplot(data=deck_planet, x='Deck', y='Counts', hue='HomePlanet')
import random
europa_deck = ['A', 'B', 'C', 'D', 'E']
mars_deck = ['D', 'E', 'F']
earth_deck = ['E', 'F', 'G']
mask_1_train = _input1['HomePlanet'] == 'Europa'
mask_2_train = _input1['HomePlanet'] == 'Mars'
mask_3_train = _input1['HomePlanet'] == 'Earth'
_input1.loc[mask_1_train, 'Deck'] = _input1.loc[mask_1_train, 'Deck'].fillna(random.choice(europa_deck))
_input1.loc[mask_2_train, 'Deck'] = _input1.loc[mask_2_train, 'Deck'].fillna(random.choice(mars_deck))
_input1.loc[mask_3_train, 'Deck'] = _input1.loc[mask_3_train, 'Deck'].fillna(random.choice(earth_deck))
mask_1_test = _input0['HomePlanet'] == 'Europa'
mask_2_test = _input0['HomePlanet'] == 'Mars'
mask_3_test = _input0['HomePlanet'] == 'Earth'
_input0.loc[mask_1_test, 'Deck'] = _input0.loc[mask_1_test, 'Deck'].fillna(random.choice(europa_deck))
_input0.loc[mask_2_test, 'Deck'] = _input0.loc[mask_2_test, 'Deck'].fillna(random.choice(mars_deck))
_input0.loc[mask_3_test, 'Deck'] = _input0.loc[mask_3_test, 'Deck'].fillna(random.choice(earth_deck))
_input1.isnull().sum()
_input0.isnull().sum()
_input1['Num'] = _input1[_input1['Num'].notnull()].Num.astype(int)
_input0['Num'] = _input0[_input0['Num'].notnull()].Num.astype(int)
sns.barplot(data=_input1, x='Deck', y='Num', hue='Transported')
_input1[(_input1['Deck'] == 'A') & _input1['Num']]['Num'].describe()
_input1[(_input1['Deck'] == 'B') & _input1['Num']]['Num'].describe()
_input1[(_input1['Deck'] == 'C') & _input1['Num']]['Num'].describe()
_input1[(_input1['Deck'] == 'D') & _input1['Num']]['Num'].describe()
_input1[(_input1['Deck'] == 'E') & _input1['Num']]['Num'].describe()
_input1[(_input1['Deck'] == 'F') & _input1['Num']]['Num'].describe()
_input1[(_input1['Deck'] == 'T') & _input1['Num']]['Num'].describe()
_input1['Num'] = _input1['Num'].fillna(_input1['Num'].median())
_input1['Deck'] = _input1['Deck'].fillna(_input1['Deck'].mode()[0])
_input1['Side'] = _input1['Side'].fillna(_input1['Side'].mode()[0])
_input0['Num'] = _input0['Num'].fillna(_input0['Num'].median())
_input0['Deck'] = _input0['Deck'].fillna(_input0['Deck'].mode()[0])
_input0['Side'] = _input0['Side'].fillna(_input0['Side'].mode()[0])
_input0.isnull().sum()
_input1.isnull().sum()
_input1.info()
_input1['group_id'] = _input1['PassengerId'].str.split('_').str[0]
_input1['group_num'] = _input1['PassengerId'].str.split('_').str[1]
_input0['group_id'] = _input0['PassengerId'].str.split('_').str[0]
_input0['group_num'] = _input0['PassengerId'].str.split('_').str[1]
_input1.info()
_input1['group_id'] = _input1['group_id'].astype(int)
_input1['group_num'] = _input1['group_num'].astype(int)
_input0['group_id'] = _input0['group_id'].astype(int)
_input0['group_num'] = _input0['group_num'].astype(int)
_input1[['HomePlanet', 'Destination', 'Deck', 'Side']]
x_cat = pd.get_dummies(_input1[['HomePlanet', 'Destination', 'Deck', 'Side']])
x_test_cat = pd.get_dummies(_input0[['HomePlanet', 'Destination', 'Deck', 'Side']])
x = pd.concat([x_cat, _input1], axis=1)
x_test = pd.concat([x_test_cat, _input0], axis=1)
y = x['Transported']
x.columns
x = x.drop(columns=['PassengerId', 'HomePlanet', 'Destination', 'Transported', 'Deck', 'Side'], axis=1, inplace=False)
x_test = x_test.drop(columns=['PassengerId', 'HomePlanet', 'Destination', 'Deck', 'Side'], axis=1, inplace=False)
print(x.shape)
print(y.shape)
print(x_test.shape)
x_test.info()
from sklearn.model_selection import train_test_split
(x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=1)
print(x_train.shape)
print(x_val.shape)
x.info()
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
xgbc_tuned = XGBClassifier(gamma=5, subsample=1, max_depth=2, colsample_bytree=1, n_estimators=70)