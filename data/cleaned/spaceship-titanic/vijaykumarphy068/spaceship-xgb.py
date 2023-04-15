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

train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df
test_df
from pandas_profiling import ProfileReport
ProfileReport(train_df, title='Pandas Profiling Report')
ProfileReport(test_df, title='Pandas Profiling Report')
train_df.columns
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=4, cols=1)
for (i, col) in enumerate(cat_cols):
    train_cat_col = train_df[col].value_counts().to_frame() / len(train_df)
    test_cat_col = test_df[col].value_counts().to_frame() / len(test_df)
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
train_df.columns
train_df.describe()
test_df.describe()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
('train:', train_df[num_cols].isnull().sum() / len(train_df[num_cols]), '       ', 'test:', test_df[num_cols].isnull().sum() / len(test_df[num_cols]))
train_df[(train_df['CryoSleep'] == True) & (train_df['ShoppingMall'] > 0)]
train_df['RoomService'].fillna(0, inplace=True)
train_df['FoodCourt'].fillna(0, inplace=True)
train_df['ShoppingMall'].fillna(0, inplace=True)
train_df['VRDeck'].fillna(0, inplace=True)
train_df['Spa'].fillna(0, inplace=True)
test_df['RoomService'].fillna(0, inplace=True)
test_df['FoodCourt'].fillna(0, inplace=True)
test_df['ShoppingMall'].fillna(0, inplace=True)
test_df['VRDeck'].fillna(0, inplace=True)
test_df['Spa'].fillna(0, inplace=True)
sns.histplot(data=train_df['Age'], kde=True, bins=40)
sns.histplot(data=test_df['Age'], kde=True, bins=40)

sns.distplot(train_df['Age'], kde=True, bins=40)
sns.distplot(test_df['Age'], kde=True, bins=40)

train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
(fig, axs) = plt.subplots(nrows=5, ncols=1, figsize=(15, 25))
hist_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'deeppink', 'plum', 'peru']
plt.figure(figsize=(1000, 600))
for (i, col) in enumerate(num_cols[1:]):
    train_num_col = np.log(train_df[train_df[col] > 0][col])
    test_num_col = np.log(test_df[test_df[col] > 0][col])
    axs[i].hist(train_num_col, edgecolor='black', color=hist_color[i], label='train ')
    axs[i].legend(loc='upper right')
    axs[i].hist(test_num_col, edgecolor='black', color=hist_color[i + 5], label='test')
    axs[i].legend(loc='upper right')
    axs[i].title.set_text(col)
for (i, col) in enumerate(cat_cols):
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Deck'] = train_df['Cabin'].str.split('/').str[0]
train_df['Num'] = train_df['Cabin'].str.split('/').str[1]
train_df['Side'] = train_df['Cabin'].str.split('/').str[2]
test_df['Deck'] = test_df['Cabin'].str.split('/').str[0]
test_df['Num'] = test_df['Cabin'].str.split('/').str[1]
test_df['Side'] = test_df['Cabin'].str.split('/').str[2]
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)
train_df.isnull().sum()
train_df['Deck'].value_counts(dropna=False) / len(train_df)
test_df['Deck'].value_counts(dropna=False) / len(test_df)
train_df['Side'].value_counts(dropna=False) / len(train_df)
X_train = train_df.loc[:, train_df.columns != 'Transported']
Y_train = train_df['Transported']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
(fig, axes) = plt.subplots(nrows=6, ncols=1, figsize=(15, 35))
for (i, col) in enumerate(cat_cols):
    sns.countplot(x=col, hue='Transported', data=train_df, ax=axes[i], palette='dark', saturation=0.5, order=train_df[col].value_counts().index)
Y_train = Y_train.map({False: 0, True: 1})
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df = np.log(X_train[num_cols])
df['Transported'] = Y_train.copy()
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, vmin=-1)
(train_df.isnull().sum(), test_df.isnull().sum())
train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)
train_df
train_df.groupby('Deck')['CryoSleep'].sum().sort_values(ascending=False)
deck_planet = train_df.groupby('HomePlanet')['Deck'].value_counts().reset_index(name='Counts').sort_values('Counts', ascending=False)
deck_planet
sns.barplot(data=deck_planet, x='Deck', y='Counts', hue='HomePlanet')
import random
europa_deck = ['A', 'B', 'C', 'D', 'E']
mars_deck = ['D', 'E', 'F']
earth_deck = ['E', 'F', 'G']
mask_1_train = train_df['HomePlanet'] == 'Europa'
mask_2_train = train_df['HomePlanet'] == 'Mars'
mask_3_train = train_df['HomePlanet'] == 'Earth'
train_df.loc[mask_1_train, 'Deck'] = train_df.loc[mask_1_train, 'Deck'].fillna(random.choice(europa_deck))
train_df.loc[mask_2_train, 'Deck'] = train_df.loc[mask_2_train, 'Deck'].fillna(random.choice(mars_deck))
train_df.loc[mask_3_train, 'Deck'] = train_df.loc[mask_3_train, 'Deck'].fillna(random.choice(earth_deck))
mask_1_test = test_df['HomePlanet'] == 'Europa'
mask_2_test = test_df['HomePlanet'] == 'Mars'
mask_3_test = test_df['HomePlanet'] == 'Earth'
test_df.loc[mask_1_test, 'Deck'] = test_df.loc[mask_1_test, 'Deck'].fillna(random.choice(europa_deck))
test_df.loc[mask_2_test, 'Deck'] = test_df.loc[mask_2_test, 'Deck'].fillna(random.choice(mars_deck))
test_df.loc[mask_3_test, 'Deck'] = test_df.loc[mask_3_test, 'Deck'].fillna(random.choice(earth_deck))
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Num'] = train_df[train_df['Num'].notnull()].Num.astype(int)
test_df['Num'] = test_df[test_df['Num'].notnull()].Num.astype(int)
sns.barplot(data=train_df, x='Deck', y='Num', hue='Transported')
train_df[(train_df['Deck'] == 'A') & train_df['Num']]['Num'].describe()
train_df[(train_df['Deck'] == 'B') & train_df['Num']]['Num'].describe()
train_df[(train_df['Deck'] == 'C') & train_df['Num']]['Num'].describe()
train_df[(train_df['Deck'] == 'D') & train_df['Num']]['Num'].describe()
train_df[(train_df['Deck'] == 'E') & train_df['Num']]['Num'].describe()
train_df[(train_df['Deck'] == 'F') & train_df['Num']]['Num'].describe()
train_df[(train_df['Deck'] == 'T') & train_df['Num']]['Num'].describe()
train_df['Num'] = train_df['Num'].fillna(train_df['Num'].median())
train_df['Deck'] = train_df['Deck'].fillna(train_df['Deck'].mode()[0])
train_df['Side'] = train_df['Side'].fillna(train_df['Side'].mode()[0])
test_df['Num'] = test_df['Num'].fillna(test_df['Num'].median())
test_df['Deck'] = test_df['Deck'].fillna(test_df['Deck'].mode()[0])
test_df['Side'] = test_df['Side'].fillna(test_df['Side'].mode()[0])
test_df.isnull().sum()
train_df.isnull().sum()
train_df.info()
train_df['group_id'] = train_df['PassengerId'].str.split('_').str[0]
train_df['group_num'] = train_df['PassengerId'].str.split('_').str[1]
test_df['group_id'] = test_df['PassengerId'].str.split('_').str[0]
test_df['group_num'] = test_df['PassengerId'].str.split('_').str[1]
train_df.info()
train_df['group_id'] = train_df['group_id'].astype(int)
train_df['group_num'] = train_df['group_num'].astype(int)
test_df['group_id'] = test_df['group_id'].astype(int)
test_df['group_num'] = test_df['group_num'].astype(int)
train_df[['HomePlanet', 'Destination', 'Deck', 'Side']]
x_cat = pd.get_dummies(train_df[['HomePlanet', 'Destination', 'Deck', 'Side']])
x_test_cat = pd.get_dummies(test_df[['HomePlanet', 'Destination', 'Deck', 'Side']])
x = pd.concat([x_cat, train_df], axis=1)
x_test = pd.concat([x_test_cat, test_df], axis=1)
y = x['Transported']
x.columns
x.drop(columns=['PassengerId', 'HomePlanet', 'Destination', 'Transported', 'Deck', 'Side'], axis=1, inplace=True)
x_test.drop(columns=['PassengerId', 'HomePlanet', 'Destination', 'Deck', 'Side'], axis=1, inplace=True)
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