import numpy as np
import pandas as pd
import plotly.express as px
spaceship_file_path = 'data/input/spaceship-titanic/train.csv'
spaceship_data = pd.read_csv(spaceship_file_path)
test_file_path = 'data/input/spaceship-titanic/test.csv'
test_data = pd.read_csv(test_file_path)


spaceship_data.info
print('The number of variables in the original data:')
spaceship_data.columns
spaceship_data.dtypes
spaceship_data.isnull().sum()
spaceship_data.duplicated().sum()
spaceship_data.isnull().sum().sum() / spaceship_data.shape[0]
for col in spaceship_data.columns:
    if spaceship_data[col].dtype == object:
        print(col, '\x08:')
        print(spaceship_data[col].value_counts(dropna=False))
for col in spaceship_data.columns:
    if spaceship_data[col].dtype != object:
        print(col, '\x08:')
        print(spaceship_data[col].value_counts(dropna=False))
import matplotlib.pyplot as plt
y = spaceship_data.Transported
plt.figure(figsize=(7, 7))
y.value_counts().plot.pie(autopct='%2.1f%%', shadow=True, textprops={'fontsize': 18}).set_title('Transported distribution')
HomePlanet_ct = spaceship_data.groupby(['HomePlanet']).count()
HomePlanet_ct
plt.figure(figsize=(7, 7))
HomePlanet_ct['PassengerId'].plot.pie(autopct='%2.1f%%', textprops={'fontsize': 18}, title='Passengers distribution')
spaceship_data.dropna(axis=0, inplace=True)
px.box(spaceship_data, y='Age', color='Transported', points='all')
px.histogram(spaceship_data, x='Age', color='Transported', title='Histogram of Age')
spaceship_data['Age'].describe()
Destination_ct = spaceship_data.groupby(['Destination']).count()
Destination_ct
plt.figure(figsize=(7, 7))
Destination_ct['PassengerId'].plot.pie(autopct='%2.1f%%', textprops={'fontsize': 18}, title='Passengers distribution in terms of destination')
px.histogram(spaceship_data, x='Destination', color='Transported')
Destination_tran_ct = spaceship_data.groupby(['Destination', 'Transported']).count()
Destination_tran_ct

def split_cabin(df):
    newcols = df['Cabin'].str.split('/', expand=True)
    newcols.index = df.index
    df['Deck'] = newcols.iloc[:, 0]
    df['Side'] = newcols.iloc[:, 2]

def add_groupid(df):
    splitdf = df['PassengerId'].str.split('_', expand=True)
    df['GroupId'] = splitdf.iloc[:, 0]

def add_groupsize(df):
    grpsizes = df.groupby('GroupId').size()
    newcol = grpsizes[df['GroupId']]
    newcol.index = df.index
    df['GroupSize'] = newcol.astype(float)

def preprocess(df):
    split_cabin(df)
    add_groupid(df)
    add_groupsize(df)
preprocess(spaceship_data)
spaceship_data.head()
px.histogram(spaceship_data, x='Deck', color='Transported')
px.histogram(spaceship_data, x='Side', color='Transported')
px.histogram(spaceship_data, x='GroupSize', color='Transported')
px.histogram(spaceship_data, x='VIP', color='Transported')
VIP_tran_ct = spaceship_data.groupby(['VIP', 'Transported']).count()
VIP_tran_ct
px.histogram(spaceship_data, x='CryoSleep', color='Transported')
CryoSleep_tran_ct = spaceship_data.groupby(['CryoSleep', 'Transported']).count()
CryoSleep_tran_ct