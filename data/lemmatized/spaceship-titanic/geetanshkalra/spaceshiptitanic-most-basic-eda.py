import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(20)
_input1.isnull().sum()
plt.figure(figsize=(15, 5))
ax = sns.countplot(x='HomePlanet', hue='Transported', data=_input1)
for i in ax.containers:
    ax.bar_label(i)
plt.figure(figsize=(15, 5))
ax = sns.countplot(x='CryoSleep', hue='Transported', data=_input1)
for i in ax.containers:
    ax.bar_label(i)
plt.figure(figsize=(15, 5))
ax = sns.countplot(x='Destination', hue='Transported', data=_input1)
for i in ax.containers:
    ax.bar_label(i)
plt.figure(figsize=(15, 5))
ax = sns.countplot(x='VIP', hue='Transported', data=_input1)
for i in ax.containers:
    ax.bar_label(i)
bins = [0, 1, 5, 10, 25, 50, 100]
labels = ['0-1', '1-5', '5-10', '10-25', '25-50', '50-100']
df_copy = _input1.copy()
df_copy['age_distribuition'] = pd.cut(_input1['Age'], bins=bins, labels=labels)
plt.figure(figsize=(15, 5))
ax = sns.countplot(x='age_distribuition', hue='Transported', data=df_copy)
for i in ax.containers:
    ax.bar_label(i)
_input1['RoomService'].max(axis=0)
_input1['FoodCourt'].max(axis=0)
_input1['ShoppingMall'].max(axis=0)
_input1['Spa'].max(axis=0)
_input1['VRDeck'].max(axis=0)

def make_bins(value):
    value_increment = 5000
    total_bins = int(value / value_increment)
    bins_list = []
    inital_value = 0
    for i in range(0, total_bins + 2):
        bins_list.append(inital_value)
        inital_value += value_increment
    return bins_list
column_name_list = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for column_name in column_name_list:
    bins = make_bins(_input1[column_name].max(axis=0))
    df_copy = _input1.copy()
    df_copy[f'{column_name}_distribution'] = pd.cut(_input1[column_name], bins=bins)
    plt.figure(figsize=(25, 7))
    ax = sns.countplot(x=f'{column_name}_distribution', hue='Transported', data=df_copy)
    for i in ax.containers:
        ax.bar_label(i)
_input1['Cabin'] = _input1['Cabin'].fillna('Not_Found', inplace=False)
cabin_deck = []
cabin_num = []
cabin_port = []
for deck in _input1['Cabin']:
    if deck != 'Not_Found':
        list_of_deck = deck.split('/')
        cabin_deck.append(list_of_deck[0])
        cabin_num.append(int(list_of_deck[1]))
        cabin_port.append(list_of_deck[2])
    else:
        cabin_deck.append('Not_Found')
        cabin_num.append('Not_Found')
        cabin_port.append('Not_Found')
_input1['Cabin_deck'] = cabin_deck
_input1['Cabin_num'] = cabin_num
_input1['Cabin_port'] = cabin_port
plt.figure(figsize=(15, 5))
ax = sns.countplot(x='Cabin_deck', hue='Transported', data=_input1)
for i in ax.containers:
    ax.bar_label(i)
plt.figure(figsize=(15, 5))
ax = sns.countplot(x='Cabin_port', hue='Transported', data=_input1)
for i in ax.containers:
    ax.bar_label(i)
passenger_group = []
passenger_num = []
for passenger in _input1['PassengerId']:
    if passenger != 'Not_Found':
        list_of_passenger = passenger.split('_')
        passenger_group.append(list_of_passenger[0])
        passenger_num.append(list_of_passenger[1])
    else:
        passenger_group.append('Not_Found')
        passenger_num.append('Not_Found')
_input1['Passenger_group'] = passenger_group
_input1['Passenger_num'] = passenger_num
_input1.head()