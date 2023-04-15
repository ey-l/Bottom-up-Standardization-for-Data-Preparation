import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
df = pd.read_csv('data/input/spaceship-titanic/train.csv')

def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True)


def total_spent(df):
    df2 = df.copy()
    df2['TotalSpent'] = df2['RoomService'] + df2['FoodCourt'] + df2['ShoppingMall'] + df2['Spa'] + df2['VRDeck']
    return df2

def name_splitter(df):
    df2 = df.copy()
    df2['Name'] = df['Name'].str.split(' ')
    df2['FirstName'] = df2['Name'].str.get(0)
    df2['LastName'] = df2['Name'].str.get(1)
    df2.drop(['Name'], axis=1, inplace=True)
    return df2

def cabin_splitter(df):
    df2 = df.copy()
    df2['Cabin'] = df2['Cabin'].astype('category')
    df2['Deck'] = df2['Cabin'].apply(lambda x: x.split('/')[0])
    df2['CabinNumber'] = df2['Cabin'].apply(lambda x: x.split('/')[1])
    df2['Side'] = df2['Cabin'].apply(lambda x: x.split('/')[2])
    df2.drop(columns=['Cabin'], inplace=True)
    return df2

def has_spent(df):
    df2 = df.copy()
    for i in range(len(df2)):
        if df2.at[i, 'RoomService'] > 0 or df2.at[i, 'FoodCourt'] > 0 or df2.at[i, 'ShoppingMall'] > 0 or (df2.at[i, 'Spa'] > 0) or (df2.at[i, 'VRDeck'] > 0):
            df2.at[i, 'HasSpent'] = True
        else:
            df2.at[i, 'HasSpent'] = False
    return df2
df = name_splitter(df)
df = cabin_splitter(df)
df = total_spent(df)
df = has_spent(df)
plot_correlation_heatmap(df)
sns.histplot(df['Age'])
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
numerical_features
df[categorical_features].describe()
df[numerical_features].describe()
(fig, ax) = plt.subplots(1, 3)
fig.set_figheight(5)
fig.set_figwidth(18)
plt1 = sns.stripplot(x='VIP', y='TotalSpent', data=df, ax=ax[0])
plt2 = sns.stripplot(x='HomePlanet', y='TotalSpent', data=df, ax=ax[1])
plt3 = sns.stripplot(x='Destination', y='TotalSpent', data=df, ax=ax[2])
plt1.set_title('VIP')
plt2.set_title('HomePlanet')
plt3.set_title('Destination')

(fig, ax) = plt.subplots(1, 3)
fig.set_figheight(5)
fig.set_figwidth(15)
plt1 = sns.countplot(x='CryoSleep', data=df, hue='VIP', ax=ax[0])
plt2 = sns.countplot(x='HomePlanet', data=df, hue='VIP', ax=ax[1])
plt3 = sns.countplot(x='Destination', data=df, hue='VIP', ax=ax[2])
plt1.set_title('CryoSleep')
plt2.set_title('HomePlanet')
plt3.set_title('Destination')

sns.countplot(x='HomePlanet', data=df)
sns.barplot(x='CryoSleep', y='TotalSpent', data=df)
(fig, ax) = plt.subplots(2, 3)
fig.set_figheight(10)
fig.set_figwidth(15)
plt1 = sns.boxenplot(x='VIP', y='Age', data=df, ax=ax[0, 0])
plt2 = sns.boxenplot(x='VIP', y='RoomService', data=df, ax=ax[0, 1])
plt3 = sns.boxenplot(x='VIP', y='FoodCourt', data=df, ax=ax[0, 2])
plt4 = sns.boxenplot(x='VIP', y='ShoppingMall', data=df, ax=ax[1, 0])
plt5 = sns.boxenplot(x='VIP', y='Spa', data=df, ax=ax[1, 1])
plt6 = sns.boxenplot(x='VIP', y='VRDeck', data=df, ax=ax[1, 2])
plt1.set_title('Age')
plt2.set_title('RoomService')
plt3.set_title('FoodCourt')
plt4.set_title('ShoppingMall')
plt5.set_title('Spa')
plt6.set_title('VRDeck')
plt2.set_ylim(0, 5000)
plt3.set_ylim(0, 15000)
plt4.set_ylim(0, 5000)
plt5.set_ylim(0, 10000)
plt6.set_ylim(0, 10000)

sns.boxenplot(x='VIP', y='TotalSpent', data=df)
(fig, ax) = plt.subplots(2, 3)
fig.set_figheight(10)
fig.set_figwidth(15)
plt1 = sns.boxenplot(x='HomePlanet', y='Age', data=df, ax=ax[0, 0])
plt2 = sns.boxenplot(x='HomePlanet', y='RoomService', data=df, ax=ax[0, 1])
plt3 = sns.boxenplot(x='HomePlanet', y='FoodCourt', data=df, ax=ax[0, 2])
plt4 = sns.boxenplot(x='HomePlanet', y='ShoppingMall', data=df, ax=ax[1, 0])
plt5 = sns.boxenplot(x='HomePlanet', y='Spa', data=df, ax=ax[1, 1])
plt6 = sns.boxenplot(x='HomePlanet', y='VRDeck', data=df, ax=ax[1, 2])
plt1.set_title('Age')
plt2.set_title('RoomService')
plt3.set_title('FoodCourt')
plt4.set_title('ShoppingMall')
plt5.set_title('Spa')
plt6.set_title('VRDeck')
plt2.set_ylim(0, 5000)
plt3.set_ylim(0, 15000)
plt4.set_ylim(0, 5000)
plt5.set_ylim(0, 10000)
plt6.set_ylim(0, 10000)

(fig, ax) = plt.subplots(2, 3)
fig.set_figheight(10)
fig.set_figwidth(15)
plt1 = sns.boxenplot(x='Destination', y='Age', data=df, ax=ax[0, 0], hue='VIP')
plt2 = sns.boxenplot(x='Destination', y='RoomService', data=df, ax=ax[0, 1], hue='VIP')
plt3 = sns.boxenplot(x='Destination', y='FoodCourt', data=df, ax=ax[0, 2], hue='VIP')
plt4 = sns.boxenplot(x='Destination', y='ShoppingMall', data=df, ax=ax[1, 0], hue='VIP')
plt5 = sns.boxenplot(x='Destination', y='Spa', data=df, ax=ax[1, 1], hue='VIP')
plt6 = sns.boxenplot(x='Destination', y='VRDeck', data=df, ax=ax[1, 2], hue='VIP')
plt1.set_title('Age')
plt2.set_title('RoomService')
plt3.set_title('FoodCourt')
plt4.set_title('ShoppingMall')
plt5.set_title('Spa')
plt6.set_title('VRDeck')
plt2.set_ylim(0, 5000)
plt3.set_ylim(0, 15000)
plt4.set_ylim(0, 5000)
plt5.set_ylim(0, 10000)
plt6.set_ylim(0, 10000)

sns.countplot(x='HomePlanet', data=df, hue='Destination')

(fig, ax) = plt.subplots(1, 2)
fig.set_figheight(5)
fig.set_figwidth(15)
plt1 = sns.countplot(x='Deck', data=df, hue='HomePlanet', ax=ax[0])
plt3 = sns.countplot(x='Side', data=df, hue='HomePlanet', ax=ax[1])
plt1.set_title('Deck')
plt3.set_title('Side')

sns.countplot(x='CryoSleep', data=df, hue='HomePlanet')
sns.boxenplot(x='HomePlanet', y='TotalSpent', data=df)
(fig, ax) = plt.subplots(1, 2)
fig.set_figheight(5)
fig.set_figwidth(15)
plt1 = sns.countplot(x='Deck', data=df, hue='CryoSleep', ax=ax[0])
plt3 = sns.countplot(x='Side', data=df, hue='CryoSleep', ax=ax[1])
plt1.set_title('Deck')
plt3.set_title('Side')

plt.figure(figsize=(20, 15), dpi=200)
sns.catplot(data=df, x='Deck', hue='CryoSleep', col='Destination', kind='count')
plt.figure(figsize=(20, 15), dpi=200)
sns.catplot(data=df, x='HasSpent', hue='CryoSleep', col='Destination', kind='count')
plt.figure(figsize=(20, 15), dpi=200)
sns.catplot(data=df, x='Deck', hue='CryoSleep', col='HomePlanet', kind='count')