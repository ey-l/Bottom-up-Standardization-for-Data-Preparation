import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
sns.set_theme(style='whitegrid')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
data_cleaner = [train, test]

train.head()
for dataset in data_cleaner:
    for column in dataset.columns:
        print('{} has {} null values'.format(column, dataset[column].isnull().sum()))
missing = train.isnull().sum().sort_values(ascending=False)
sns.barplot(y=missing.index, x=missing.values)
plt.xticks(rotation=0)

categoricalFeatures = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
target = ['Transported']
numericFeatures = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, ax) = plt.subplots(2, 2, figsize=[8, 6])
for (idx, column) in enumerate(categoricalFeatures):
    plt.subplot(2, 2, idx + 1)
    sns.countplot(data=train, x=column)
    fig.tight_layout()

(fig, ax) = plt.subplots(3, 2, figsize=[8, 6], sharey=False)
ax = ax.flatten()
for (idx, column) in enumerate(numericFeatures):
    sns.histplot(data=train, x=column, kde=False, ax=ax[idx], bins=10)
    fig.tight_layout()

(fig, ax) = plt.subplots(2, 2, figsize=[8, 6])
for (idx, column) in enumerate(categoricalFeatures):
    plt.subplot(2, 2, idx + 1)
    sns.barplot(data=train, x=column, y='Transported')
    fig.tight_layout()

plt.subplots(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, fmt='.1g', cmap=sns.diverging_palette(20, 220, n=200), linewidths=0.5, linecolor='grey')
