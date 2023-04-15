import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
train_df
test_df
train_df.info()
test_df.info()
missingno.matrix(train_df)
missingno.matrix(test_df)
train_df.info()
train_df = train_df.astype({'Transported': 'int'})
train_df
train_df.info()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
train_df[num_cols].corr()
sns.heatmap(train_df[num_cols].corr(), annot=True, cmap='coolwarm')
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='RoomService', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='Spa', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='ShoppingMall', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=train_df, x='Age', hue='Transported', bins=8, ax=ax)

(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=train_df, x='Age', hue='Transported', shade=True, ax=ax)
