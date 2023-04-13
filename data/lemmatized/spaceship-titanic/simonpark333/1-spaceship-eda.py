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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')
_input1
_input0
_input1.info()
_input0.info()
missingno.matrix(_input1)
missingno.matrix(_input0)
_input1.info()
_input1 = _input1.astype({'Transported': 'int'})
_input1
_input1.info()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']
_input1[num_cols].corr()
sns.heatmap(_input1[num_cols].corr(), annot=True, cmap='coolwarm')
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='RoomService', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Spa', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='ShoppingMall', hue='Transported', bins=40, ax=ax)
ax.set_xlim(0, 10000)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.histplot(data=_input1, x='Age', hue='Transported', bins=8, ax=ax)
(fig, ax) = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.kdeplot(data=_input1, x='Age', hue='Transported', shade=True, ax=ax)