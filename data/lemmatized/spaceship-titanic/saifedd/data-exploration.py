import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.head()
_input1.tail()
_input1.info()
_input0.info()
cat_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for col in cat_cols:
    print(col)
    print('unique values : ', _input1[col].unique())
    print('number of unique values : ', len(_input1[col].unique()))
    print('values count : \n', _input1[col].value_counts())
    print('--------------------------------------------------------')
cont_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axes) = plt.subplots(2, len(cont_cols) // 2, figsize=(15, 6))
sns.set_style('whitegrid')
for (i, col) in enumerate(cont_cols):
    ax = axes.flat[i]
    sns.histplot(x=col, data=_input1, kde=True, bins=50, ax=ax, zorder=2)
    ax.set_title(col)
    ax.set(xlabel=None, ylabel=None)
fig.suptitle('Continuous Variables')
plt.tight_layout()
from sklearn.impute import SimpleImputer
simp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
simp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for col in cat_cols:
    _input1[col] = simp_cat.fit_transform(_input1[col].values.reshape(-1, 1))[:, 0]
    _input0[col] = simp_cat.transform(_input0[col].values.reshape(-1, 1))[:, 0]
for col in cont_cols:
    _input1[col] = simp_num.fit_transform(_input1[col].values.reshape(-1, 1))[:, 0]
    _input0[col] = simp_num.transform(_input0[col].values.reshape(-1, 1))[:, 0]
_input1.info()
(fig, axes) = plt.subplots(2, 3, figsize=(15, 9))
sns.set_style('whitegrid')
for (i, col) in enumerate(cont_cols):
    ax = axes.flat[i]
    sns.kdeplot(data=_input1, x=col, hue='Transported', ax=ax, shade=True)
    plt.title(col)
plt.suptitle('data distribution by target variable')
plt.tight_layout()
_input1.head()
(fig, axes) = plt.subplots(2, 2, figsize=(10, 9))
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
sns.set_style('ticks')
for (i, col) in enumerate(cat_cols):
    ax = axes.flat[i]
    sns.countplot(data=_input1, x=col, ax=ax)
    plt.title(col)
fig.suptitle('count of categories')
plt.tight_layout()
(fig, axes) = plt.subplots(2, 2, figsize=(15, 9))
plt.title('number of transported by categorie in qualitative variables')
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
sns.set_style('ticks')
for (i, col) in enumerate(cat_cols):
    ax = axes.flat[i]
    sns.countplot(data=_input1, x=col, ax=ax, hue='Transported')
    plt.title(col)
fig.suptitle('number of transported by categorie in qualitative variables')
plt.tight_layout()
quant_cols = _input1[cont_cols]
plt.figure(figsize=(10, 8))
sns.heatmap(quant_cols.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
plt.title('qunatitative variables correlation')
_input1.head()
(fig, axes) = plt.subplots(2, 3, figsize=(12, 9))
sns.set_style('ticks')
for (i, col) in enumerate(cont_cols):
    ax = axes.flat[i]
    sns.stripplot(x='Transported', y=col, data=_input1, ax=ax)
    plt.title(col)
fig.suptitle('Continuous variables by Target')
plt.tight_layout()
_input1.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
_input1['Transported'] = le.fit_transform(_input1['Transported'])
for col in cat_cols:
    _input1[col] = le.fit_transform(_input1[col])
    _input0[col] = le.fit_transform(_input0[col])
_input0.info()
_input1.head()
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
X = _input1.copy()
y = X.pop('Transported')
pid = _input0['PassengerId']
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=False)
_input0.info()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
model = RandomForestClassifier(random_state=0)
baseline_score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print('Accuracy score : ', baseline_score.mean())