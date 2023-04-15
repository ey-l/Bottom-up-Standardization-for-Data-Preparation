import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
samp = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head()
train.tail()
train.info()
test.info()
cat_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for col in cat_cols:
    print(col)
    print('unique values : ', train[col].unique())
    print('number of unique values : ', len(train[col].unique()))
    print('values count : \n', train[col].value_counts())
    print('--------------------------------------------------------')
cont_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(fig, axes) = plt.subplots(2, len(cont_cols) // 2, figsize=(15, 6))
sns.set_style('whitegrid')
for (i, col) in enumerate(cont_cols):
    ax = axes.flat[i]
    sns.histplot(x=col, data=train, kde=True, bins=50, ax=ax, zorder=2)
    ax.set_title(col)
    ax.set(xlabel=None, ylabel=None)
fig.suptitle('Continuous Variables')
plt.tight_layout()

from sklearn.impute import SimpleImputer
simp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
simp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for col in cat_cols:
    train[col] = simp_cat.fit_transform(train[col].values.reshape(-1, 1))[:, 0]
    test[col] = simp_cat.transform(test[col].values.reshape(-1, 1))[:, 0]
for col in cont_cols:
    train[col] = simp_num.fit_transform(train[col].values.reshape(-1, 1))[:, 0]
    test[col] = simp_num.transform(test[col].values.reshape(-1, 1))[:, 0]
train.info()
(fig, axes) = plt.subplots(2, 3, figsize=(15, 9))
sns.set_style('whitegrid')
for (i, col) in enumerate(cont_cols):
    ax = axes.flat[i]
    sns.kdeplot(data=train, x=col, hue='Transported', ax=ax, shade=True)
    plt.title(col)
plt.suptitle('data distribution by target variable')
plt.tight_layout()
train.head()
(fig, axes) = plt.subplots(2, 2, figsize=(10, 9))
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
sns.set_style('ticks')
for (i, col) in enumerate(cat_cols):
    ax = axes.flat[i]
    sns.countplot(data=train, x=col, ax=ax)
    plt.title(col)
fig.suptitle('count of categories')
plt.tight_layout()
(fig, axes) = plt.subplots(2, 2, figsize=(15, 9))
plt.title('number of transported by categorie in qualitative variables')
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
sns.set_style('ticks')
for (i, col) in enumerate(cat_cols):
    ax = axes.flat[i]
    sns.countplot(data=train, x=col, ax=ax, hue='Transported')
    plt.title(col)
fig.suptitle('number of transported by categorie in qualitative variables')
plt.tight_layout()
quant_cols = train[cont_cols]
plt.figure(figsize=(10, 8))
sns.heatmap(quant_cols.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
plt.title('qunatitative variables correlation')
train.head()
(fig, axes) = plt.subplots(2, 3, figsize=(12, 9))
sns.set_style('ticks')
for (i, col) in enumerate(cont_cols):
    ax = axes.flat[i]
    sns.stripplot(x='Transported', y=col, data=train, ax=ax)
    plt.title(col)
fig.suptitle('Continuous variables by Target')
plt.tight_layout()
train.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Transported'] = le.fit_transform(train['Transported'])
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])
test.info()
train.head()
train.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
X = train.copy()
y = X.pop('Transported')
pid = test['PassengerId']
test.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
test.info()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
model = RandomForestClassifier(random_state=0)
baseline_score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print('Accuracy score : ', baseline_score.mean())