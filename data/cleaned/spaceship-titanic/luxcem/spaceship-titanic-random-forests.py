import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn import metrics
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
n_train = len(train)
n_test = len(test)
data = pd.concat([train, test]).set_index('PassengerId')
data.head()
import missingno as msno

msno.matrix(data)
train.isna().sum()
data.describe()
data.describe(include='object')
cat_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
con_col = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cat_col:
    print(data[col].value_counts())
    print()
for col in con_col:
    data[col] = data[col].fillna(data[col].mean())
for col in cat_col:
    data[col] = data[col].fillna(data[col].value_counts().idxmax())
data.describe()
(fig, axs) = plt.subplots(ncols=len(con_col), figsize=(16, 5))
for (i, col) in enumerate(con_col):
    sns.histplot(data, x=col, ax=axs[i], bins='doane')
money_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in money_col:
    data[col] = np.log(data[col] + 1)
(fig, axs) = plt.subplots(ncols=len(con_col), figsize=(16, 5))
for (i, col) in enumerate(con_col):
    sns.histplot(data, x=col, ax=axs[i], bins='sqrt')
cabin_col = ['Deck', 'Num', 'Side']
data[cabin_col] = data['Cabin'].str.split('/', expand=True)
data[cabin_col] = data[cabin_col].fillna(method='ffill')
for col in cabin_col:
    print(data[col].value_counts())
    print()
data.Num.describe()
data['Age'] = data['Age'] / data['Age'].max()
data.head()
ohe_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
data = pd.get_dummies(data, prefix=ohe_columns, columns=ohe_columns, drop_first=True)
data = data.drop(['Name', 'Cabin', 'Num'], axis=1)
data.head()
corr = data[:n_train].corr()
(fig, ax) = pyplot.subplots(figsize=(15, 15))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

X_train = data[:n_train].drop(['Transported'], axis=1)
y_train = data[:n_train]['Transported'].astype(int)
print(X_train.shape, y_train.shape)
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2)
X_sample = data[:n_train].sample(500)
y_sample = X_sample['Transported'].astype(int)
X_sample = X_sample.drop(['Transported'], axis=1)
print(X_sample.shape, y_sample.shape)