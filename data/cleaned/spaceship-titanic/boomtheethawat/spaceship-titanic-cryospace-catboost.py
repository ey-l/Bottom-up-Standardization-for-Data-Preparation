import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
original_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
original_test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
original_df.head()
original_test_df.head()
original_df.describe(percentiles=[i / 10 for i in range(1, 10)])
corr = original_df.corr()
heatmap = sb.heatmap(corr, cmap='summer_r', annot=True)
(f, (ax1, ax2)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
sb.boxplot(x=original_df['Age'], hue='Transported', data=original_df, ax=ax1[0])
sb.boxplot(x=original_df['RoomService'], hue='Transported', data=original_df, ax=ax1[1])
sb.boxplot(x=original_df['FoodCourt'], hue='Transported', data=original_df, ax=ax1[2])
sb.boxplot(x=original_df['ShoppingMall'], hue='Transported', data=original_df, ax=ax2[0])
sb.boxplot(x=original_df['Spa'], hue='Transported', data=original_df, ax=ax2[1])
sb.boxplot(x=original_df['VRDeck'], hue='Transported', data=original_df, ax=ax2[2])

(f, (ax1, ax2)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
sb.kdeplot(x=original_df['Age'], hue='Transported', data=original_df, ax=ax1[0])
sb.kdeplot(x=original_df['RoomService'], hue='Transported', data=original_df, ax=ax1[1])
sb.kdeplot(x=original_df['FoodCourt'], hue='Transported', data=original_df, ax=ax1[2])
sb.kdeplot(x=original_df['ShoppingMall'], hue='Transported', data=original_df, ax=ax2[0])
sb.kdeplot(x=original_df['Spa'], hue='Transported', data=original_df, ax=ax2[1])
sb.kdeplot(x=original_df['VRDeck'], hue='Transported', data=original_df, ax=ax2[2])

original_df[['Cabin1', 'Cabin2', 'Cabin3']] = original_df['Cabin'].str.split('/', expand=True)
original_df[['Id', 'GroupSize']] = original_df['PassengerId'].str.split('_', expand=True)
df = original_df.drop(columns=['Name', 'PassengerId', 'VIP', 'Cabin', 'Cabin2', 'Id'])
original_test_df[['Cabin1', 'Cabin2', 'Cabin3']] = original_test_df['Cabin'].str.split('/', expand=True)
original_test_df[['Id', 'GroupSize']] = original_test_df['PassengerId'].str.split('_', expand=True)
test_df = original_test_df.drop(columns=['Name', 'PassengerId', 'VIP', 'Cabin', 'Cabin2', 'Id'])
sb.set_palette('pastel')
sb.set()
(f, (ax1, ax2)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
sb.countplot(data=original_df, x='HomePlanet', hue='Transported', ax=ax1[0], palette=['r', 'g'])
sb.countplot(data=original_df, x='Destination', hue='Transported', ax=ax1[1], palette=['r', 'g'])
sb.countplot(data=original_df, x='CryoSleep', hue='Transported', ax=ax1[2], palette=['r', 'g'])
sb.countplot(data=original_df, x='VIP', hue='Transported', ax=ax2[0], palette=['r', 'g'])
sb.countplot(data=original_df, x='Cabin1', hue='Transported', ax=ax2[1], palette=['r', 'g'])
sb.countplot(data=original_df, x='Cabin3', hue='Transported', ax=ax2[2], palette=['r', 'g'])


def handleoutlier(colname):
    q1 = df[colname].quantile(0.05)
    q3 = df[colname].quantile(0.95)
    iqr = q3 - q1
    lowerlimit = q1 - 1.5 * iqr
    upperlimit = q3 + 1.5 * iqr
    df.loc[df[colname] < lowerlimit, [colname]] = lowerlimit
    df.loc[df[colname] > upperlimit, [colname]] = upperlimit
for j in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    handleoutlier(j)

def handleoutlier_test_df(colname):
    q1 = test_df[colname].quantile(0.05)
    q3 = test_df[colname].quantile(0.95)
    iqr = q3 - q1
    lowerlimit = q1 - 1.5 * iqr
    upperlimit = q3 + 1.5 * iqr
    test_df.loc[test_df[colname] < lowerlimit, [colname]] = lowerlimit
    test_df.loc[test_df[colname] > upperlimit, [colname]] = upperlimit
for j in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    handleoutlier_test_df(j)
df.isnull().sum()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df['Cabin1'] = df['Cabin1'].fillna('F')
df['Cabin3'] = df['Cabin3'].fillna('S')
x = imputer.fit_transform(df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = x
df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
df['Destination'] = df['Destination'].fillna('TRAPPIST-1e')
df['CryoSleep'] = df['CryoSleep'].fillna(False)
imputer = KNNImputer(n_neighbors=5)
test_df['Cabin1'] = test_df['Cabin1'].fillna('F')
test_df['Cabin3'] = test_df['Cabin3'].fillna('S')
x = imputer.fit_transform(test_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])
test_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = x
test_df['HomePlanet'] = test_df['HomePlanet'].fillna('Earth')
test_df['Destination'] = test_df['Destination'].fillna('TRAPPIST-1e')
test_df['CryoSleep'] = test_df['CryoSleep'].fillna(False)
df['SumSpend'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
df['AgeCat'] = pd.cut(df.Age, bins=[-1, 5, 12, 18, 50, 150], labels=[0, 1, 2, 3, 4])
df['AgeCat'] = df['AgeCat'].astype(int)
df = df.drop(columns=['Age'])
test_df['SumSpend'] = test_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
test_df['AgeCat'] = pd.cut(test_df.Age, bins=[-1, 5, 12, 18, 50, 150], labels=[0, 1, 2, 3, 4])
test_df['AgeCat'] = test_df['AgeCat'].astype(int)
test_df = test_df.drop(columns=['Age'])
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include=['bool']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
cb1 = pd.get_dummies(df['Cabin1'], prefix='Cabin1_')
cb3 = pd.get_dummies(df['Cabin3'], prefix='Cabin3_')
df = pd.concat([df, cb1, cb3], axis=1)
df['GroupSize'] = df['GroupSize'].astype(int)
(uhp, indiceuhp) = np.unique(np.array(df['HomePlanet']), return_inverse=True)
(ud, indiceud) = np.unique(np.array(df['Destination']), return_inverse=True)
one_hot_homeplanet = np.zeros((indiceuhp.size, indiceuhp.max() + 1))
one_hot_homeplanet[np.arange(indiceuhp.size), indiceuhp] = 1
one_hot_homeplanet = one_hot_homeplanet.astype(int)
one_hot_destination = np.zeros((indiceud.size, indiceud.max() + 1))
one_hot_destination[np.arange(indiceud.size), indiceud] = 1
one_hot_destination = one_hot_destination.astype(int)
df[['Earth', 'Europa', 'Mars']] = one_hot_homeplanet
df[['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']] = one_hot_destination
df = df.drop(columns=['HomePlanet', 'Destination', 'Cabin1', 'Cabin3'])
for col in test_df.select_dtypes(include=['bool']):
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])
cb1 = pd.get_dummies(test_df['Cabin1'], prefix='Cabin1_')
cb3 = pd.get_dummies(test_df['Cabin3'], prefix='Cabin3_')
test_df = pd.concat([test_df, cb1, cb3], axis=1)
test_df['GroupSize'] = test_df['GroupSize'].astype(int)
(uhp, indiceuhp) = np.unique(np.array(test_df['HomePlanet']), return_inverse=True)
(ud, indiceud) = np.unique(np.array(test_df['Destination']), return_inverse=True)
one_hot_homeplanet = np.zeros((indiceuhp.size, indiceuhp.max() + 1))
one_hot_homeplanet[np.arange(indiceuhp.size), indiceuhp] = 1
one_hot_homeplanet = one_hot_homeplanet.astype(int)
one_hot_destination = np.zeros((indiceud.size, indiceud.max() + 1))
one_hot_destination[np.arange(indiceud.size), indiceud] = 1
one_hot_destination = one_hot_destination.astype(int)
test_df[['Earth', 'Europa', 'Mars']] = one_hot_homeplanet
test_df[['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']] = one_hot_destination
test_df = test_df.drop(columns=['HomePlanet', 'Destination', 'Cabin1', 'Cabin3'])
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler
train_df = df.drop(columns=['Transported'])
target_df = df['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(train_df, target_df, test_size=0.3, random_state=5981)
RandomForestClassifier = RandomForestClassifier()
print('---------- RandomForest ----------')
clf = RandomForestClassifier