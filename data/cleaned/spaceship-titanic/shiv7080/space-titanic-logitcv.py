import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm
import missingno as mn
from scipy import stats
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, power_transform
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from scipy import stats
sns.set_style('whitegrid')
sns.color_palette('flare')
sns.set_palette(sns.color_palette('flare'))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_c = pd.read_csv('data/input/spaceship-titanic/test.csv')
test = test_c.copy()


enc = LabelEncoder()
train['Transported'] = enc.fit_transform(train['Transported'])
sns.countplot(data=train, x=train.Transported)
print('shape of train data:', train.shape)
print('shape of test data:', test.shape)

print('=' * 100)

num_col = [a for a in train.columns if train[a].dtype in ['float64']]
cat_col = [a for a in train.columns if train[a].dtype in ['object']]
print('\nNumerical_features', num_col)
print('\nCategorical_features', cat_col)
train.head()
y = train['Transported']
train.head()

print('=' * 100)


def fill_values(df):
    df['ShoppingMall'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['ShoppingMall']])
    df['RoomService'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['RoomService']])
    df['FoodCourt'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['FoodCourt']])
    df['Spa'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['Spa']])
    df['VRDeck'] = SimpleImputer(strategy='constant', fill_value=0).fit_transform(df[['VRDeck']])
    df['Age'] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df[['Age']])
    df['HomePlanet'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['HomePlanet']])
    df['CryoSleep'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['CryoSleep']])
    df['Cabin'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Cabin']])
    df['Destination'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Destination']])
    df['VIP'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['VIP']])
fill_values(train)
fill_values(test)
print(train.info())
print(test.info())
train[['Group_No', 'No_in_group']] = train['PassengerId'].str.split('_', expand=True)
test[['Group_No', 'No_in_group']] = test['PassengerId'].str.split('_', expand=True)
train['total_exp'] = train['ShoppingMall'] + train['Spa'] + train['VRDeck'] + train['FoodCourt'] + train['RoomService']
test['total_exp'] = test['ShoppingMall'] + test['Spa'] + test['VRDeck'] + test['FoodCourt'] + test['RoomService']
train[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = train['Cabin'].str.split('/', expand=True)
test[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = test['Cabin'].str.split('/', expand=True)
train.drop(columns=['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
test.drop(columns=['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
cat_col = [a for a in train.columns if train[a].dtype in ['object']]
num_col = [a for a in train.columns if train[a].dtype in ['float64']]
print('Categorical features: ', cat_col)
print('Numerical features: ', num_col)
(fig, ax) = plt.subplots(3, 3, figsize=(20, 15))
sns.kdeplot(ax=ax[0, 0], x='Age', data=train, fill=True, hue=round(train['Age'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 1], x='RoomService', data=train, fill=True, hue=round(train['RoomService'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 2], x='FoodCourt', data=train, fill=True, hue=round(train['FoodCourt'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 0], x='ShoppingMall', data=train, fill=True, hue=round(train['ShoppingMall'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 1], x='Spa', data=train, fill=True, hue=round(train['Spa'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 2], x='VRDeck', data=train, fill=True, hue=round(train['VRDeck'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[2, 0], x='total_exp', data=train, fill=True, hue=round(train['total_exp'].skew(), 2), palette='Spectral')
train[num_col] = power_transform(train[num_col], method='yeo-johnson', standardize=True)
test[num_col] = power_transform(test[num_col], method='yeo-johnson', standardize=True)
(fig, ax) = plt.subplots(3, 3, figsize=(20, 15))
sns.kdeplot(ax=ax[0, 0], x='Age', data=train, fill=True, hue=round(train['Age'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 1], x='RoomService', data=train, fill=True, hue=round(train['RoomService'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[0, 2], x='FoodCourt', data=train, fill=True, hue=round(train['FoodCourt'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 0], x='ShoppingMall', data=train, fill=True, hue=round(train['ShoppingMall'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 1], x='Spa', data=train, fill=True, hue=round(train['Spa'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[1, 2], x='VRDeck', data=train, fill=True, hue=round(train['VRDeck'].skew(), 2), palette='Spectral')
sns.kdeplot(ax=ax[2, 0], x='total_exp', data=train, fill=True, hue=round(train['total_exp'].skew(), 2), palette='Spectral')
train[cat_col] = OrdinalEncoder().fit_transform(train[cat_col])
test[cat_col] = OrdinalEncoder().fit_transform(test[cat_col])
features = num_col + cat_col
X = train[features]
y = train['Transported']
cv = 5
metric = []
kfold = KFold(n_splits=cv, random_state=42, shuffle=True)
for (idx, (train_idx, val_idx)) in enumerate(kfold.split(train)):
    (xtrain, xvalid) = (X.loc[train_idx], X.loc[val_idx])
    (ytrain, yvalid) = (y.loc[train_idx], y.loc[val_idx])
    log_model = LogisticRegression(penalty='l2', C=10, max_iter=1000)