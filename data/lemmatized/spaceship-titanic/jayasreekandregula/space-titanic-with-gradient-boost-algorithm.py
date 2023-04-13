import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.isnull().sum()
_input1 = _input1.drop(['Name'], axis=1, inplace=False)
_input0 = _input0.drop(['Name'], axis=1, inplace=False)

def random_imputation(df):
    random_list = df['Age'].dropna().sample(df['Age'].isnull().sum(), random_state=0)
    df['Age_random'] = df['Age']
    random_list.index = df[df['Age'].isnull()].index
    df.loc[df['Age_random'].isnull(), 'Age_random'] = random_list
    return df
_input1['Age'] = np.where(_input1['Age'] == 0, np.nan, _input1['Age'])
_input0['Age'] = np.where(_input0['Age'] == 0, np.nan, _input0['Age'])
_input1 = random_imputation(_input1)
_input0 = random_imputation(_input0)
_input1['AgeBand'] = pd.cut(_input1['Age_random'], 5)
_input1['AgeBand'].value_counts()

def computate_ageBand(dataset):
    dataset.loc[dataset['Age_random'] <= 16, 'Age_random'] = 0
    dataset.loc[(dataset['Age_random'] > 16) & (dataset['Age_random'] <= 31), 'Age_random'] = 1
    dataset.loc[(dataset['Age_random'] > 31) & (dataset['Age_random'] <= 46), 'Age_random'] = 2
    dataset.loc[(dataset['Age_random'] > 46) & (dataset['Age_random'] <= 62), 'Age_random'] = 3
    dataset.loc[(dataset['Age_random'] > 62) & (dataset['Age_random'] <= 79), 'Age_random'] = 4
computate_ageBand(_input1)
computate_ageBand(_input0)
_input1['Age_random'].value_counts()
sns.countplot(data=_input1, x='Age_random', hue='Transported')
(fig, ax) = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.flatten()
sns.countplot(ax=ax[0], x='VIP', data=_input1)
sns.countplot(ax=ax[1], x='CryoSleep', data=_input1)
sns.countplot(ax=ax[2], x='Destination', data=_input1)
sns.countplot(ax=ax[3], x='Age_random', data=_input1)
_input1['Group'] = _input1['PassengerId'].str[0:4]
_input0['Group'] = _input0['PassengerId'].str[0:4]
_input1['Group'] = _input1['Group'].astype(int)
groups = []
for x in _input1['Group']:
    groups.append(len(_input1[_input1['Group'] == x]))
_input0['Group'] = _input0['Group'].astype(int)
groups1 = []
for x in _input0['Group']:
    groups1.append(len(_input0[_input0['Group'] == x]))
_input1['GroupsCount'] = groups
_input0['GroupsCount'] = groups1
sns.countplot(data=_input1, x='GroupsCount', hue='Transported')
_input1 = _input1.drop(['Group'], axis=1, inplace=False)
_input0 = _input0.drop(['Group'], axis=1, inplace=False)

def fillnan_with_zero(df, column):
    df[column] = df[column].fillna(0, inplace=False)
fillnan_with_zero(_input1, 'VRDeck')
fillnan_with_zero(_input1, 'Spa')
fillnan_with_zero(_input1, 'ShoppingMall')
fillnan_with_zero(_input1, 'FoodCourt')
fillnan_with_zero(_input1, 'RoomService')
fillnan_with_zero(_input0, 'VRDeck')
fillnan_with_zero(_input0, 'Spa')
fillnan_with_zero(_input0, 'ShoppingMall')
fillnan_with_zero(_input0, 'FoodCourt')
fillnan_with_zero(_input0, 'RoomService')
_input1['Expenses'] = _input1['VRDeck'] + _input1['Spa'] + _input1['ShoppingMall'] + _input1['FoodCourt'] + _input1['RoomService']
_input0['Expenses'] = _input0['VRDeck'] + _input0['Spa'] + _input0['ShoppingMall'] + _input0['FoodCourt'] + _input0['RoomService']
sns.distplot(_input1['Expenses'])
_input1['cabin'] = _input1['Cabin'].str[0]
_input1['cabin_port'] = _input1['Cabin'].str[-1]
_input0['cabin'] = _input0['Cabin'].str[0]
_input0['cabin_port'] = _input0['Cabin'].str[-1]
ids = _input0['PassengerId']
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Age', 'ShoppingMall', 'FoodCourt', 'VRDeck', 'Spa', 'RoomService'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Age', 'ShoppingMall', 'FoodCourt', 'VRDeck', 'Spa', 'RoomService'], axis=1, inplace=False)
_input1 = _input1.drop(['AgeBand'], axis=1, inplace=False)
_input1.head(5)
_input0.head(5)
target = _input1.Transported
_input1 = _input1.drop(['Transported'], axis=1, inplace=False)
from sklearn.preprocessing import LabelEncoder

def label_encode(df):
    df = df.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
    return df
_input1 = label_encode(_input1)
_input0 = label_encode(_input0)
_input1
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
X = _input1
y = target
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
set_config(display='diagram')
cat_pipeline = Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
cat_pipeline
num_pipeline = Pipeline([('minmax', MinMaxScaler()), ('standard', StandardScaler())])
num_pipeline
preprocess = ColumnTransformer([('categorical', cat_pipeline, ['HomePlanet', 'CryoSleep', 'cabin', 'cabin_port', 'Destination', 'VIP']), ('numerical', num_pipeline, ['Age_random', 'Expenses'])])
pipe = make_pipeline(preprocess, GradientBoostingClassifier())
pipe