import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.isnull().sum()
df_train.drop(['Name'], axis=1, inplace=True)
df_test.drop(['Name'], axis=1, inplace=True)

def random_imputation(df):
    random_list = df['Age'].dropna().sample(df['Age'].isnull().sum(), random_state=0)
    df['Age_random'] = df['Age']
    random_list.index = df[df['Age'].isnull()].index
    df.loc[df['Age_random'].isnull(), 'Age_random'] = random_list
    return df
df_train['Age'] = np.where(df_train['Age'] == 0, np.nan, df_train['Age'])
df_test['Age'] = np.where(df_test['Age'] == 0, np.nan, df_test['Age'])
df_train = random_imputation(df_train)
df_test = random_imputation(df_test)
df_train['AgeBand'] = pd.cut(df_train['Age_random'], 5)
df_train['AgeBand'].value_counts()

def computate_ageBand(dataset):
    dataset.loc[dataset['Age_random'] <= 16, 'Age_random'] = 0
    dataset.loc[(dataset['Age_random'] > 16) & (dataset['Age_random'] <= 31), 'Age_random'] = 1
    dataset.loc[(dataset['Age_random'] > 31) & (dataset['Age_random'] <= 46), 'Age_random'] = 2
    dataset.loc[(dataset['Age_random'] > 46) & (dataset['Age_random'] <= 62), 'Age_random'] = 3
    dataset.loc[(dataset['Age_random'] > 62) & (dataset['Age_random'] <= 79), 'Age_random'] = 4
computate_ageBand(df_train)
computate_ageBand(df_test)
df_train['Age_random'].value_counts()
sns.countplot(data=df_train, x='Age_random', hue='Transported')
(fig, ax) = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.flatten()
sns.countplot(ax=ax[0], x='VIP', data=df_train)
sns.countplot(ax=ax[1], x='CryoSleep', data=df_train)
sns.countplot(ax=ax[2], x='Destination', data=df_train)
sns.countplot(ax=ax[3], x='Age_random', data=df_train)
df_train['Group'] = df_train['PassengerId'].str[0:4]
df_test['Group'] = df_test['PassengerId'].str[0:4]
df_train['Group'] = df_train['Group'].astype(int)
groups = []
for x in df_train['Group']:
    groups.append(len(df_train[df_train['Group'] == x]))
df_test['Group'] = df_test['Group'].astype(int)
groups1 = []
for x in df_test['Group']:
    groups1.append(len(df_test[df_test['Group'] == x]))
df_train['GroupsCount'] = groups
df_test['GroupsCount'] = groups1
sns.countplot(data=df_train, x='GroupsCount', hue='Transported')
df_train.drop(['Group'], axis=1, inplace=True)
df_test.drop(['Group'], axis=1, inplace=True)

def fillnan_with_zero(df, column):
    df[column].fillna(0, inplace=True)
fillnan_with_zero(df_train, 'VRDeck')
fillnan_with_zero(df_train, 'Spa')
fillnan_with_zero(df_train, 'ShoppingMall')
fillnan_with_zero(df_train, 'FoodCourt')
fillnan_with_zero(df_train, 'RoomService')
fillnan_with_zero(df_test, 'VRDeck')
fillnan_with_zero(df_test, 'Spa')
fillnan_with_zero(df_test, 'ShoppingMall')
fillnan_with_zero(df_test, 'FoodCourt')
fillnan_with_zero(df_test, 'RoomService')
df_train['Expenses'] = df_train['VRDeck'] + df_train['Spa'] + df_train['ShoppingMall'] + df_train['FoodCourt'] + df_train['RoomService']
df_test['Expenses'] = df_test['VRDeck'] + df_test['Spa'] + df_test['ShoppingMall'] + df_test['FoodCourt'] + df_test['RoomService']
sns.distplot(df_train['Expenses'])
df_train['cabin'] = df_train['Cabin'].str[0]
df_train['cabin_port'] = df_train['Cabin'].str[-1]
df_test['cabin'] = df_test['Cabin'].str[0]
df_test['cabin_port'] = df_test['Cabin'].str[-1]
ids = df_test['PassengerId']
df_train.drop(['PassengerId', 'Cabin', 'Age', 'ShoppingMall', 'FoodCourt', 'VRDeck', 'Spa', 'RoomService'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Cabin', 'Age', 'ShoppingMall', 'FoodCourt', 'VRDeck', 'Spa', 'RoomService'], axis=1, inplace=True)
df_train.drop(['AgeBand'], axis=1, inplace=True)
df_train.head(5)
df_test.head(5)
target = df_train.Transported
df_train.drop(['Transported'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

def label_encode(df):
    df = df.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
    return df
df_train = label_encode(df_train)
df_test = label_encode(df_test)
df_train
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
X = df_train
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