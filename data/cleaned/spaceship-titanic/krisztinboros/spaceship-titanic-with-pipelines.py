import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
sns.set_theme()
path = 'data/input/spaceship-titanic'
data = pd.read_csv('{}/train.csv'.format(path))
data.sample(10)
data.info()
data.describe()
(fig, ax) = plt.subplots(ncols=2, nrows=2, figsize=(12, 10), dpi=100)
sns.countplot(x='HomePlanet', data=data, hue='Transported', ax=ax[0, 0])
sns.countplot(x='CryoSleep', data=data, hue='Transported', ax=ax[0, 1])
sns.histplot(x='Age', data=data, hue='Transported', ax=ax[1, 0])
sns.countplot(x='VIP', data=data, hue='Transported', ax=ax[1, 1])

def preproc_cols(table):
    table['group_id'] = table['PassengerId'].str.split('_', expand=True)[0]
    table['passenger_id'] = table['PassengerId'].str.split('_', expand=True)[1]
    table.index = table['PassengerId']
    table[['cabin_deck', 'cabin_num', 'cabin_side']] = table['Cabin'].str.split('/', expand=True)
    table['cabin_num'] = table['cabin_num'].astype(float)
    data['Age'] = pd.cut(data['Age'], bins=[0, 20, 30, 40, 80], labels=[0, 1, 2, 3]).astype(float)
    table.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
    return table
preproc_cols(data)
data.sample(10)
groups = dict(data.groupby('group_id').size())
data['group_count'] = data['group_id'].apply(lambda x: groups.get(x))
data['is_alone'] = data['group_count'].apply(lambda x: 0 if x > 1 else 1)
data.sample(10)

def cat_to_num(table):
    table['HomePlanet'].replace({'Europa': 0, 'Earth': 1, 'Mars': 2}, inplace=True)
    table['CryoSleep'].replace({False: 0, True: 1}, inplace=True)
    table['Destination'].replace({'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}, inplace=True)
    table['VIP'].replace({False: 0, True: 1}, inplace=True)
    if 'Transported' in table.columns:
        table['Transported'].replace({False: 0, True: 1}, inplace=True)
    table['cabin_deck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}, inplace=True)
    table['cabin_side'].replace({'P': 0, 'S': 1}, inplace=True)
    return table
cat_to_num(data)
categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'cabin_deck', 'cabin_side', 'group_count', 'is_alone']
numerical = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'cabin_num']
data1 = data.drop(['group_id', 'passenger_id'], axis=1)
data1.info()

def fill_cat_missing(x):
    x.fillna(x.median(), inplace=True)
    return x

def fill_cont_missing(x):
    x.fillna(int(x.mean()), inplace=True)
    return x

def handle_missing(table):
    for column in table.columns:
        if column in categorical:
            fill_cat_missing(table[column])
        elif column in numerical:
            fill_cont_missing(table[column])
        else:
            next
    return table
handle_missing(data1)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical)])
X = data1.drop('Transported', axis=1)
y = data1['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=34)
svm_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=123))
parameters = {'svc__C': [0.01, 0.1, 1, 10, 100], 'svc__gamma': [0.01, 0.1, 1, 10, 100]}
svm = GridSearchCV(svm_pipe, n_jobs=3, param_grid=parameters, cv=5, scoring='accuracy', verbose=1)