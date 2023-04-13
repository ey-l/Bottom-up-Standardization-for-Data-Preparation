import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import choices, seed
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier, Pool
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
spend_vars = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']]
sns.pairplot(spend_vars, hue='Transported', diag_kind='hist')
(_, ax) = plt.subplots(1, 2, figsize=(15, 5))
plt.sca(ax[0])
sns.countplot(data=_input1[['HomePlanet', 'Destination', 'Transported']], x='HomePlanet', hue='Transported')
plt.sca(ax[1])
sns.countplot(data=_input1[['HomePlanet', 'Destination', 'Transported']], x='Destination', hue='Transported')
(_, ax) = plt.subplots(1, 2, figsize=(15, 5))
plt.sca(ax[0])
sns.countplot(data=_input1[['Transported', 'CryoSleep']], x='CryoSleep', hue='Transported')
plt.sca(ax[1])
sns.countplot(data=_input1[['Transported', 'VIP']], x='VIP', hue='Transported')
_input1[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = _input1.Cabin.str.extract('(\\w+)/(\\d+)/(\\w+)')
(_, ax) = plt.subplots(1, 2, figsize=(15, 5))
plt.sca(ax[0])
sns.countplot(data=_input1[['Transported', 'Cabin_deck']], x='Cabin_deck', hue='Transported')
plt.sca(ax[1])
sns.countplot(data=_input1[['Transported', 'Cabin_side']], x='Cabin_side', hue='Transported')
(X_train, X_test, y_train, y_test) = train_test_split(_input1.drop('Transported', axis=1), _input1.Transported, random_state=0)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
X_train.info()

def Impute_from_Group(df, column):
    Missing = df[[column, 'Group']].loc[df[column].isnull()]
    NotMissing = df[[column, 'Group']].loc[~df[column].isnull()]
    Update = Missing.reset_index().merge(NotMissing, on='Group', how='inner', suffixes=['_old', '']).drop_duplicates()
    Update = Update.set_index('PassengerId', inplace=False)
    df.loc[Update.index, column] = Update[column]
    return df

def Feature_Eng(df_in):
    df = df_in.copy()
    df['Group'] = df.PassengerId.str.extract('(\\d{4})_\\d{2}')
    df = df.set_index('PassengerId', inplace=False)
    df['LastName'] = df.Name.str.extract('[\\w+]\\s(\\w+)')
    df = df.drop(['Name'], axis=1, inplace=False)
    df[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = df.Cabin.str.extract('(\\w+)/(\\d+)/(\\w+)')
    df = df.drop(['Cabin', 'Cabin_num'], axis=1, inplace=False)
    df = Impute_from_Group(df, 'HomePlanet')
    df = Impute_from_Group(df, 'Destination')
    df = Impute_from_Group(df, 'Cabin_deck')
    df = Impute_from_Group(df, 'Cabin_side')
    idx2impute = df.loc[df.Cabin_side.isnull(), 'Cabin_side'].index.tolist()
    Imbalance = (round(len(df) / 2) - df.Cabin_side.value_counts()['S']) / (round(len(df) / 2) - df.Cabin_side.value_counts()['P'])
    seed(1)
    df.loc[idx2impute, 'Cabin_side'] = choices(['S', 'P'], weights=[Imbalance, 1], k=len(idx2impute))
    true_idx = df[df.CryoSleep == True].index
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df.loc[true_idx, spend_cols] = 0
    df.loc[true_idx, 'VIP'] = False
    df['Total'] = df[spend_cols].sum(axis=1)
    true_idx = df[(df.Total == 0) & (df.VIP == False) & df.CryoSleep.isnull()].index
    false_idx = df[((df.Total > 0) | (df.VIP == True)) & df.CryoSleep.isnull()].index
    df.loc[true_idx, 'CryoSleep'] = True
    df.loc[false_idx, 'CryoSleep'] = False
    true_idx = df[df.CryoSleep == True].index
    df.loc[true_idx, spend_cols] = 0
    df.loc[true_idx, 'VIP'] = False
    cols2drop = ['LastName', 'Group']
    df = df.drop(cols2drop, axis=1, inplace=False)
    return df
X_train_2 = Feature_Eng(X_train)
X_test_2 = Feature_Eng(X_test)
X_train_2.info()
cat_feats = X_train_2.dtypes[X_train_2.dtypes == object].index.tolist()
num_feats = X_train_2.dtypes[X_train_2.dtypes == float].index.tolist()
ord_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ordEncode', OrdinalEncoder(dtype=int))])
num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='error', drop='if_binary'))])
preprocessor = ColumnTransformer(transformers=[('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)])
param_grid = {'classifier__iterations': [50, 100, 150], 'classifier__depth': [4, 5, 6], 'classifier__learning_rate': [0.1, 0.15, 0.2]}
cat_model = CatBoostClassifier(silent=True)
cat_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', cat_model)])
grid_pipe = GridSearchCV(cat_clf, param_grid, n_jobs=4)