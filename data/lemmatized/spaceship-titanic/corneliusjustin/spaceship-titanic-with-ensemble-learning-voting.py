import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input0.info()
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
(_input1.shape, _input0.shape)

def plot_null(df):
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=df.isna().sum().sort_values().index, y=df.isna().sum().sort_values().values)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.bar_label(ax.containers[0])
plot_null(_input1)
plot_null(_input0)
num_var = _input1.select_dtypes(exclude=['O', bool]).columns.values
num_var
plot_null(_input1.loc[:, num_var])
plot_null(_input0.loc[:, num_var])
num_imputer = SimpleImputer(strategy='mean')
_input1.loc[:, num_var] = num_imputer.fit_transform(_input1.loc[:, num_var])
plot_null(_input1.loc[:, num_var])
_input0.loc[:, num_var] = num_imputer.transform(_input0.loc[:, num_var])
plot_null(_input0.loc[:, num_var])
plot_null(_input1)
plot_null(_input0)
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
passenger_id_df = _input0[['PassengerId']]
_input0 = _input0.drop('PassengerId', axis=1)
cat_var = _input1.select_dtypes('O').columns.values
_input1.loc[:, cat_var].head()
_input1.loc[:, cat_var].describe()
cat_imputer = SimpleImputer(strategy='most_frequent')
_input1.loc[:, ['HomePlanet', 'Destination']] = cat_imputer.fit_transform(_input1[['HomePlanet', 'Destination']])
_input0.loc[:, ['HomePlanet', 'Destination']] = cat_imputer.transform(_input0[['HomePlanet', 'Destination']])
plot_null(_input1)
plot_null(_input0)
cabin_split_train_df = _input1['Cabin'].str.split('/', expand=True)
cabin_split_train_df.columns = ['cabin_deck', 'cabin_deck_num', 'cabin_side']
cabin_split_train_df.head()
cabin_split_test_df = _input0['Cabin'].str.split('/', expand=True)
cabin_split_test_df.columns = ['cabin_deck', 'cabin_deck_num', 'cabin_side']
cabin_split_test_df.head()
_input1 = pd.concat([_input1, cabin_split_train_df], axis=1).drop('Cabin', axis=1)
_input0 = pd.concat([_input0, cabin_split_test_df], axis=1).drop('Cabin', axis=1)
_input1.head()
_input0.head()
cat_var_new = _input1.select_dtypes('O').columns.values
_input1.loc[:, cat_var_new].describe()
_input0.loc[:, cat_var_new].describe()
_input1 = _input1.drop('cabin_deck_num', axis=1, inplace=False)
_input0 = _input0.drop('cabin_deck_num', axis=1, inplace=False)
cat_var_new = _input1.select_dtypes('O').columns.values
_input1.loc[:, ['cabin_deck', 'cabin_side']] = cat_imputer.fit_transform(_input1.loc[:, ['cabin_deck', 'cabin_side']])
_input0.loc[:, ['cabin_deck', 'cabin_side']] = cat_imputer.transform(_input0.loc[:, ['cabin_deck', 'cabin_side']])
plot_null(_input1.loc[:, cat_var_new])
plot_null(_input0.loc[:, cat_var_new])
_input1[['CryoSleep', 'VIP']].describe()
for var in num_var:
    sns.boxplot(x='CryoSleep', y=var, data=_input1)
for var in num_var:
    sns.boxplot(x='VIP', y=var, data=_input1)
_input1['total_bill'] = _input1.loc[:, num_var[1:]].sum(axis=1).values
_input0['total_bill'] = _input0.loc[:, num_var[1:]].sum(axis=1).values
_input1.head()
_input0.head()
for var in ['CryoSleep', 'VIP']:
    sns.boxplot(x=var, y='total_bill', data=_input1)
for var in ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']:
    sns.countplot(x=var, data=_input1, hue='CryoSleep')
for var in ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']:
    sns.countplot(x=var, data=_input1, hue='VIP')
train_vip_df = _input1.loc[:, np.append(num_var, ['total_bill', 'VIP'])].copy()
test_vip_df = _input0.loc[:, np.append(num_var, ['total_bill', 'VIP'])].copy()
(train_vip_df.shape, test_vip_df.shape)
train_vip_df_nonull = train_vip_df.dropna(axis=0, how='any').copy()
train_vip_df_null = train_vip_df.loc[train_vip_df['VIP'].isna(), :].drop('VIP', axis=1)
test_vip_df_null = test_vip_df.loc[test_vip_df['VIP'].isna(), :].drop('VIP', axis=1)
(train_vip_df_nonull.shape, train_vip_df_null.shape, test_vip_df_null.shape)
train_vip_null_index = train_vip_df.loc[train_vip_df['VIP'].isna(), :].index
test_vip_null_index = test_vip_df.loc[test_vip_df['VIP'].isna(), :].index
train_vip_df_nonull.VIP.value_counts()
encoder = LabelEncoder()
train_vip_df_nonull.loc[:, 'VIP'] = encoder.fit_transform(train_vip_df_nonull.VIP)
X_vip = train_vip_df_nonull.drop('VIP', axis=1).values
y_vip = train_vip_df_nonull[['VIP']].values.ravel()
sm = SMOTE(sampling_strategy=0.75)
(X_vip_res, y_vip_res) = sm.fit_resample(X_vip, y_vip)
(X_vip_res.shape, y_vip_res.shape)
(y_vip_res.sum(), (1 - y_vip_res).sum())
scaler = StandardScaler()
X_vip_res = scaler.fit_transform(X_vip_res)
train_vip_df_null = scaler.transform(train_vip_df_null.values)
test_vip_df_null = scaler.transform(test_vip_df_null.values)
(X_train_vip, X_val_vip, y_train_vip, y_val_vip) = train_test_split(X_vip_res, y_vip_res, test_size=0.2, random_state=42)
(X_train_vip.shape, X_val_vip.shape, y_train_vip.shape, y_val_vip.shape)
forest_vip_clf = RandomForestClassifier()