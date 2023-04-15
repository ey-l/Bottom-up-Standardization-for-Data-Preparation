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
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df.head()
test_df.head()
train_df.info()
test_df.info()
train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)
(train_df.shape, test_df.shape)

def plot_null(df):
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=df.isna().sum().sort_values().index, y=df.isna().sum().sort_values().values)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.bar_label(ax.containers[0])

plot_null(train_df)
plot_null(test_df)
num_var = train_df.select_dtypes(exclude=['O', bool]).columns.values
num_var
plot_null(train_df.loc[:, num_var])
plot_null(test_df.loc[:, num_var])
num_imputer = SimpleImputer(strategy='mean')
train_df.loc[:, num_var] = num_imputer.fit_transform(train_df.loc[:, num_var])
plot_null(train_df.loc[:, num_var])
test_df.loc[:, num_var] = num_imputer.transform(test_df.loc[:, num_var])
plot_null(test_df.loc[:, num_var])
plot_null(train_df)
plot_null(test_df)
train_df.drop('PassengerId', axis=1, inplace=True)
passenger_id_df = test_df[['PassengerId']]
test_df = test_df.drop('PassengerId', axis=1)
cat_var = train_df.select_dtypes('O').columns.values
train_df.loc[:, cat_var].head()
train_df.loc[:, cat_var].describe()
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df.loc[:, ['HomePlanet', 'Destination']] = cat_imputer.fit_transform(train_df[['HomePlanet', 'Destination']])
test_df.loc[:, ['HomePlanet', 'Destination']] = cat_imputer.transform(test_df[['HomePlanet', 'Destination']])
plot_null(train_df)
plot_null(test_df)
cabin_split_train_df = train_df['Cabin'].str.split('/', expand=True)
cabin_split_train_df.columns = ['cabin_deck', 'cabin_deck_num', 'cabin_side']
cabin_split_train_df.head()
cabin_split_test_df = test_df['Cabin'].str.split('/', expand=True)
cabin_split_test_df.columns = ['cabin_deck', 'cabin_deck_num', 'cabin_side']
cabin_split_test_df.head()
train_df = pd.concat([train_df, cabin_split_train_df], axis=1).drop('Cabin', axis=1)
test_df = pd.concat([test_df, cabin_split_test_df], axis=1).drop('Cabin', axis=1)
train_df.head()
test_df.head()
cat_var_new = train_df.select_dtypes('O').columns.values
train_df.loc[:, cat_var_new].describe()
test_df.loc[:, cat_var_new].describe()
train_df.drop('cabin_deck_num', axis=1, inplace=True)
test_df.drop('cabin_deck_num', axis=1, inplace=True)
cat_var_new = train_df.select_dtypes('O').columns.values
train_df.loc[:, ['cabin_deck', 'cabin_side']] = cat_imputer.fit_transform(train_df.loc[:, ['cabin_deck', 'cabin_side']])
test_df.loc[:, ['cabin_deck', 'cabin_side']] = cat_imputer.transform(test_df.loc[:, ['cabin_deck', 'cabin_side']])
plot_null(train_df.loc[:, cat_var_new])
plot_null(test_df.loc[:, cat_var_new])
train_df[['CryoSleep', 'VIP']].describe()
for var in num_var:
    sns.boxplot(x='CryoSleep', y=var, data=train_df)

for var in num_var:
    sns.boxplot(x='VIP', y=var, data=train_df)

train_df['total_bill'] = train_df.loc[:, num_var[1:]].sum(axis=1).values
test_df['total_bill'] = test_df.loc[:, num_var[1:]].sum(axis=1).values
train_df.head()
test_df.head()
for var in ['CryoSleep', 'VIP']:
    sns.boxplot(x=var, y='total_bill', data=train_df)

for var in ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']:
    sns.countplot(x=var, data=train_df, hue='CryoSleep')

for var in ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']:
    sns.countplot(x=var, data=train_df, hue='VIP')

train_vip_df = train_df.loc[:, np.append(num_var, ['total_bill', 'VIP'])].copy()
test_vip_df = test_df.loc[:, np.append(num_var, ['total_bill', 'VIP'])].copy()
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