import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_train.head()
df_test.head()
nan_train = df_train.columns[df_train.isna().any()].tolist()
n_nan_train = df_train[nan_train].isna().sum()
print(f'TRAINING SET\nMissing values:\n{n_nan_train}')
plt.figure(figsize=(14, 5))
plt.title('missing values on training set')
sns.barplot(y=n_nan_train, x=nan_train, palette='viridis')
nan_test = df_test.columns[df_test.isna().any()].tolist()
n_nan_test = df_test[nan_test].isna().sum()
print(f'TESTING SET\nMissing values:\n{n_nan_test}')
plt.figure(figsize=(14, 5))
plt.title('missing values on testing set')
sns.barplot(y=n_nan_test, x=nan_test, palette='magma')
group_number = df_train['PassengerId'].apply(lambda x: x.split('_')).values
group = list(map(lambda x: x[0], group_number))
passenger_number = list(map(lambda x: x[1], group_number))
df_train['group'] = group
df_train['passenger_num'] = passenger_number
df_train['passenger_num'] = df_train['passenger_num'].astype('int64')
mode = df_train['group'].mode()[0]
maxP_inGroup = len(df_train[df_train['group'] == mode])
print('The maximum number of passengers in a single group is', maxP_inGroup)
df_train['group_size'] = 0
for i in range(maxP_inGroup):
    curr_group = df_train[df_train['passenger_num'] == i + 1]['group'].to_numpy()
    df_train.loc[df_train['group'].isin(curr_group), ['group_size']] = i + 1
group_number = df_test['PassengerId'].apply(lambda x: x.split('_')).values
group = list(map(lambda x: x[0], group_number))
passenger_num = list(map(lambda x: x[1], group_number))
df_test['group'] = group
df_test['passenger_num'] = passenger_num
df_test['passenger_num'] = df_test['passenger_num'].astype('int64')
mode = df_test['group'].mode()[0]
maxP_inGroup = len(df_test[df_test['group'] == mode])
print('The maximum number of passengers in the same group is', maxP_inGroup)
df_test['group_size'] = 0
for i in range(maxP_inGroup):
    curr_group = df_test[df_test['passenger_num'] == i + 1]['group'].to_numpy()
    df_test.loc[df_test['group'].isin(curr_group), ['group_size']] = i + 1
df_train['InGroup'] = df_train['group_size'] == 1
df_test['InGroup'] = df_test['group_size'] == 1
df_train.head()
df_test.head()
df_train['Deck'] = df_train['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
df_test['Deck'] = df_test['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
df_train['Num'] = df_train['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
df_test['Num'] = df_test['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
df_train['Side'] = df_train['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
df_test['Side'] = df_test['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
passengerID_test_list = df_test['PassengerId'].tolist()
df_train.drop('PassengerId', axis=1, inplace=True)
df_test.drop('PassengerId', axis=1, inplace=True)
df_train.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)
df_train.drop('Name', axis=1, inplace=True)
df_test.drop('Name', axis=1, inplace=True)
df_train.drop('group', axis=1, inplace=True)
df_test.drop('group', axis=1, inplace=True)
df_train.drop('passenger_num', axis=1, inplace=True)
df_test.drop('passenger_num', axis=1, inplace=True)
df_train.drop('group_size', axis=1, inplace=True)
df_test.drop('group_size', axis=1, inplace=True)
df_train.head()
df_test.head()
df_train.dtypes
num_feature = list(df_train.select_dtypes(include='number'))
categ_feature = list(df_train.select_dtypes(exclude='number'))
test_categ_feature = list(df_test.select_dtypes(exclude='number'))
print(colored('Numerical features:', 'blue'), num_feature)
print(colored('Categorical features (train):', 'red'), categ_feature)
print(colored('Categorical features (test):', 'red'), test_categ_feature)
for feature in num_feature:
    df_train[feature].fillna(df_train[feature].median(), inplace=True)
    df_test[feature].fillna(df_test[feature].median(), inplace=True)
for feature in categ_feature:
    df_train[feature].fillna(df_train[feature].value_counts().index[0], inplace=True)
for feature in test_categ_feature:
    df_test[feature].fillna(df_test[feature].value_counts().index[0], inplace=True)
print(colored('df_train NaN value =', 'green'), df_train.isna().sum().sum())
print(colored('df_test NaN value =', 'green'), df_test.isna().sum().sum())
color = ['#755540', '#f8c1e1', '#f0e4cb', '#f2efe5', '#ed1c24', '#fff01f', '#1f2eff']
fig = plt.figure(figsize=(15, 10))
for (i, col) in enumerate(categ_feature):
    ax = fig.add_subplot(3, 3, i + 1)
    sns.countplot(x=df_train[col], palette=color, ax=ax)
fig.tight_layout()

fig = plt.figure(figsize=(16, 10))
for (i, col) in enumerate(num_feature):
    ax = fig.add_subplot(2, 4, i + 1)
    df_train.groupby(['Transported'])[col].mean().plot(kind='bar', color=['#755540', '#f8c1e1'])
    ax.set_ylabel(col)
fig.tight_layout()

df_train.columns
df_train.dtypes
LABELS = df_test.columns
encoder = LabelEncoder()
for col in LABELS:
    if df_train[col].dtype == 'O':
        df_train[col] = encoder.fit_transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
    elif df_train[col].dtype == 'bool':
        df_train[col] = df_train[col].astype('int')
        df_test[col] = df_test[col].astype('int')
df_train['Transported'] = df_train['Transported'].astype('int')
LABELS_MM = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
mm_scaler = MinMaxScaler()
df_train[LABELS_MM] = mm_scaler.fit_transform(df_train[LABELS_MM])
df_test[LABELS_MM] = mm_scaler.transform(df_test[LABELS_MM])
df_train.head()
df_test.head()
x = df_train[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'InGroup', 'Deck', 'Num', 'Side']]
y = df_train['Transported']
x.head()
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=1)
params = {'max_depth': [3, 6], 'gamma': [0, 1], 'learning_rate': [0.01, 0.02, 0.05, 0.1], 'n_estimators': [100, 200, 500, 1000], 'colsample_bytree': [0.3, 0.5, 0.7]}
xgb_grid = GridSearchCV(estimator=XGBClassifier(), param_grid=params, cv=2)
best_params = {'colsample_bytree': 0.7, 'gamma': 1, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200}
xgboost_model = XGBClassifier(colsample_bytree=0.7, gamma=1, learning_rate=0.05, max_depth=6, n_estimators=200)