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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
nan_train = _input1.columns[_input1.isna().any()].tolist()
n_nan_train = _input1[nan_train].isna().sum()
print(f'TRAINING SET\nMissing values:\n{n_nan_train}')
plt.figure(figsize=(14, 5))
plt.title('missing values on training set')
sns.barplot(y=n_nan_train, x=nan_train, palette='viridis')
nan_test = _input0.columns[_input0.isna().any()].tolist()
n_nan_test = _input0[nan_test].isna().sum()
print(f'TESTING SET\nMissing values:\n{n_nan_test}')
plt.figure(figsize=(14, 5))
plt.title('missing values on testing set')
sns.barplot(y=n_nan_test, x=nan_test, palette='magma')
group_number = _input1['PassengerId'].apply(lambda x: x.split('_')).values
group = list(map(lambda x: x[0], group_number))
passenger_number = list(map(lambda x: x[1], group_number))
_input1['group'] = group
_input1['passenger_num'] = passenger_number
_input1['passenger_num'] = _input1['passenger_num'].astype('int64')
mode = _input1['group'].mode()[0]
maxP_inGroup = len(_input1[_input1['group'] == mode])
print('The maximum number of passengers in a single group is', maxP_inGroup)
_input1['group_size'] = 0
for i in range(maxP_inGroup):
    curr_group = _input1[_input1['passenger_num'] == i + 1]['group'].to_numpy()
    _input1.loc[_input1['group'].isin(curr_group), ['group_size']] = i + 1
group_number = _input0['PassengerId'].apply(lambda x: x.split('_')).values
group = list(map(lambda x: x[0], group_number))
passenger_num = list(map(lambda x: x[1], group_number))
_input0['group'] = group
_input0['passenger_num'] = passenger_num
_input0['passenger_num'] = _input0['passenger_num'].astype('int64')
mode = _input0['group'].mode()[0]
maxP_inGroup = len(_input0[_input0['group'] == mode])
print('The maximum number of passengers in the same group is', maxP_inGroup)
_input0['group_size'] = 0
for i in range(maxP_inGroup):
    curr_group = _input0[_input0['passenger_num'] == i + 1]['group'].to_numpy()
    _input0.loc[_input0['group'].isin(curr_group), ['group_size']] = i + 1
_input1['InGroup'] = _input1['group_size'] == 1
_input0['InGroup'] = _input0['group_size'] == 1
_input1.head()
_input0.head()
_input1['Deck'] = _input1['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
_input0['Deck'] = _input0['Cabin'].apply(lambda x: str(x).split('/')[0] if np.all(pd.notnull(x)) else x)
_input1['Num'] = _input1['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
_input0['Num'] = _input0['Cabin'].apply(lambda x: int(str(x).split('/')[1]) if np.all(pd.notnull(x)) else x)
_input1['Side'] = _input1['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
_input0['Side'] = _input0['Cabin'].apply(lambda x: str(x).split('/')[2] if np.all(pd.notnull(x)) else x)
passengerID_test_list = _input0['PassengerId'].tolist()
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
_input0 = _input0.drop('PassengerId', axis=1, inplace=False)
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input1 = _input1.drop('group', axis=1, inplace=False)
_input0 = _input0.drop('group', axis=1, inplace=False)
_input1 = _input1.drop('passenger_num', axis=1, inplace=False)
_input0 = _input0.drop('passenger_num', axis=1, inplace=False)
_input1 = _input1.drop('group_size', axis=1, inplace=False)
_input0 = _input0.drop('group_size', axis=1, inplace=False)
_input1.head()
_input0.head()
_input1.dtypes
num_feature = list(_input1.select_dtypes(include='number'))
categ_feature = list(_input1.select_dtypes(exclude='number'))
test_categ_feature = list(_input0.select_dtypes(exclude='number'))
print(colored('Numerical features:', 'blue'), num_feature)
print(colored('Categorical features (train):', 'red'), categ_feature)
print(colored('Categorical features (test):', 'red'), test_categ_feature)
for feature in num_feature:
    _input1[feature] = _input1[feature].fillna(_input1[feature].median(), inplace=False)
    _input0[feature] = _input0[feature].fillna(_input0[feature].median(), inplace=False)
for feature in categ_feature:
    _input1[feature] = _input1[feature].fillna(_input1[feature].value_counts().index[0], inplace=False)
for feature in test_categ_feature:
    _input0[feature] = _input0[feature].fillna(_input0[feature].value_counts().index[0], inplace=False)
print(colored('df_train NaN value =', 'green'), _input1.isna().sum().sum())
print(colored('df_test NaN value =', 'green'), _input0.isna().sum().sum())
color = ['#755540', '#f8c1e1', '#f0e4cb', '#f2efe5', '#ed1c24', '#fff01f', '#1f2eff']
fig = plt.figure(figsize=(15, 10))
for (i, col) in enumerate(categ_feature):
    ax = fig.add_subplot(3, 3, i + 1)
    sns.countplot(x=_input1[col], palette=color, ax=ax)
fig.tight_layout()
fig = plt.figure(figsize=(16, 10))
for (i, col) in enumerate(num_feature):
    ax = fig.add_subplot(2, 4, i + 1)
    _input1.groupby(['Transported'])[col].mean().plot(kind='bar', color=['#755540', '#f8c1e1'])
    ax.set_ylabel(col)
fig.tight_layout()
_input1.columns
_input1.dtypes
LABELS = _input0.columns
encoder = LabelEncoder()
for col in LABELS:
    if _input1[col].dtype == 'O':
        _input1[col] = encoder.fit_transform(_input1[col])
        _input0[col] = encoder.transform(_input0[col])
    elif _input1[col].dtype == 'bool':
        _input1[col] = _input1[col].astype('int')
        _input0[col] = _input0[col].astype('int')
_input1['Transported'] = _input1['Transported'].astype('int')
LABELS_MM = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
mm_scaler = MinMaxScaler()
_input1[LABELS_MM] = mm_scaler.fit_transform(_input1[LABELS_MM])
_input0[LABELS_MM] = mm_scaler.transform(_input0[LABELS_MM])
_input1.head()
_input0.head()
x = _input1[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'InGroup', 'Deck', 'Num', 'Side']]
y = _input1['Transported']
x.head()
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=1)
params = {'max_depth': [3, 6], 'gamma': [0, 1], 'learning_rate': [0.01, 0.02, 0.05, 0.1], 'n_estimators': [100, 200, 500, 1000], 'colsample_bytree': [0.3, 0.5, 0.7]}
xgb_grid = GridSearchCV(estimator=XGBClassifier(), param_grid=params, cv=2)
best_params = {'colsample_bytree': 0.7, 'gamma': 1, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200}
xgboost_model = XGBClassifier(colsample_bytree=0.7, gamma=1, learning_rate=0.05, max_depth=6, n_estimators=200)