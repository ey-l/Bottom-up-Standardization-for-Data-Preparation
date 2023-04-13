import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.isna().sum()
_input1.head()
_input1 = _input1.astype({'VIP': 'bool', 'CryoSleep': 'bool'})
_input0 = _input0.astype({'VIP': 'bool', 'CryoSleep': 'bool'})
import missingno as msno
import matplotlib.pyplot as plt
msno.heatmap(_input1)
s = _input1.dtypes == 'object'
col_objects = list(s[s].index)
for col in col_objects:
    print(_input1[col].value_counts())
_input1[['Cabin_A', 'Cabin_B', 'Cabin_C']] = _input1['Cabin'].str.split('/', expand=True)
_input0[['Cabin_A', 'Cabin_B', 'Cabin_C']] = _input0['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop('Cabin', axis=1)
_input0 = _input0.drop('Cabin', axis=1)
print(_input1.dtypes, _input0.dtypes)
s = _input1.dtypes == 'object'
col_objects = list(s[s].index)
_input1[col_objects] = _input1[col_objects].fillna('Unknown')
_input0[col_objects] = _input0[col_objects].fillna('Unknown')
_input1.sample(20)
pd.concat([pd.DataFrame(_input1.isna().sum()), pd.DataFrame(_input1.dtypes)], axis=1)
n = _input1.dtypes == 'float'
col_nums = list(n[n].index)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_num_columns(df):
    (fig, ax) = plt.subplots(1, len(col_nums), figsize=(15, 8), tight_layout=True)
    for (i, col) in enumerate(col_nums):
        ax[i] = sns.violinplot(data=df, y=col, ax=ax[i])
plot_num_columns(_input1)
num_non_age = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

def find_upper_outlier(col):
    Q3 = np.quantile(col, 0.75)
    Q1 = np.quantile(col, 0.25)
    limit_finder = 1.5 * (Q3 - Q1)
    upper_limit = Q3 + limit_finder
    return upper_limit
for col in col_nums:
    median = np.quantile(_input1[col], 0.5)
    _input1[col] = _input1[col].fillna(median)
    _input0[col] = _input0[col].fillna(median)
for col in col_nums:
    median = np.median(_input1[~_input1[col].isna()][col])
    _input1[col] = _input1[col].fillna(median)
    _input0[col] = _input0[col].fillna(median)
print(_input1.isna().sum(), _input0.isna().sum())
sns.countplot(data=_input1, x='Transported')

def obj_is_transported_by_group(df, col):
    grouped = df.groupby(col).Transported.mean().sort_values(ascending=False).head(30)
    (fig, ax) = plt.subplots(figsize=(8, 7))
    ax = grouped.plot(kind='bar', label=col)
    plt.title('Percentage of Transported by ' + col)
    plt.xticks(rotation=45)
    return ax

def num_is_transported_by_group(df, col, choice):
    (fig, ax) = plt.subplots(figsize=(8, 7))
    if choice == 'violin':
        ax = sns.violinplot(data=df, y=col, x='Transported')
    if choice == 'box':
        ax = sns.boxplot(data=df, y=col, x='Transported')
    plt.xticks(rotation=45)
    return ax
for col in col_objects:
    obj_is_transported_by_group(_input1, col)
for col in col_nums:
    num_is_transported_by_group(_input1, col, 'violin')
for col in col_nums:
    num_is_transported_by_group(_input1, col, 'box')
light_pallete = sns.light_palette('steelblue', as_cmap=True)
plt.figure(figsize=(10, 10))
sns.heatmap(_input1.corr(), cmap=light_pallete, annot=True)
_input1.head()
X = _input1.drop(['Transported', 'Name', 'PassengerId'], axis=1)
y = _input1['Transported']
to_predict_X = _input0.drop(['Name', 'PassengerId'], axis=1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
print(X_train.shape, y_train.shape)
ohe = OneHotEncoder()
rs = RobustScaler(quantile_range=(0, 10))
scaler = StandardScaler()
preprocessor = ColumnTransformer(transformers=[('cat', ohe, ['HomePlanet', 'Destination', 'Cabin_A', 'Cabin_C']), ('remove_outlier', rs, num_non_age), ('scaler', scaler, col_nums)])
dt = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', dt)])
params_dt = {'classifier__criterion': ['gini', 'entropy'], 'classifier__min_samples_leaf': [0.04, 0.06, 0.08], 'classifier__max_features': [0.2, 0.4, 0.6, 0.8, 'auto'], 'classifier__max_depth': [3, 4, 5, 6, 7]}
grid_dt = GridSearchCV(estimator=pipeline, param_grid=params_dt, scoring='accuracy', cv=10, n_jobs=-1)